"""FastAPI application entry-point.

Registers routers, CORS, exception handlers, and startup initialization.
"""

from __future__ import annotations

# Fix Windows C-runtime DLL collision between PyTorch and PyArrow (0xC0000005)
# If pyarrow is present, importing it before torch prevents an access violation crash on Windows.
try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run startup / shutdown logic."""
    logger.info("🚀 Starting DocChat AI Backend")

    # Initialize auth database
    from auth.database import init_db

    await init_db()

    # Check Supabase PostgreSQL connectivity
    if settings.database_url:
        try:
            from database.session import ping_db

            if ping_db():
                logger.info("✅ Supabase PostgreSQL connected successfully")
            else:
                logger.warning("⚠️  Could not connect to Supabase PostgreSQL")
        except Exception as exc:
            logger.warning("⚠️  Error connecting to Supabase PostgreSQL: %s", exc)

    # Check Groq key
    if not settings.groq_api_key:
        logger.warning("⚠️  GROQ_API_KEY is not set — chat will NOT work.")
    else:
        logger.info("✅ Groq API key configured (model: %s)", settings.groq_model)

    # Eagerly initialise the vector store + embedding model (downloads ~90 MB on first run)
    from services.deps import get_vector_store

    try:
        logger.info("Loading embedding model & vector store …")
        get_vector_store()
        logger.info("✅ Vector store and embedding model ready")
    except Exception as exc:
        logger.critical(
            "❌ Failed to initialize vector store and embedding model: %s",
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"DocChat AI startup failed during embedding model / vector store initialization: {exc}"
        ) from exc

    yield  # ---- application is running ----

    logger.info("👋 Shutting down DocChat AI Backend")


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocChat AI — RAG Document Chatbot",
    description="Upload documents, ask questions, get grounded answers with source citations.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handler ───────────────────────────────────────────────


@app.exception_handler(Exception)
async def _global_exception_handler(_request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )


# ── Routers ─────────────────────────────────────────────────────────────────

from auth.router import router as auth_router  # noqa: E402
from routes.chat import router as chat_router  # noqa: E402
from routes.conversations import router as conversations_router  # noqa: E402
from routes.documents import router as documents_router  # noqa: E402
from routes.upload import router as upload_router  # noqa: E402

app.include_router(auth_router, tags=["Auth"])
app.include_router(upload_router, tags=["Upload"])
app.include_router(documents_router, tags=["Documents"])
app.include_router(conversations_router, tags=["Conversations"])
app.include_router(chat_router, tags=["Chat"])


@app.get("/", tags=["Health"])
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "healthy", "service": "DocChat AI Backend", "version": "1.0.0"}
