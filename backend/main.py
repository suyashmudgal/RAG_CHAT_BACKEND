"""FastAPI application entry-point.

Registers routers, CORS, exception handlers, and startup initialization.
"""

from __future__ import annotations

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

    # Check Groq key
    if not settings.groq_api_key:
        logger.warning("⚠️  GROQ_API_KEY is not set — chat will NOT work.")
    else:
        logger.info("✅ Groq API key configured (model: %s)", settings.groq_model)

    # Eagerly initialise the vector store + embedding model (downloads ~90 MB on first run)
    from services.deps import get_vector_store

    logger.info("Loading embedding model & vector store …")
    get_vector_store()
    logger.info("✅ Vector store and embedding model ready")

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
from routes.documents import router as documents_router  # noqa: E402
from routes.upload import router as upload_router  # noqa: E402

app.include_router(auth_router, tags=["Auth"])
app.include_router(upload_router, tags=["Upload"])
app.include_router(documents_router, tags=["Documents"])
app.include_router(chat_router, tags=["Chat"])


@app.get("/", tags=["Health"])
async def health_check():
    """Simple health-check endpoint."""
    return {"status": "healthy", "service": "DocChat AI Backend", "version": "1.0.0"}
