"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from .env file."""

    # --- Groq API ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # --- Embedding (local, free) ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_db"

    # --- File uploads ---
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 200

    # --- RAG & Retrieval ---
    top_k_results: int = 6
    retrieval_top_k: int = 6
    similarity_threshold: float = 0.20
    citation_margin: float = 0.25
    max_citations: int = 4

    # --- Authentication ---
    jwt_secret_key: str = "change-me-to-a-random-secret"
    jwt_expiry_hours: int = 72
    auth_db_path: str = "./auth.db"

    # --- Google OAuth ---
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    frontend_url: str = "http://localhost:5173"

    # --- Database (Supabase PostgreSQL) ---
    database_url: str = ""

    # --- Server ---
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @property
    def sync_database_url(self) -> str:
        """Return a SQLAlchemy-compatible synchronous connection string."""
        url = self.database_url.strip()
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url


settings = Settings()
