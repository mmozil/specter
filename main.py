"""Murdock — Especialista tributário brasileiro com IA."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.core.config import settings
from src.core.database import async_session, init_db
from src.api.routes import router
from src.api.llm_routes import router as llm_router

logging.basicConfig(
    level=logging.INFO if settings.ENVIRONMENT == "production" else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown do Murdock."""
    logger.info("═══ MURDOCK v%s inicializando ═══", settings.APP_VERSION)

    # Criar extensões + tabelas
    await init_db()
    logger.info("Database inicializado (pgvector + pg_trgm)")

    # Seed dos provedores de LLM a partir do ambiente (idempotente — só no 1º boot)
    try:
        from src.services.llm_config import seed_from_env_if_empty

        async with async_session() as db:
            await seed_from_env_if_empty(db)
    except Exception as e:  # noqa: BLE001
        logger.warning("Seed de provedores de LLM falhou (seguindo sem): %s", e)

    yield

    logger.info("═══ MURDOCK encerrado ═══")


app = FastAPI(
    title="Murdock — Especialista Tributário",
    description="Agente IA especializado em direito tributário, contábil e fiscal brasileiro. "
                "Fontes exclusivamente oficiais (gov.br, jus.br).",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENVIRONMENT == "development" else [
        "https://murdock.hovio.com.br",
        "https://hovio.com.br",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")
app.include_router(llm_router, prefix="/api")

# Static files (frontend)
app.mount("/static", StaticFiles(directory="src/static"), name="static")


@app.get("/")
async def root():
    """Serve a UI do chat (sem cache para sempre pegar a versão mais recente)."""
    return FileResponse(
        "src/static/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
    )
