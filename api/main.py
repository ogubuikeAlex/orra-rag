from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .routers.health import router as health_router
from .routers.rag import router as rag_router
from .services.ragas import RagasEvaluationService
from .services.retrieval import GistKnowledgeService, HybridRAGService, LanceKnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    knowledge_base = LanceKnowledgeBase(settings)
    gist_service = GistKnowledgeService(settings)
    rag_service = HybridRAGService(settings, knowledge_base, gist_service)
    ragas_service = RagasEvaluationService(rag_service)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.knowledge_base = knowledge_base
        app.state.gist_service = gist_service
        app.state.rag_service = rag_service
        app.state.ragas_service = ragas_service
        if settings.auto_index_on_startup:
            await asyncio.to_thread(knowledge_base.ensure_index)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(rag_router, prefix=settings.api_prefix)
    return app


app = create_app()
