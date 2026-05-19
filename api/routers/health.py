from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    knowledge_base = request.app.state.knowledge_base
    indexed_chunks = await asyncio.to_thread(knowledge_base.count)
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        table_name=settings.table_name,
        indexed_chunks=indexed_chunks,
        gist_enabled=settings.gist_enabled,
    )
