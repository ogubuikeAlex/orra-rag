from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, HTTPException, Request

from ..schemas import (
    IndexRebuildResponse,
    QueryRequest,
    QueryResponse,
    RagasTestRequest,
    RagasTestResponse,
    RetrievalRequest,
    RetrievalResponse,
    SourceExcerpt,
)
from ..services.ragas import RagasDependencyError

router = APIRouter()


@router.post("/retrieve", response_model=RetrievalResponse, tags=["rag"])
async def retrieve_context(payload: RetrievalRequest, request: Request) -> RetrievalResponse:
    rag_service = request.app.state.rag_service
    try:
        chunks, context = await rag_service.retrieve(
            payload.query,
            top_k_docs=payload.top_k_docs,
            top_k_gist=payload.top_k_gist,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RetrievalResponse(
        query=payload.query,
        sources=[_to_source_excerpt(chunk) for chunk in chunks],
        context=context,
    )


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_orra(payload: QueryRequest, request: Request) -> QueryResponse:
    rag_service = request.app.state.rag_service
    try:
        answer, chunks, context = await rag_service.answer(
            payload.query,
            top_k_docs=payload.top_k_docs,
            top_k_gist=payload.top_k_gist,
        )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        query=payload.query,
        answer=answer,
        sources=[_to_source_excerpt(chunk) for chunk in chunks],
        context=context if payload.include_context else None,
    )


@router.post("/index/rebuild", response_model=IndexRebuildResponse, tags=["index"])
async def rebuild_index(request: Request) -> IndexRebuildResponse:
    knowledge_base = request.app.state.knowledge_base
    try:
        summary = await asyncio.to_thread(knowledge_base.rebuild_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IndexRebuildResponse(status="ok", **summary)


@router.post("/ragas/test", response_model=RagasTestResponse, tags=["ragas"])
async def test_ragas(payload: RagasTestRequest, request: Request) -> RagasTestResponse:
    ragas_service = request.app.state.ragas_service
    try:
        result = await ragas_service.evaluate_samples(
            payload.samples,
            auto_generate_response=payload.auto_generate_response,
            include_prepared_samples=payload.include_prepared_samples,
        )
    except RagasDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return RagasTestResponse(**result)


def _to_source_excerpt(chunk) -> SourceExcerpt:
    return SourceExcerpt(
        id=chunk.id,
        kind=chunk.kind,
        title=chunk.title,
        source=chunk.source,
        excerpt=chunk.excerpt(),
        url=chunk.url,
        score=chunk.score,
    )
