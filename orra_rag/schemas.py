# from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="User question about ORRA.")
    top_k_docs: int = Field(
        default=5, ge=1, le=20, description="Max chunks to retrieve from the indexed docs.")
    top_k_gist: int = Field(
        default=5, ge=1, le=20, description="Max entries to retrieve from the FAQ gist.")
    include_context: bool = Field(
        default=False, description="Return the raw assembled context alongside the answer")


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k_docs: int = Field(default=5, ge=1, le=20)
    top_k_gist: int = Field(default=5, ge=1, le=20)


class SourceExcerpt(BaseModel):
    id: str
    kind: Literal["doc", "gist"]
    title: str
    source: str
    excerpt: str
    url: str | None = None
    score: float | None = None


class RetrievalResponse(BaseModel):
    query: str
    sources: list[SourceExcerpt]
    context: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceExcerpt]
    context: str | None = None


class IndexRebuildResponse(BaseModel):
    status: str
    indexed_files: int
    indexed_chunks: int
    table_name: str
    db_path: str


class HealthResponse(BaseModel):
    status: str
    environment: str
    table_name: str
    indexed_chunks: int
    gist_enabled: bool
