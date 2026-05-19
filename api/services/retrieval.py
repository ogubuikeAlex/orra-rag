from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import lancedb
import tiktoken
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker

from ..config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_DOC_SUFFIXES = {".md", ".mdx", ".txt"}
TEXT_FIELDS = ("question", "answer", "content", "body", "summary", "text", "description")
TITLE_FIELDS = ("title", "question", "name", "heading")
URL_FIELDS = ("url", "link", "source_url", "href")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    kind: str
    title: str
    source: str
    text: str
    url: str | None = None
    score: float | None = None

    def excerpt(self, max_chars: int = 320) -> str:
        normalized = " ".join(self.text.split())
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[: max_chars - 3].rstrip()}..."


def get_embedding_model():
    registry = get_registry()
    return registry.get("gemini-text").create(name="gemini-embedding-001")


def get_document_model():
    embedding_model = get_embedding_model()

    class IndexedDocument(LanceModel):
        id: str
        title: str
        source: str
        kind: str
        text: str = embedding_model.SourceField()
        vector: Vector(embedding_model.ndims()) = embedding_model.VectorField()

    return IndexedDocument


def chunk_text(
    text: str,
    max_tokens: int = 1200,
    overlap_tokens: int = 150,
    encoding_name: str = "cl100k_base",
):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    if not tokens:
        return

    step = max(max_tokens - overlap_tokens, 1)
    for start in range(0, len(tokens), step):
        end = start + max_tokens
        chunk = encoding.decode(tokens[start:end]).strip()
        if chunk:
            yield chunk
        if end >= len(tokens):
            break


def collect_document_chunks(
    docs_dir: Path,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    docs: list[dict[str, Any]] = []
    indexed_files = 0

    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_DOC_SUFFIXES:
            continue

        indexed_files += 1
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        relative_source = file_path.relative_to(docs_dir).as_posix()
        title = file_path.stem.replace("-", " ").replace("_", " ").strip() or relative_source
        for index, chunk in enumerate(
            chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        ):
            docs.append(
                {
                    "id": f"{relative_source}:{index}",
                    "title": title,
                    "source": relative_source,
                    "kind": "doc",
                    "text": chunk,
                }
            )

    return docs, indexed_files


class LanceKnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._db = None
        self._table = None

    def _connect(self):
        self.settings.ensure_directories()
        if self._db is None:
            self._db = lancedb.connect(str(self.settings.db_dir))
        return self._db

    def open_table(self):
        if self._table is not None:
            return self._table

        db = self._connect()
        try:
            self._table = db.open_table(self.settings.table_name)
        except Exception:
            if not self.settings.google_api_key:
                logger.warning(
                    "Skipping LanceDB initialization because GOOGLE_API_KEY is not configured."
                )
                return None
            logger.info("LanceDB table %s not found; rebuilding index.", self.settings.table_name)
            self.rebuild_index()
        return self._table

    def ensure_index(self) -> None:
        self.open_table()

    def rebuild_index(self) -> dict[str, int | str]:
        if not self.settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required to build the LanceDB index.")

        db = self._connect()
        db.drop_table(self.settings.table_name, ignore_missing=True)
        table = db.create_table(
            self.settings.table_name,
            schema=get_document_model(),
            mode="overwrite",
        )
        table.create_fts_index("text", replace=True)

        docs, indexed_files = collect_document_chunks(
            self.settings.docs_dir,
            max_tokens=self.settings.chunk_size_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if docs:
            table.add(docs)
        self._table = table

        return {
            "indexed_files": indexed_files,
            "indexed_chunks": len(docs),
            "table_name": self.settings.table_name,
            "db_path": str(self.settings.db_dir),
        }

    def search(
        self,
        query: str,
        limit: int | None = None,
        query_type: str | None = None,
        reranker_weight: float | None = None,
    ) -> list[RetrievedChunk]:
        table = self.open_table()
        if table is None:
            return []

        effective_query_type = query_type or self.settings.retrieval_query_type
        if not self.settings.google_api_key and effective_query_type == "hybrid":
            effective_query_type = "fts"

        reranker = LinearCombinationReranker(
            weight=reranker_weight if reranker_weight is not None else self.settings.reranker_weight
        )
        results = (
            table.search(
                query=query,
                query_type=effective_query_type,
            )
            .rerank(reranker=reranker)
            .limit(limit or self.settings.retrieval_limit)
            .to_list()
        )

        chunks: list[RetrievedChunk] = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    id=str(result["id"]),
                    kind=str(result.get("kind", "doc")),
                    title=str(result.get("title", result["id"])),
                    source=str(result.get("source", "docs")),
                    text=str(result.get("text", "")),
                    score=_extract_score(result),
                )
            )
        return chunks

    def count(self) -> int:
        try:
            table = self.open_table()
            if table is None:
                return 0
            return int(table.count_rows())
        except Exception:
            return 0


class GistKnowledgeService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: Any = None
        self._cache_expires_at: datetime | None = None

    async def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        if not self.settings.gist_enabled:
            return []

        payload = await self._get_payload()
        records = list(self._iter_records(payload))
        if not records:
            return []

        query_tokens = _tokenize(query)
        scored_chunks: list[RetrievedChunk] = []

        for index, record in enumerate(records):
            text = _extract_record_text(record)
            if not text:
                continue

            score = _score_record(query_tokens, record, text)
            if score <= 0:
                continue

            title = _extract_first(record, TITLE_FIELDS) or f"Gist entry {index + 1}"
            url = _extract_first(record, URL_FIELDS) or self.settings.gist_raw_url
            scored_chunks.append(
                RetrievedChunk(
                    id=str(record.get("id", f"gist:{index}")),
                    kind="gist",
                    title=title,
                    source="github-gist",
                    text=text,
                    url=url,
                    score=float(score),
                )
            )

        scored_chunks.sort(key=lambda item: item.score or 0, reverse=True)
        return scored_chunks[:limit]

    async def _get_payload(self) -> Any:
        now = datetime.now(UTC)
        if self._cache is not None and self._cache_expires_at and now < self._cache_expires_at:
            return self._cache

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.get(self.settings.gist_raw_url)
            response.raise_for_status()
            payload = response.json()

        self._cache = payload
        self._cache_expires_at = now + timedelta(seconds=self.settings.gist_cache_ttl_seconds)
        return payload

    def _iter_records(self, payload: Any):
        container = _resolve_records_container(payload, self.settings.gist_records_path)
        yield from _walk_records(container)


class HybridRAGService:
    def __init__(
        self,
        settings: Settings,
        knowledge_base: LanceKnowledgeBase,
        gist_service: GistKnowledgeService,
    ):
        self.settings = settings
        self.knowledge_base = knowledge_base
        self.gist_service = gist_service

    async def retrieve(
        self,
        query: str,
        top_k_docs: int,
        top_k_gist: int,
    ) -> tuple[list[RetrievedChunk], str]:
        doc_task = asyncio.to_thread(self.knowledge_base.search, query, top_k_docs)
        gist_task = self.gist_service.search(query, top_k_gist)
        doc_chunks, gist_chunks = await asyncio.gather(doc_task, gist_task)
        all_chunks = [*doc_chunks, *gist_chunks]
        return all_chunks, format_context(all_chunks)

    async def answer(
        self,
        query: str,
        top_k_docs: int,
        top_k_gist: int,
    ) -> tuple[str, list[RetrievedChunk], str]:
        chunks, context = await self.retrieve(query, top_k_docs=top_k_docs, top_k_gist=top_k_gist)
        if not chunks:
            return (
                "I could not find any matching ORRA documentation or FAQ entries for that question.",
                [],
                "",
            )

        if not self.settings.google_api_key:
            fallback = (
                "Relevant ORRA context was found, but answer generation is disabled because "
                "GOOGLE_API_KEY is not configured."
            )
            return fallback, chunks, context

        prompt = _build_prompt(query, context)
        answer = await self._generate_answer(prompt)
        return answer, chunks, context

    async def _generate_answer(self, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.google_generation_model}:generateContent"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            response = await client.post(
                url,
                params={"key": self.settings.google_api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        texts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    texts.append(text.strip())

        if not texts:
            raise ValueError("Google model response did not include a text answer.")

        return "\n".join(texts).strip()


def format_context(chunks: list[RetrievedChunk]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[{index}] {chunk.title}",
                    f"Kind: {chunk.kind}",
                    f"Source: {chunk.source}",
                    f"Content: {chunk.text.strip()}",
                ]
            )
        )
    return "\n\n".join(sections)


def _build_prompt(query: str, context: str) -> str:
    return (
        "You are the ORRA documentation assistant.\n"
        "Answer only from the supplied context.\n"
        "If the context is incomplete, say you are unsure instead of inventing details.\n"
        "Prefer concise, product-facing answers suitable for a frontend help experience.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context}"
    )


def _resolve_records_container(payload: Any, path: str | None) -> Any:
    if not path:
        return payload

    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return []
        else:
            return []
    return current


def _walk_records(node: Any):
    if isinstance(node, list):
        for item in node:
            yield from _walk_records(item)
        return

    if isinstance(node, dict):
        if _extract_record_text(node):
            yield node
            return
        for value in node.values():
            yield from _walk_records(value)


def _extract_record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    title = _extract_first(record, TITLE_FIELDS)
    if title:
        parts.append(title)

    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    return "\n".join(parts).strip()


def _extract_first(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _tokenize(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


def _score_record(query_tokens: set[str], record: dict[str, Any], text: str) -> int:
    text_tokens = _tokenize(text)
    overlap = len(query_tokens & text_tokens)
    if overlap == 0:
        return 0

    score = overlap * 10
    title = (_extract_first(record, TITLE_FIELDS) or "").lower()
    for token in query_tokens:
        if token in title:
            score += 5
    return score


def _extract_score(result: dict[str, Any]) -> float | None:
    for key in ("_relevance_score", "_score", "_distance"):
        value = result.get(key)
        if isinstance(value, (float, int)):
            return float(value)
    return None
