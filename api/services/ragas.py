from __future__ import annotations

import math
from typing import Any

from ..schemas import RagasPreparedSample, RagasSampleRequest
from .retrieval import HybridRAGService

DEFAULT_RESULT_FIELDS = {"user_input", "response", "reference", "retrieved_contexts"}


class RagasDependencyError(RuntimeError):
    pass


class RagasEvaluationService:
    def __init__(self, rag_service: HybridRAGService):
        self.rag_service = rag_service

    async def evaluate_samples(
        self,
        samples: list[RagasSampleRequest],
        auto_generate_response: bool = True,
        include_prepared_samples: bool = True,
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("At least one RAGAS sample is required.")

        prepared_samples = await self._prepare_samples(samples, auto_generate_response)
        result = self._evaluate(prepared_samples)
        rows = self._extract_rows(result)
        metrics = self._summarize_metrics(rows)

        return {
            "status": "ok",
            "sample_count": len(prepared_samples),
            "metrics": metrics,
            "rows": rows,
            "prepared_samples": prepared_samples if include_prepared_samples else None,
        }

    async def _prepare_samples(
        self,
        samples: list[RagasSampleRequest],
        auto_generate_response: bool,
    ) -> list[RagasPreparedSample]:
        prepared: list[RagasPreparedSample] = []

        for sample in samples:
            response = sample.response
            retrieved_contexts = list(sample.retrieved_contexts or [])

            if response is None and auto_generate_response:
                response, chunks, _ = await self.rag_service.answer(
                    sample.query,
                    top_k_docs=sample.top_k_docs,
                    top_k_gist=sample.top_k_gist,
                )
                if not retrieved_contexts:
                    retrieved_contexts = [chunk.text for chunk in chunks]
            elif not retrieved_contexts:
                chunks, _ = await self.rag_service.retrieve(
                    sample.query,
                    top_k_docs=sample.top_k_docs,
                    top_k_gist=sample.top_k_gist,
                )
                retrieved_contexts = [chunk.text for chunk in chunks]

            if response is None:
                raise ValueError(
                    "Each sample needs a response, or auto_generate_response must be true."
                )

            prepared.append(
                RagasPreparedSample(
                    query=sample.query,
                    reference=sample.reference,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )
            )

        return prepared

    def _evaluate(self, samples: list[RagasPreparedSample]) -> Any:
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
        except ImportError as exc:
            raise RagasDependencyError(
                "RAGAS is not installed. Run `uv sync --extra eval` and configure your "
                "RAGAS-compatible evaluator before calling /v1/ragas/test."
            ) from exc

        evaluation_dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=sample.query,
                    response=sample.response,
                    reference=sample.reference,
                    retrieved_contexts=sample.retrieved_contexts,
                )
                for sample in samples
            ]
        )
        return evaluate(
            dataset=evaluation_dataset,
            show_progress=False,
            raise_exceptions=True,
        )

    def _extract_rows(self, result: Any) -> list[dict[str, Any]]:
        if not hasattr(result, "to_pandas"):
            return []

        dataframe = result.to_pandas()
        rows = dataframe.to_dict(orient="records")
        return [self._normalize_row(row) for row in rows]

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            normalized[key] = self._normalize_value(value)
        return normalized

    def _normalize_value(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        return value

    def _summarize_metrics(self, rows: list[dict[str, Any]]) -> dict[str, float | None]:
        summary: dict[str, float | None] = {}
        if not rows:
            return summary

        metric_names = [
            key for key in rows[0] if key not in DEFAULT_RESULT_FIELDS and not key.startswith("_")
        ]
        for name in metric_names:
            values = [
                float(row[name])
                for row in rows
                if isinstance(row.get(name), (int, float))
            ]
            summary[name] = sum(values) / len(values) if values else None
        return summary
