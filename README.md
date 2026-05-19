# ORRA RAG API

`orra-rag` is a A RAGAS-evaluated enterprise hybrid RAG system built for orra. Its exposed via a FastAPI RAG service for querying ORRA docs and FAQs.

> I built this to help improve how users interact with [ORRA](https://orra.xyz/) documentation and FAQ retrieval but you can absolutely use it for your own projects

It combines:

- local docs in [`docs`](./docs)
- LanceDB hybrid search for indexed documentation
- live FAQ/context retrieval from a GitHub Gist JSON payload
- optional Gemini answer generation for frontend-facing responses

## API shape

- `GET /health`
- `GET /v1/health`
- `POST /v1/retrieve`
- `POST /v1/query`
- `POST /v1/index/rebuild`
- `POST /v1/ragas/test`

## Project structure

The live API now lives entirely under [`api`](./api):

- `api/main.py`
- `api/routers/`
- `api/services/`
- `api/schemas.py`
- `api/config.py`

File-level documentation for the runtime path is in:

- [`docs/fastapi-runtime-files.md`](./docs/fastapi-runtime-files.md)
- [`docs/api-file-guide.md`](./docs/api-file-guide.md)

When a query is made from the ORRA UI search, it calls `POST /v1/query` with a JSON body like:

```json
{
  "query": "How does ORRA work?",
  "top_k_docs": 5,
  "top_k_gist": 5,
  "include_context": false
}
```

## Environment

You can adapt this to your own project.

Create a local `.env` file from `.env.example`.

Required for full functionality:

- `GOOGLE_API_KEY`
- `APP_GIST_RAW_URL`

Optional:

- `APP_GIST_RECORDS_PATH`
- `ALLOWED_ORIGINS`
- `PORT`

## Run with uv

Install dependencies:

```bash
uv sync
```

Install the optional RAGAS dependency when you want to use the evaluation endpoint:

```bash
uv sync --extra eval
```

Start the API:

```bash
uv run orra-rag
```

Or with Uvicorn directly:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Indexing

At startup, the service creates the LanceDB index from files inside `docs/` if the table does not exist.

To force a rebuild:

```bash
curl -X POST http://localhost:8080/v1/index/rebuild
```

## RAGAS testing

`POST /v1/ragas/test` evaluates one or more samples against the live pipeline.

Example:

```json
{
  "samples": [
    {
      "query": "How does ORRA work?",
      "reference": "ORRA coordinates AI agents and tools through workflows.",
      "top_k_docs": 5,
      "top_k_gist": 5
    }
  ],
  "auto_generate_response": true,
  "include_prepared_samples": true
}
```

## Notes

- If `GOOGLE_API_KEY` is missing, retrieval still works but answer generation returns a configuration warning.
- `POST /v1/ragas/test` returns `503` until the optional `eval` extra is installed.
