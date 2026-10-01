# RideNear AI service

Python service for vehicle ranking and rental-policy retrieval. Rank and RAG are part of this service.

The Section 6 demo script, demo account, and `gpt-4o-mini` requirement live in the API README: [../ridenear-api/README.md](../ridenear-api/README.md). Hosting is not done yet. Deployment notes are in [../ridenear-api/docs/deployment.md](../ridenear-api/docs/deployment.md).

## Health

Unauthenticated `GET /health` returns a JSON body whose `status` is `ok` when the service is up. The API treats `aiService` as `up` only when that body has `status: ok`. The response also includes `vectorCount`.

## Environment

Copy `.env.example` to `.env`. Set values locally. Do not commit `.env`. `AI_SERVICE_TOKEN` must match the API.

| Variable | Role |
| --- | --- |
| `CHROMA_PATH` | `CHROMA_PATH=/data/chroma` is for the container. A local run should use a writable folder such as `./data/chroma` |
| `AI_SERVICE_TOKEN` | Shared token required on rank and RAG routes |
| `LLM_BASE_URL` | OpenAI-compatible API origin |
| `LLM_API_KEY` | Optional. Empty, or a key the provider rejects, answers from the retrieved guide text. A key the provider accepts phrases that text with `gpt-4o-mini` |
| `LLM_MODEL` | Must be `gpt-4o-mini` for the chat steps |
| `RAG_MIN_SCORE` | Minimum retrieval score for policy chunks |

## Local run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, set `AI_SERVICE_TOKEN` to the same value as the API, then:

```bash
uvicorn app.main:app --reload --port 8000
```

Confirm `GET http://localhost:8000/health` returns `status: ok` before the API health check. Do not start Section 6 while the API reports `aiService` as `down`.
