import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

import app.rag as rag
from app.main import app
from app.rag import (
    compose_answer,
    create_embedding_function,
    load_chunks,
    passes_score_gate,
)
from tests.fake_embeddings import HashEmbedding

FUEL_AND_LATE = "What is the fuel policy and a late return?"
CHAUFFEUR = "Can I book a chauffeur?"


class GuideTextTests(unittest.TestCase):
    def test_fifteen_sentences_match_the_prd(self):
        chunks = load_chunks()
        self.assertEqual(
            [chunk["id"] for chunk in chunks],
            [
                "cancel-1",
                "cancel-2",
                "cancel-3",
                "late-1",
                "late-2",
                "late-3",
                "ins-1",
                "ins-2",
                "ins-3",
                "deposit-1",
                "deposit-2",
                "deposit-3",
                "fuel-1",
                "fuel-2",
                "fuel-3",
            ],
        )
        self.assertEqual(
            [chunk["document"] for chunk in chunks],
            ["Cancellation"] * 3
            + ["Late return"] * 3
            + ["Insurance"] * 3
            + ["Security deposit"] * 3
            + ["Fuel policy"] * 3,
        )
        prd = (Path(__file__).resolve().parents[2] / "PRD.md").read_text(encoding="utf-8")
        for chunk in chunks:
            self.assertIn(chunk["text"], prd)


class ScoreGateTests(unittest.TestCase):
    def test_exact_minimum_is_grounded_and_below_is_not(self):
        self.assertTrue(passes_score_gate(0.35, 0.35))
        self.assertFalse(passes_score_gate(0.349, 0.35))

        calls: list[list[str]] = []

        def complete(_query: str, excerpts: list[str]) -> str:
            calls.append(excerpts)
            return "Use the guide."

        hit = {
            "document": "Fuel policy",
            "chunkId": "fuel-1",
            "score": 0.35,
            "text": "Vehicles are handed over with a full tank.",
        }
        grounded = compose_answer("fuel", [hit], score_floor=0.35, complete=complete)
        self.assertTrue(grounded["grounded"])
        self.assertEqual(grounded["answer"], "Use the guide.")
        self.assertEqual(grounded["citations"], [
            {"document": "Fuel policy", "chunkId": "fuel-1", "score": 0.35},
        ])
        self.assertEqual(calls, [["Vehicles are handed over with a full tank."]])

        missed = compose_answer(
            "chauffeur",
            [{**hit, "score": 0.349, "text": "Vehicles are handed over with a full tank."}],
            score_floor=0.35,
            complete=complete,
        )
        self.assertFalse(missed["grounded"])
        self.assertEqual(missed["answer"], "")
        self.assertEqual(missed["citations"], [])
        self.assertEqual(len(calls), 1)
        self.assertNotIn("full tank", str(missed))

    def test_unset_and_invalid_min_score_use_the_default_gate(self):
        with mock.patch.dict(os.environ, {"RAG_MIN_SCORE": "not-a-float"}):
            self.assertEqual(rag.min_score(), 0.35)

        unset = {key: value for key, value in os.environ.items() if key != "RAG_MIN_SCORE"}
        with mock.patch.dict(os.environ, unset, clear=True):
            self.assertNotIn("RAG_MIN_SCORE", os.environ)
            self.assertEqual(rag.min_score(), 0.35)
            index = rag.PolicyIndex(None, None, [])
            calls: list[list[str]] = []

            def complete(_query: str, excerpts: list[str]) -> str:
                calls.append(excerpts)
                return "Use the guide."

            previous = rag._answer_fn
            rag.set_answer_fn(complete)
            try:
                hit = {
                    "document": "Fuel policy",
                    "chunkId": "fuel-1",
                    "score": 0.35,
                    "text": "Vehicles are handed over with a full tank.",
                }
                with mock.patch("app.rag.retrieve", return_value=[hit]):
                    grounded = index.answer("fuel")
                self.assertTrue(grounded["grounded"])
                self.assertEqual(grounded["answer"], "Use the guide.")
                with mock.patch(
                    "app.rag.retrieve",
                    return_value=[{**hit, "score": 0.349}],
                ):
                    missed = index.answer("chauffeur")
                self.assertFalse(missed["grounded"])
                self.assertEqual(missed["answer"], "")
                self.assertEqual(missed["citations"], [])
                self.assertEqual(len(calls), 1)
                self.assertNotIn("full tank", str(missed))
            finally:
                rag.set_answer_fn(previous)


class AnswerModelTests(unittest.TestCase):
    def test_empty_tool_list_and_only_the_excerpts(self):
        captured: dict = {}

        class Response:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "From the guide."}}]}

        def post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return Response()

        with mock.patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "http://llm.internal/v1",
                "LLM_API_KEY": "sentinel-llm-key",
                "LLM_MODEL": "gpt-4o-mini",
            },
        ):
            with mock.patch("app.rag.httpx.post", post):
                prose = rag.complete_answer("fuel?", ["chunk a", "chunk b"])

        self.assertEqual(prose, "From the guide.")
        self.assertEqual(captured["url"], "http://llm.internal/v1/chat/completions")
        self.assertEqual(captured["json"]["model"], "gpt-4o-mini")
        self.assertEqual(captured["json"]["temperature"], 0.2)
        self.assertEqual(captured["json"]["tools"], [])
        content = captured["json"]["messages"][1]["content"]
        self.assertIn("chunk a", content)
        self.assertIn("chunk b", content)
        self.assertNotIn("one day of rental would be charged", content)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ,
            {
                "AI_SERVICE_TOKEN": "rag-secret",
                "CHROMA_PATH": self._tmp.name,
                "RAG_MIN_SCORE": "0.35",
            },
        )
        self._env.start()
        self._embedding = rag._embedding_factory
        self._answer = rag._answer_fn
        rag.set_embedding_factory(lambda: HashEmbedding())
        rag.set_answer_fn(self._fail_if_called)
        self.index = rag.build_index()

    def tearDown(self):
        self.index.close()
        rag.set_embedding_factory(self._embedding)
        rag.set_answer_fn(self._answer)
        self._env.stop()
        self._tmp.cleanup()

    @staticmethod
    def _fail_if_called(_query: str, _excerpts: list[str]) -> str:
        raise AssertionError("answer model should not be called")

    def test_second_ingest_stays_at_fifteen(self):
        self.assertEqual(self.index.vector_count(), 15)
        self.index.ingest()
        self.assertEqual(self.index.vector_count(), 15)
        again = rag.build_index()
        try:
            self.assertEqual(again.vector_count(), 15)
        finally:
            again.close()

    def test_top_three_include_fuel_and_late_return(self):
        seen: dict[str, list[str]] = {}

        def complete(_query: str, excerpts: list[str]) -> str:
            seen["excerpts"] = excerpts
            return "Return it full. A late return costs one extra day."

        rag.set_answer_fn(complete)
        result = self.index.answer(FUEL_AND_LATE)
        self.assertTrue(result["grounded"])
        self.assertEqual(result["answer"], "Return it full. A late return costs one extra day.")
        citations = result["citations"]
        self.assertEqual(len(citations), 3)
        documents = [item["document"] for item in citations]
        self.assertIn("Fuel policy", documents)
        self.assertIn("Late return", documents)
        self.assertEqual(len(seen["excerpts"]), 3)
        known = {chunk["text"] for chunk in load_chunks()}
        self.assertTrue(all(excerpt in known for excerpt in seen["excerpts"]))
        joined = "\n".join(seen["excerpts"])
        for citation in citations:
            self.assertNotIn("text", citation)
        self.assertIn("full tank", joined)
        self.assertIn("late", joined.lower())

    def test_chauffeur_is_ungrounded_without_the_model(self):
        result = self.index.answer(CHAUFFEUR)
        self.assertFalse(result["grounded"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])
        self.assertNotIn("full tank", str(result))


class RagHttpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ,
            {
                "AI_SERVICE_TOKEN": "rag-secret",
                "CHROMA_PATH": self._tmp.name,
                "RAG_MIN_SCORE": "0.35",
            },
        )
        self._env.start()
        self._embedding = rag._embedding_factory
        self._answer = rag._answer_fn
        rag.set_embedding_factory(lambda: HashEmbedding())
        rag.set_answer_fn(lambda _query, _excerpts: "Guide prose.")
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        rag.set_embedding_factory(self._embedding)
        rag.set_answer_fn(self._answer)
        self._env.stop()
        self._tmp.cleanup()

    def test_missing_or_wrong_token_is_401_and_health_stays_open(self):
        body = {"query": CHAUFFEUR}
        missing = self.client.post("/rag/query", json=body)
        wrong = self.client.post(
            "/rag/query",
            json=body,
            headers={"X-Service-Token": "rag-secreX"},
        )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.json()["vectorCount"], 15)

    def test_query_returns_grounded_answer(self):
        response = self.client.post(
            "/rag/query",
            json={"query": FUEL_AND_LATE},
            headers={"X-Service-Token": "rag-secret"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["grounded"])
        self.assertEqual(body["answer"], "Guide prose.")
        self.assertLessEqual(len(body["citations"]), 3)
        self.assertGreaterEqual(body["citations"][0]["score"], 0.35)


class AnswerFailureHttpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ,
            {
                "AI_SERVICE_TOKEN": "rag-secret",
                "CHROMA_PATH": self._tmp.name,
                "RAG_MIN_SCORE": "0.35",
            },
        )
        self._env.start()
        self._embedding = rag._embedding_factory
        rag.set_embedding_factory(lambda: HashEmbedding())
        self.assertIs(rag._answer_fn, rag.complete_answer)
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        rag.set_embedding_factory(self._embedding)
        self._env.stop()
        self._tmp.cleanup()

    def _query(self):
        return self.client.post(
            "/rag/query",
            json={"query": FUEL_AND_LATE},
            headers={"X-Service-Token": "rag-secret"},
        )

    def _assert_unavailable(self, response):
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Answer model unavailable")
        self.assertNotIn("full tank", response.text)

    def test_missing_api_key_is_unavailable(self):
        with mock.patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://llm.internal/v1", "LLM_API_KEY": ""},
        ):
            response = self._query()
        self._assert_unavailable(response)

    def test_transport_error_is_unavailable(self):
        def post(*_args, **_kwargs):
            raise rag.httpx.ConnectError("Vehicles are handed over with a full tank.")

        with mock.patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://llm.internal/v1", "LLM_API_KEY": "sentinel-llm-key"},
        ):
            with mock.patch("app.rag.httpx.post", post):
                response = self._query()
        self._assert_unavailable(response)

    def test_non_200_model_response_is_unavailable(self):
        class Response:
            status_code = 503
            text = "Vehicles are handed over with a full tank."

            def json(self):
                return {"error": "Vehicles are handed over with a full tank."}

        with mock.patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://llm.internal/v1", "LLM_API_KEY": "sentinel-llm-key"},
        ):
            with mock.patch("app.rag.httpx.post", return_value=Response()):
                response = self._query()
        self._assert_unavailable(response)

    def test_non_string_message_is_unavailable(self):
        class Response:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": {
                                    "text": "Vehicles are handed over with a full tank.",
                                },
                            },
                        },
                    ],
                }

        with mock.patch.dict(
            os.environ,
            {"LLM_BASE_URL": "http://llm.internal/v1", "LLM_API_KEY": "sentinel-llm-key"},
        ):
            with mock.patch("app.rag.httpx.post", return_value=Response()):
                response = self._query()
        self._assert_unavailable(response)


class FactoryTests(unittest.TestCase):
    def test_production_factory_names_minilm(self):
        with mock.patch(
            "chromadb.utils.embedding_functions.ONNXMiniLM_L6_V2"
        ) as constructor:
            constructor.return_value = object()
            create_embedding_function()
        constructor.assert_called_once_with()
