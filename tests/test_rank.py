import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app.rag as rag
from app.main import app
from app.rank import rank_vehicles
from tests.fake_embeddings import HashEmbedding


def vehicle(
    id: str,
    distance_km: float,
    price: float,
    kind: str = "suv",
    transmission: str = "automatic",
) -> dict:
    return {
        "id": id,
        "distanceKm": distance_km,
        "pricePerDay": price,
        "type": kind,
        "transmission": transmission,
    }


class RankFormulaTests(unittest.TestCase):
    def test_weights_and_three_decimals(self):
        # distance 1 km / radius 3 → 1 - 1/3
        # 0.40 * (2/3) + 0.25 + 0.20 + 0.15 = 0.8666... → 0.867
        ranked = rank_vehicles(
            radius_km=3,
            vehicle_type=None,
            transmission=None,
            max_price_per_day=None,
            vehicles=[vehicle("only", 1, 1000, "sedan", "manual")],
        )
        self.assertEqual(ranked[0]["rankScore"], 0.867)
        self.assertGreaterEqual(ranked[0]["rankScore"], 0)
        self.assertLessEqual(ranked[0]["rankScore"], 1)

    def test_floors_distance_and_price(self):
        over_radius = rank_vehicles(
            radius_km=10,
            vehicle_type="suv",
            transmission="automatic",
            max_price_per_day=50,
            vehicles=[vehicle("far", 20, 100, "sedan", "manual")],
        )
        self.assertEqual(over_radius[0]["rankScore"], 0)

        over_price = rank_vehicles(
            radius_km=10,
            vehicle_type="suv",
            transmission="automatic",
            max_price_per_day=50,
            vehicles=[vehicle("pricey", 0, 100)],
        )
        # distance 1, type 1, transmission 1, price floored to 0
        self.assertEqual(over_price[0]["rankScore"], 0.75)

    def test_max_price_zero(self):
        free = rank_vehicles(
            radius_km=10,
            vehicle_type=None,
            transmission=None,
            max_price_per_day=0,
            vehicles=[vehicle("free", 0, 0)],
        )
        self.assertEqual(free[0]["rankScore"], 1)

        paid = rank_vehicles(
            radius_km=10,
            vehicle_type=None,
            transmission=None,
            max_price_per_day=0,
            vehicles=[vehicle("paid", 0, 10)],
        )
        self.assertEqual(paid[0]["rankScore"], 0.75)

    def test_omitted_type_scores_as_match(self):
        omitted = rank_vehicles(
            radius_km=10,
            vehicle_type=None,
            transmission=None,
            max_price_per_day=None,
            vehicles=[vehicle("sedan", 0, 2200, "sedan", "manual")],
        )
        requested = rank_vehicles(
            radius_km=10,
            vehicle_type="suv",
            transmission=None,
            max_price_per_day=None,
            vehicles=[vehicle("sedan", 0, 2200, "sedan", "manual")],
        )
        self.assertEqual(omitted[0]["rankScore"], 1)
        self.assertEqual(requested[0]["rankScore"], 0.8)
        self.assertNotIn("SUV", requested[0]["reason"])
        self.assertNotIn("sedan", requested[0]["reason"])

    def test_tie_break_distance_then_id(self):
        ranked = rank_vehicles(
            radius_km=10,
            vehicle_type=None,
            transmission=None,
            max_price_per_day=None,
            vehicles=[
                vehicle("m", 30, 1000),
                vehicle("z", 12, 1000),
                vehicle("b", 12, 1000),
                vehicle("a", 12, 1000),
            ],
        )
        self.assertEqual([item["id"] for item in ranked], ["a", "b", "z", "m"])
        self.assertEqual(len({item["rankScore"] for item in ranked}), 1)

        # Unrounded sums differ, displayed scores match, so distance wins.
        rounded_tie = rank_vehicles(
            radius_km=10000,
            vehicle_type="suv",
            transmission="automatic",
            max_price_per_day=1000,
            vehicles=[
                vehicle("far", 2, 499.04),
                vehicle("near", 1, 500),
            ],
        )
        self.assertEqual([item["rankScore"] for item in rounded_tie], [0.875, 0.875])
        self.assertEqual([item["id"] for item in rounded_tie], ["near", "far"])

    def test_closest_full_match_reason_and_echoed_distance(self):
        ranked = rank_vehicles(
            radius_km=10,
            vehicle_type="suv",
            transmission="automatic",
            max_price_per_day=3000,
            vehicles=[
                vehicle("far", 4, 2800),
                vehicle("near", 1.2, 2600),
            ],
        )
        self.assertEqual(ranked[0]["id"], "near")
        self.assertEqual(ranked[0]["distanceKm"], 1.2)
        self.assertEqual(
            ranked[0]["reason"],
            "Closest automatic SUV within the budget.",
        )
        self.assertEqual(ranked[1]["reason"], "SUV, automatic, within the budget, 4 km.")

        at_cap = rank_vehicles(
            radius_km=10,
            vehicle_type="suv",
            transmission="automatic",
            max_price_per_day=3000,
            vehicles=[vehicle("cap", 1.2, 3000)],
        )
        self.assertEqual(
            at_cap[0]["reason"],
            "Closest automatic SUV within the budget.",
        )
        self.assertNotIn("lat", ranked[0])
        self.assertNotIn("lng", ranked[0])

    def test_empty_list(self):
        self.assertEqual(
            rank_vehicles(
                radius_km=10,
                vehicle_type="suv",
                transmission="automatic",
                max_price_per_day=3000,
                vehicles=[],
            ),
            [],
        )


class RankHttpTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(
            os.environ,
            {"AI_SERVICE_TOKEN": "rank-secret", "CHROMA_PATH": self._tmp.name},
        )
        self._env.start()
        self._embedding = rag._embedding_factory
        self._answer = rag._answer_fn
        rag.set_embedding_factory(lambda: HashEmbedding())
        rag.set_answer_fn(lambda _query, _excerpts: "unused")
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        rag.set_embedding_factory(self._embedding)
        rag.set_answer_fn(self._answer)
        self._env.stop()
        self._tmp.cleanup()

    def test_missing_or_wrong_token_is_401(self):
        body = {
            "radiusKm": 10,
            "vehicles": [vehicle("v1", 1, 1000)],
        }
        missing = self.client.post("/rank", json=body)
        wrong = self.client.post(
            "/rank",
            json=body,
            headers={"X-Service-Token": "rank-secreX"},
        )
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)

    def test_health_stays_open_and_rank_accepts_token(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")

        ranked = self.client.post(
            "/rank",
            json={
                "radiusKm": 10,
                "type": "suv",
                "transmission": "automatic",
                "maxPricePerDay": 3000,
                "lat": 12.97,
                "lng": 77.6,
                "vehicles": [vehicle("near", 1.2, 2600)],
            },
            headers={"X-Service-Token": "rank-secret"},
        )
        self.assertEqual(ranked.status_code, 200)
        body = ranked.json()
        self.assertEqual(body[0]["rankScore"], 0.735)
        self.assertEqual(
            body[0]["reason"],
            "Closest automatic SUV within the budget.",
        )
        self.assertNotIn("lat", body[0])
        self.assertEqual(body[0]["distanceKm"], 1.2)


if __name__ == "__main__":
    unittest.main()
