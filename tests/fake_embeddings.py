"""Keyword vectors so RAG tests never download all-MiniLM-L6-v2."""

from __future__ import annotations

import math

PHRASES = [
    ("full tank", 0),
    ("shortfall is charged", 1),
    ("correct fuel", 2),
    ("The return time is", 3),
    ("The late fee is", 4),
    ("Repeated late returns", 5),
    ("may cancel a confirmed", 6),
    ("at least 24 hours", 7),
    ("inside 24 hours", 8),
    ("third-party liability", 9),
    ("renter\u2019s responsibility", 10),
    ("add-on cover", 11),
    ("securityDeposit", 12),
    ("refundable after", 13),
    ("may be withheld", 14),
]


class HashEmbedding:
    def __call__(self, input):
        return [self._vector(text) for text in input]

    def embed_query(self, input):
        return self.__call__(input)

    def is_legacy(self) -> bool:
        return True

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def name() -> str:
        return "hash-test"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbedding":
        return HashEmbedding()

    @staticmethod
    def validate_config(config: dict) -> None:
        return None

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        return None

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 16
        matched = False
        for phrase, index in PHRASES:
            if phrase in text:
                vector[index] = 1.0
                matched = True
        lowered = text.lower()
        if "chauffeur" in lowered:
            vector[15] = 1.0
            matched = True
        if not matched:
            if "fuel" in lowered:
                vector[0] = 1.0
            if "late" in lowered:
                vector[3] = 1.0
            if vector == [0.0] * 16:
                vector[15] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]
