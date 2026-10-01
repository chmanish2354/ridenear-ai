"""Score search candidates. Coordinates are not accepted or echoed."""

from __future__ import annotations

import os
import secrets
from decimal import Decimal, ROUND_HALF_UP

TYPE_LABELS = {
    "suv": "SUV",
    "sedan": "sedan",
    "hatchback": "hatchback",
    "muv": "MUV",
}


def service_token_matches(provided: str | None) -> bool:
    expected = os.environ.get("AI_SERVICE_TOKEN", "")
    if not expected or not provided or len(provided) != len(expected):
        return False
    return secrets.compare_digest(provided, expected)


def rank_vehicles(
    *,
    radius_km: float,
    vehicle_type: str | None,
    transmission: str | None,
    max_price_per_day: float | None,
    vehicles: list[dict],
) -> list[dict]:
    if not vehicles:
        return []

    closest_distance = min(vehicle["distanceKm"] for vehicle in vehicles)
    scored: list[tuple[Decimal, dict]] = []
    for vehicle in vehicles:
        distance_score = _distance_feature(vehicle["distanceKm"], radius_km)
        price_score = _price_feature(vehicle["pricePerDay"], max_price_per_day)
        type_score = _match_feature(vehicle_type, vehicle["type"])
        transmission_score = _match_feature(transmission, vehicle["transmission"])
        total = (
            Decimal("0.40") * distance_score
            + Decimal("0.25") * price_score
            + Decimal("0.20") * type_score
            + Decimal("0.15") * transmission_score
        )
        rank_score = _round_score(total)
        reason = _reason(
            vehicle=vehicle,
            requested_type=vehicle_type,
            requested_transmission=transmission,
            max_price_per_day=max_price_per_day,
            type_score=type_score,
            transmission_score=transmission_score,
            price_score=price_score,
            is_closest=vehicle["distanceKm"] == closest_distance,
        )
        scored.append(
            (
                Decimal(str(rank_score)),
                {
                    "id": vehicle["id"],
                    "distanceKm": vehicle["distanceKm"],
                    "pricePerDay": vehicle["pricePerDay"],
                    "type": vehicle["type"],
                    "transmission": vehicle["transmission"],
                    "rankScore": rank_score,
                    "reason": reason,
                },
            )
        )

    scored.sort(key=lambda item: (-item[0], item[1]["distanceKm"], item[1]["id"]))
    return [item[1] for item in scored]


def _distance_feature(distance_km: float, radius_km: float) -> Decimal:
    if radius_km <= 0:
        return Decimal(1) if distance_km == 0 else Decimal(0)
    raw = Decimal(1) - (Decimal(str(distance_km)) / Decimal(str(radius_km)))
    return _floor_unit(raw)


def _price_feature(price: float, max_price: float | None) -> Decimal:
    if max_price is None:
        return Decimal(1)
    if max_price == 0:
        return Decimal(1) if price == 0 else Decimal(0)
    if max_price < 0:
        return Decimal(0)
    raw = Decimal(1) - (Decimal(str(price)) / Decimal(str(max_price)))
    return _floor_unit(raw)


def _match_feature(requested: str | None, actual: str) -> Decimal:
    if requested is None:
        return Decimal(1)
    return Decimal(1) if requested == actual else Decimal(0)


def _floor_unit(value: Decimal) -> Decimal:
    if value < 0:
        return Decimal(0)
    if value > 1:
        return Decimal(1)
    return value


def _round_score(value: Decimal) -> float:
    quantized = value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return float(_floor_unit(quantized))


def _reason(
    *,
    vehicle: dict,
    requested_type: str | None,
    requested_transmission: str | None,
    max_price_per_day: float | None,
    type_score: Decimal,
    transmission_score: Decimal,
    price_score: Decimal,
    is_closest: bool,
) -> str:
    within_budget = (
        max_price_per_day is not None
        and vehicle["pricePerDay"] <= max_price_per_day
    )
    full_preferences = (
        type_score == 1
        and transmission_score == 1
        and within_budget
        and requested_type is not None
        and requested_transmission is not None
    )
    if full_preferences and is_closest:
        label = TYPE_LABELS.get(requested_type, requested_type)
        return f"Closest {requested_transmission} {label} within the budget."

    signals: list[str] = []
    if type_score > 0:
        kind = requested_type or vehicle["type"]
        signals.append(TYPE_LABELS.get(kind, kind))
    if transmission_score > 0:
        signals.append(requested_transmission or vehicle["transmission"])
    if price_score > 0:
        signals.append(
            "within the budget" if max_price_per_day is not None else "no price cap"
        )
    signals.append(f"{_format_km(vehicle['distanceKm'])} km")
    return f"{', '.join(signals)}."


def _format_km(distance_km: float) -> str:
    text = format(Decimal(str(distance_km)).normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text
