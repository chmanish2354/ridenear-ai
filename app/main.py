from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict

from app.rag import build_index
from app.rank import rank_vehicles, service_token_matches


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = build_index()
    try:
        yield
    finally:
        app.state.index.close()


app = FastAPI(title="RideNear AI", lifespan=lifespan)


class RankVehicle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    distanceKm: float
    pricePerDay: float
    type: str
    transmission: str


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    radiusKm: float
    type: str | None = None
    transmission: str | None = None
    maxPricePerDay: float | None = None
    vehicles: list[RankVehicle]


class RagQuery(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str


def require_service_token(
    x_service_token: Annotated[str | None, Header()] = None,
) -> None:
    if not service_token_matches(x_service_token):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict[str, object]:
    index = getattr(app.state, "index", None)
    count = index.vector_count() if index is not None else 0
    return {"status": "ok", "vectorCount": count}


@app.post("/rank", dependencies=[Depends(require_service_token)])
def rank(body: RankRequest) -> list[dict]:
    return rank_vehicles(
        radius_km=body.radiusKm,
        vehicle_type=body.type,
        transmission=body.transmission,
        max_price_per_day=body.maxPricePerDay,
        vehicles=[vehicle.model_dump() for vehicle in body.vehicles],
    )


@app.post("/rag/query", dependencies=[Depends(require_service_token)])
def rag_query(body: RagQuery) -> dict:
    return app.state.index.answer(body.query)
