import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from utils.predictor import MC_SAMPLES, load_runtime, predict
from graphs import MoleculeInputError


class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1)


class PredictionRow(BaseModel):
    task: str
    logit: float
    probability: float
    total: float
    aleatoric: float
    epistemic: float
    domain: Optional[Dict[str, Any]]


class PredictionResponse(BaseModel):
    smiles: str
    tier1: List[PredictionRow]
    hershberger: PredictionRow
    uterotrophic: PredictionRow
    mc_samples: int


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.runtime = load_runtime()
    application.state.prediction_lock = asyncio.Lock()
    yield


app = FastAPI(title="PredictED", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    runtime = request.app.state.runtime
    return {
        "status": "ready",
        "models": ["tier1", "hershberger", "uterotrophic"],
        "tier1_tasks": len(runtime.assays),
        "applicability_domains": {
            name: len(domain.task_domains)
            for name, domain in runtime.domains.items()
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest, request: Request):
    async with request.app.state.prediction_lock:
        try:
            return await run_in_threadpool(
                predict, payload.smiles, request.app.state.runtime, MC_SAMPLES)
        except MoleculeInputError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
