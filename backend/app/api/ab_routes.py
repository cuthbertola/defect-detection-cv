"""A/B Testing API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])

# Import framework
from app.api.ab_testing import ab_framework


class StartTestRequest(BaseModel):
    test_name: str
    model_a: str = "v1_baseline"
    model_b: str = "v2_optimized"
    weight_a: float = 0.5


class RecordResultRequest(BaseModel):
    model_name: str
    inference_time: float
    has_defect: bool
    confidence: float


@router.post("/start")
async def start_ab_test(request: StartTestRequest):
    """Start a new A/B test."""
    try:
        result = ab_framework.start_test(
            test_name=request.test_name,
            model_a=request.model_a,
            model_b=request.model_b,
            weight_a=request.weight_a
        )
        return {"status": "success", "test": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/select-model")
async def select_model():
    """Select a model based on A/B test weights."""
    try:
        model = ab_framework.select_model()
        return {"selected_model": model}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/record")
async def record_result(request: RecordResultRequest):
    """Record an inference result."""
    ab_framework.record_result(
        model_name=request.model_name,
        inference_time=request.inference_time,
        has_defect=request.has_defect,
        confidence=request.confidence
    )
    return {"status": "recorded"}


@router.get("/stats")
async def get_stats():
    """Get current A/B test statistics."""
    return ab_framework.get_test_stats()


@router.get("/winner")
async def get_winner(metric: str = "avg_inference_time_ms"):
    """Determine the winning model."""
    return ab_framework.determine_winner(metric)


@router.post("/save")
async def save_results():
    """Save test results to file."""
    ab_framework.save_results()
    return {"status": "saved", "file": "ab_test_results.json"}


@router.get("/models")
async def list_models():
    """List registered models."""
    return {
        "models": [
            {"name": name, "weight": model.weight, "requests": model.request_count}
            for name, model in ab_framework.models.items()
        ]
    }
