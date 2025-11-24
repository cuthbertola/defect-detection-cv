"""MLflow API routes for experiment tracking."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(prefix="/mlflow", tags=["MLflow"])


class TrainingRunRequest(BaseModel):
    model_name: str
    model_type: str = "yolov8n"
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    metrics: Dict[str, float]
    parameters: Optional[Dict[str, str]] = None


class TrainingRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


@router.post("/log-training", response_model=TrainingRunResponse)
async def log_training_run(request: TrainingRunRequest):
    """Log a training run to MLflow."""
    try:
        from app.mlflow_tracking import log_training_run
        
        run_id = log_training_run(
            model_name=request.model_name,
            model_type=request.model_type,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            metrics=request.metrics,
            parameters=request.parameters
        )
        
        return TrainingRunResponse(
            run_id=run_id,
            status="success",
            message=f"Training run logged successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best-model")
async def get_best_model(metric: str = "f1_score"):
    """Get the best model based on a metric."""
    try:
        from app.mlflow_tracking import get_best_model
        
        best = get_best_model(metric)
        if best is None:
            return {"message": "No models found", "model": None}
        return {"message": "Best model found", "model": best}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def list_runs():
    """List all training runs."""
    try:
        from app.mlflow_tracking import list_all_runs
        
        runs = list_all_runs()
        return {"total_runs": len(runs), "runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def get_experiments():
    """Get experiment information."""
    try:
        import mlflow
        
        experiments = mlflow.search_experiments()
        return {
            "total_experiments": len(experiments),
            "experiments": [
                {
                    "name": exp.name,
                    "experiment_id": exp.experiment_id,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
