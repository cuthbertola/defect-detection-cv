"""MLflow experiment tracking for defect detection models."""

import mlflow
import mlflow.pytorch
import mlflow.onnx
from pathlib import Path
import json
from datetime import datetime

# Set tracking URI (local for now, can be remote server)
MLFLOW_TRACKING_URI = "file:///app/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name
EXPERIMENT_NAME = "defect-detection"


def get_or_create_experiment():
    """Get or create MLflow experiment."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "defect-detection", "team": "cv-engineering"}
        )
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def log_training_run(
    model_name: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    metrics: dict,
    model_path: str = None,
    parameters: dict = None
):
    """
    Log a training run to MLflow.
    
    Args:
        model_name: Name of the model
        model_type: Type of model (yolov8n, yolov8s, etc.)
        epochs: Number of training epochs
        batch_size: Batch size used
        learning_rate: Learning rate
        metrics: Dictionary of metrics (accuracy, f1, precision, recall, etc.)
        model_path: Path to saved model file
        parameters: Additional parameters to log
    """
    experiment_id = get_or_create_experiment()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        
        if parameters:
            for key, value in parameters.items():
                mlflow.log_param(key, value)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model artifact
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path)
        
        # Log tags
        mlflow.set_tag("framework", "pytorch")
        mlflow.set_tag("task", "object_detection")
        mlflow.set_tag("dataset", "synthetic_defects")
        
        return mlflow.active_run().info.run_id


def log_inference_metrics(
    model_version: str,
    inference_time_ms: float,
    confidence: float,
    has_defect: bool,
    image_size: tuple
):
    """Log inference metrics for monitoring."""
    experiment_id = get_or_create_experiment()
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("image_width", image_size[0])
        mlflow.log_param("image_height", image_size[1])
        
        mlflow.log_metric("inference_time_ms", inference_time_ms)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("defect_detected", 1 if has_defect else 0)
        
        mlflow.set_tag("run_type", "inference")


def get_best_model(metric: str = "f1_score"):
    """Get the best model based on a metric."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return None
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type != 'inference'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )
    
    if len(runs) > 0:
        return runs.iloc[0].to_dict()
    return None


def list_all_runs():
    """List all training runs."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        return []
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.run_type != 'inference'"
    )
    
    return runs.to_dict('records') if len(runs) > 0 else []
