"""YOLOv8 training script with MLflow tracking."""

import torch
from ultralytics import YOLO
import mlflow
import yaml
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.logger import setup_logger

logger = setup_logger("yolov8_trainer", log_file="logs/training.log")


class YOLOv8Trainer:
    """YOLOv8 model trainer with MLflow tracking."""
    
    def __init__(self, config_path: str = "configs/train_config.yaml"):
        """Initialize trainer with configuration."""
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded configuration from {config_path}")
        
        # Initialize model
        model_name = self.config['model']['name']
        self.model = YOLO(f"{model_name}.pt")
        logger.info(f"✅ Initialized {model_name} model")
        
        # Check device
        device = self.config['hardware']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, using CPU")
            self.config['hardware']['device'] = 'cpu'
        
        # Setup MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """Setup MLflow experiment tracking."""
        
        mlflow_config = self.config['mlflow']
        mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
        mlflow.set_experiment(mlflow_config['experiment_name'])
        logger.info(f"✅ MLflow tracking: {mlflow_config['tracking_uri']}")
    
    def train(self):
        """Train the YOLOv8 model."""
        
        logger.info("🚀 Starting training...")
        
        training_config = self.config['training']
        model_config = self.config['model']
        
        # Start MLflow run
        run_name = f"{self.config['mlflow']['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            
            # Log parameters
            params = {
                "model_name": model_config['name'],
                "epochs": training_config['epochs'],
                "batch_size": training_config['batch_size'],
                "learning_rate": training_config['learning_rate'],
                "optimizer": training_config['optimizer'],
                "image_size": model_config['input_size'],
                "device": self.config['hardware']['device']
            }
            mlflow.log_params(params)
            logger.info(f"📊 Training parameters: {params}")
            
            # Train model
            results = self.model.train(
                data="data/dataset.yaml",
                epochs=training_config['epochs'],
                batch=training_config['batch_size'],
                imgsz=model_config['input_size'],
                lr0=training_config['learning_rate'],
                optimizer=training_config['optimizer'],
                device=self.config['hardware']['device'],
                workers=self.config['hardware']['workers'],
                project=self.config['paths']['output_dir'],
                name="yolov8_run",
                exist_ok=True,
                pretrained=model_config['pretrained'],
                patience=training_config['early_stopping_patience'],
                save=True,
                save_period=10
            )
            
            # Log final metrics
            model_path = Path(self.config['paths']['output_dir']) / "yolov8_run" / "weights" / "best.pt"
            if model_path.exists():
                mlflow.log_artifact(str(model_path))
                logger.info(f"✅ Model saved: {model_path}")
            
            logger.info(f"🎉 Training complete! Run: {run_name}")
            
            return results


def main():
    """Main training function."""
    
    logger.info("=" * 60)
    logger.info("YOLOv8 DEFECT DETECTION TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Create trainer
    trainer = YOLOv8Trainer()
    
    # Train model
    results = trainer.train()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING PIPELINE COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
