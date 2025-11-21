"""Simple YOLOv8 training script."""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.logger import setup_logger

logger = setup_logger("yolov8_trainer", log_file="logs/training.log")


def main():
    """Main training function."""
    
    logger.info("=" * 60)
    logger.info("YOLOv8 DEFECT DETECTION TRAINING")
    logger.info("=" * 60)
    
    # Load config
    with open("configs/train_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("✅ Loaded configuration")
    
    # Initialize model
    model_name = config['model']['name']
    model = YOLO(f"{model_name}.pt")
    logger.info(f"✅ Initialized {model_name} model")
    
    # Check device
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    
    logger.info(f"🚀 Starting training on {device}...")
    logger.info(f"📊 Epochs: {config['training']['epochs']}")
    logger.info(f"📊 Batch size: {config['training']['batch_size']}")
    
    # Train model
    results = model.train(
        data="data/dataset.yaml",
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['model']['input_size'],
        lr0=config['training']['learning_rate'],
        optimizer=config['training']['optimizer'],
        device=device,
        workers=config['hardware']['workers'],
        project=config['paths']['output_dir'],
        name="yolov8_run",
        exist_ok=True,
        pretrained=config['model']['pretrained'],
        patience=config['training']['early_stopping_patience'],
        save=True,
        save_period=10,
        plots=True
    )
    
    # Save model
    model_path = Path(config['paths']['output_dir']) / "yolov8_run" / "weights" / "best.pt"
    if model_path.exists():
        # Copy to production
        prod_path = Path("models/production/best.pt")
        prod_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(model_path, prod_path)
        logger.info(f"✅ Model saved to: {prod_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
