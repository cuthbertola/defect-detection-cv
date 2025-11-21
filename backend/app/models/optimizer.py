"""Model optimization to ONNX format."""

import torch
from ultralytics import YOLO
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
import time
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.logger import setup_logger

logger = setup_logger("optimizer", log_file="logs/optimization.log")


class ModelOptimizer:
    """Optimize PyTorch model to ONNX format."""
    
    def __init__(self, model_path: str):
        """Initialize optimizer with model path."""
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(str(self.model_path))
        logger.info(f"✅ Loaded model from {model_path}")
    
    def export_to_onnx(self, output_path: str = None) -> str:
        """Export model to ONNX format."""
        
        if output_path is None:
            output_path = "models/production/model.onnx"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔄 Exporting model to ONNX format...")
        
        # Export to ONNX
        self.model.export(
            format="onnx",
            imgsz=640,
            simplify=True,
            dynamic=False,
            opset=12
        )
        
        # Move to desired location
        onnx_path = self.model_path.with_suffix('.onnx')
        
        if output_path != onnx_path and onnx_path.exists():
            import shutil
            shutil.move(str(onnx_path), str(output_path))
        
        logger.info(f"✅ Model exported to {output_path}")
        
        # Verify ONNX model
        self.verify_onnx(output_path)
        
        # Benchmark
        self.benchmark(output_path)
        
        return str(output_path)
    
    def verify_onnx(self, onnx_path: str):
        """Verify ONNX model is valid."""
        
        logger.info("🔍 Verifying ONNX model...")
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model is valid")
        except Exception as e:
            logger.error(f"❌ ONNX validation failed: {e}")
            raise
    
    def benchmark(self, onnx_path: str, num_runs: int = 100):
        """Benchmark ONNX model inference time."""
        
        logger.info(f"⚡ Benchmarking model ({num_runs} runs)...")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path))
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm-up runs
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        logger.info(f"\n📊 Benchmark Results:")
        logger.info(f"  Average: {avg_time:.2f} ± {std_time:.2f} ms")
        logger.info(f"  Min:     {min_time:.2f} ms")
        logger.info(f"  Max:     {max_time:.2f} ms")
        
        # Model size
        model_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        logger.info(f"  Size:    {model_size:.2f} MB")
        
        # Check targets
        target_time = 100
        target_size = 50
        
        if avg_time < target_time:
            logger.info(f"✅ Inference time target met (<{target_time}ms)")
        else:
            logger.warning(f"⚠️ Inference time ({avg_time:.2f}ms) exceeds target ({target_time}ms)")
        
        if model_size < target_size:
            logger.info(f"✅ Model size target met (<{target_size}MB)")
        else:
            logger.warning(f"⚠️ Model size ({model_size:.2f}MB) exceeds target ({target_size}MB)")


def main():
    """Main optimization function."""
    
    logger.info("=" * 60)
    logger.info("MODEL OPTIMIZATION PIPELINE")
    logger.info("=" * 60)
    
    # Path to trained model
    model_path = "models/production/best.pt"
    
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        return
    
    # Create optimizer
    optimizer = ModelOptimizer(model_path)
    
    # Export to ONNX
    onnx_path = optimizer.export_to_onnx("models/production/model.onnx")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"🎉 OPTIMIZATION COMPLETE!")
    logger.info(f"Model saved: {onnx_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
