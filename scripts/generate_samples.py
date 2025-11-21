#!/usr/bin/env python3
"""Generate synthetic sample data for testing."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from app.core.logger import setup_logger

logger = setup_logger("sample_generator")


def create_sample_image(output_path: Path, has_defect: bool = False):
    """Create a sample product image."""
    
    # Create base image
    img = Image.new('RGB', (640, 640), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Draw product (circle)
    draw.ellipse([100, 100, 540, 540], fill=(150, 150, 150), outline=(100, 100, 100))
    
    if has_defect:
        # Add random defects
        for _ in range(random.randint(1, 3)):
            x = random.randint(150, 490)
            y = random.randint(150, 490)
            size = random.randint(20, 40)
            draw.ellipse([x, y, x+size, y+size], fill=(255, 0, 0))
    
    img.save(output_path)


def main():
    """Generate sample dataset."""
    
    categories = ["bottle", "cable", "capsule", "hazelnut", "metal_nut"]
    
    logger.info("🎨 Generating sample images...")
    
    for category in categories:
        base_path = Path(f"data/raw/{category}")
        
        # Training images (good only)
        train_good_dir = base_path / "train" / "good"
        for i in range(50):
            create_sample_image(train_good_dir / f"good_{i:04d}.png", has_defect=False)
        
        # Test images (good)
        test_good_dir = base_path / "test" / "good"
        for i in range(10):
            create_sample_image(test_good_dir / f"good_{i:04d}.png", has_defect=False)
        
        # Test images (defect)
        test_defect_dir = base_path / "test" / "defect"
        for i in range(10):
            create_sample_image(test_defect_dir / f"defect_{i:04d}.png", has_defect=True)
        
        logger.info(f"✅ Generated samples for {category}")
    
    total = len(categories) * 70
    logger.info(f"\n✅ Generated {total} sample images")
    logger.info("\n📝 Next: Data preprocessing - we'll do this together next!")


if __name__ == "__main__":
    main()
