#!/usr/bin/env python3
"""
Download and prepare MVTec AD dataset structure.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.core.logger import setup_logger

logger = setup_logger("dataset_downloader")


def create_dataset_structure():
    """Create MVTec AD dataset structure."""
    
    data_dir = Path("data/raw")
    categories = ["bottle", "cable", "capsule", "hazelnut", "metal_nut"]
    
    for category in categories:
        # Create directory structure
        (data_dir / category / "train" / "good").mkdir(parents=True, exist_ok=True)
        (data_dir / category / "test" / "good").mkdir(parents=True, exist_ok=True)
        (data_dir / category / "test" / "defect").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Created structure for {category}")
    
    logger.info("\n✅ Dataset structure created!")
    logger.info("\n�� Next: Run 'python3 scripts/generate_samples.py' to create sample images")


if __name__ == "__main__":
    create_dataset_structure()
