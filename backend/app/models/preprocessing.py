"""Data preprocessing for YOLO format."""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.core.logger import setup_logger

logger = setup_logger("preprocessing")


class ImagePreprocessor:
    """Preprocess images for YOLO training."""
    
    def __init__(self, target_size: int = 640):
        self.target_size = target_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image maintaining aspect ratio with padding."""
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to square
        pad_w = (self.target_size - new_w) // 2
        pad_h = (self.target_size - new_h) // 2
        
        padded = cv2.copyMakeBorder(
            resized,
            pad_h, self.target_size - new_h - pad_h,
            pad_w, self.target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )
        
        return padded
    
    def process_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize_image(image)
        return image


def convert_to_yolo_format(data_root: Path, output_root: Path):
    """Convert MVTec AD format to YOLO format."""
    
    preprocessor = ImagePreprocessor()
    
    # Create output directories
    (output_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    categories = [d for d in data_root.iterdir() if d.is_dir()]
    
    stats = {"train_good": 0, "val_good": 0, "val_defect": 0}
    
    for category in categories:
        logger.info(f"\n📁 Processing {category.name}...")
        
        # Training images (good)
        train_good_imgs = list((category / "train" / "good").glob("*.png"))
        for img_path in tqdm(train_good_imgs, desc="Train (good)"):
            process_and_save(img_path, output_root, "train", 0, preprocessor)
            stats["train_good"] += 1
        
        # Validation images (good)
        test_good_imgs = list((category / "test" / "good").glob("*.png"))
        for img_path in tqdm(test_good_imgs, desc="Val (good)"):
            process_and_save(img_path, output_root, "val", 0, preprocessor)
            stats["val_good"] += 1
        
        # Validation images (defect)
        test_defect_imgs = list((category / "test" / "defect").glob("*.png"))
        for img_path in tqdm(test_defect_imgs, desc="Val (defect)"):
            process_and_save(img_path, output_root, "val", 1, preprocessor)
            stats["val_defect"] += 1
    
    logger.info(f"\n✅ Conversion complete!")
    logger.info(f"Train (good): {stats['train_good']}")
    logger.info(f"Val (good): {stats['val_good']}")
    logger.info(f"Val (defect): {stats['val_defect']}")
    logger.info(f"Total: {sum(stats.values())} images")


def process_and_save(
    img_path: Path,
    output_root: Path,
    split: str,
    class_id: int,
    preprocessor: ImagePreprocessor
):
    """Process and save image with YOLO label."""
    
    # Process image
    image = preprocessor.process_image(img_path)
    
    # Save image
    output_img_path = output_root / "images" / split / img_path.name
    cv2.imwrite(str(output_img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Create YOLO label file
    label_path = output_root / "labels" / split / f"{img_path.stem}.txt"
    
    if class_id == 1:  # Defect class
        with open(label_path, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
    else:  # Good class - empty label
        label_path.touch()


if __name__ == "__main__":
    data_root = Path("data/raw")
    output_root = Path("data/processed")
    
    logger.info("🔄 Converting dataset to YOLO format...")
    convert_to_yolo_format(data_root, output_root)
