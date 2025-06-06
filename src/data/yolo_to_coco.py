"""
YOLO to COCO Format Converter for Pineapple Detection Dataset

Converts the 176 pineapple images from YOLO format annotations to COCO format
for compatibility with Detectron2 Mask R-CNN training.

Input: 
- Images: src/data/images/*.jpg (1368x912 pixels)
- Labels: src/data/labels/*.txt (normalized YOLO format)

Output:
- COCO format JSON annotation file
- Verified image-annotation pairs
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
from PIL import Image
import numpy as np
from datetime import datetime


class YOLOToCOCOConverter:
    """Convert YOLO format annotations to COCO format for Detectron2."""
    
    def __init__(self, images_dir: str = "src/data/images", 
                 labels_dir: str = "src/data/labels",
                 output_dir: str = "outputs/dataset"):
        """
        Initialize the converter.
        
        Args:
            images_dir: Directory containing .jpg images
            labels_dir: Directory containing .txt YOLO annotations
            output_dir: Directory to save COCO format files
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO format structure
        self.coco_data = {
            "info": {
                "description": "Pineapple Detection Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "PROAC",
                "date_created": datetime.now().isoformat()
            },
            "categories": [
                {
                    "id": 1,
                    "name": "pineapple",
                    "supercategory": "fruit"
                }
            ],
            "images": [],
            "annotations": []
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
    def get_image_files(self) -> List[Path]:
        """Get all .jpg image files from the images directory."""
        image_files = list(self.images_dir.glob("*.jpg"))
        print(f"Found {len(image_files)} image files")
        return sorted(image_files)
    
    def get_corresponding_label_file(self, image_path: Path) -> Path:
        """Get the corresponding .txt label file for an image."""
        # Convert image filename to label filename
        # Example: 100_0945_0001_JPG.rf.3c4251ab70427c4f9574e0dab396a5a3.jpg
        # becomes: 100_0945_0001_JPG.rf.3c4251ab70427c4f9574e0dab396a5a3.txt
        label_name = image_path.stem + ".txt"
        label_path = self.labels_dir / label_name
        return label_path
    
    def parse_yolo_annotation(self, label_file: Path, img_width: int, img_height: int) -> List[Dict]:
        """
        Parse YOLO format annotation file.
        
        YOLO format: class_id center_x center_y width height (normalized 0-1)
        Example: 0 0.3838304093567252 0.35723684210526313 0.010146198830409436 0.015855263157894775
        
        Args:
            label_file: Path to .txt annotation file
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        if not label_file.exists():
            print(f"Warning: Label file not found: {label_file}")
            return annotations
            
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid annotation format in {label_file}: {line}")
                continue
                
            try:
                class_id = int(parts[0])  # Should be 0 for pineapple
                center_x = float(parts[1])  # Normalized center x
                center_y = float(parts[2])  # Normalized center y
                width = float(parts[3])     # Normalized width
                height = float(parts[4])    # Normalized height
                
                # Convert normalized coordinates to pixel coordinates
                pixel_center_x = center_x * img_width
                pixel_center_y = center_y * img_height
                pixel_width = width * img_width
                pixel_height = height * img_height
                
                # Convert to COCO format (top-left corner + width + height)
                bbox_x = pixel_center_x - (pixel_width / 2)
                bbox_y = pixel_center_y - (pixel_height / 2)
                
                # Ensure bounding box is within image bounds
                bbox_x = max(0, min(bbox_x, img_width))
                bbox_y = max(0, min(bbox_y, img_height))
                bbox_width = min(pixel_width, img_width - bbox_x)
                bbox_height = min(pixel_height, img_height - bbox_y)
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": 1,  # Pineapple category
                    "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0,
                    "segmentation": []  # Will be empty for bounding box only
                }
                
                annotations.append(annotation)
                self.annotation_id += 1
                
            except ValueError as e:
                print(f"Warning: Error parsing annotation in {label_file}: {line}, Error: {e}")
                continue
                
        return annotations
    
    def process_image(self, image_path: Path) -> bool:
        """
        Process a single image and its annotations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Error: Could not load image {image_path}")
                return False
                
            height, width, channels = img.shape
            
            # Verify expected dimensions (1368x912)
            if width != 1368 or height != 912:
                print(f"Warning: Unexpected image dimensions for {image_path}: {width}x{height}")
            
            # Add image info to COCO data
            image_info = {
                "id": self.image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
                "date_captured": datetime.now().isoformat()
            }
            self.coco_data["images"].append(image_info)
            
            # Get corresponding label file
            label_file = self.get_corresponding_label_file(image_path)
            
            # Parse annotations
            annotations = self.parse_yolo_annotation(label_file, width, height)
            self.coco_data["annotations"].extend(annotations)
            
            print(f"Processed {image_path.name}: {len(annotations)} annotations")
            self.image_id += 1
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def convert_dataset(self) -> str:
        """
        Convert the entire dataset from YOLO to COCO format.
        
        Returns:
            str: Path to the generated COCO annotation file
        """
        print("Starting YOLO to COCO conversion...")
        print(f"Images directory: {self.images_dir}")
        print(f"Labels directory: {self.labels_dir}")
        
        # Get all image files
        image_files = self.get_image_files()
        
        if not image_files:
            raise ValueError(f"No image files found in {self.images_dir}")
        
        # Process each image
        successful_count = 0
        for image_path in image_files:
            if self.process_image(image_path):
                successful_count += 1
        
        print(f"\nConversion completed:")
        print(f"- Total images found: {len(image_files)}")
        print(f"- Successfully processed: {successful_count}")
        print(f"- Total annotations: {len(self.coco_data['annotations'])}")
        print(f"- Average annotations per image: {len(self.coco_data['annotations'])/successful_count:.2f}")
        
        # Save COCO annotation file
        output_file = self.output_dir / "annotations.json"
        with open(output_file, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        
        print(f"COCO annotations saved to: {output_file}")
        return str(output_file)
    
    def create_dataset_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.15):
        """
        Create train/validation/test splits and save separate annotation files.
        
        Args:
            train_ratio: Ratio of images for training (default: 0.8 = 140 images)
            val_ratio: Ratio of images for validation (default: 0.15 = 26 images)
            test_ratio: Remaining for test (default: 0.05 = 10 images)
        """
        total_images = len(self.coco_data["images"])
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count
        
        print(f"\nCreating dataset splits:")
        print(f"- Train: {train_count} images")
        print(f"- Validation: {val_count} images")
        print(f"- Test: {test_count} images")
        
        # Shuffle images for random split (with fixed seed for reproducibility)
        np.random.seed(42)
        image_indices = np.random.permutation(total_images)
        
        # Split indices
        train_indices = set(image_indices[:train_count])
        val_indices = set(image_indices[train_count:train_count + val_count])
        test_indices = set(image_indices[train_count + val_count:])
        
        # Create split datasets
        splits = {
            "train": {"images": [], "annotations": []},
            "val": {"images": [], "annotations": []},
            "test": {"images": [], "annotations": []}
        }
        
        # Assign images to splits
        for idx, image_info in enumerate(self.coco_data["images"]):
            if idx in train_indices:
                splits["train"]["images"].append(image_info)
            elif idx in val_indices:
                splits["val"]["images"].append(image_info)
            else:
                splits["test"]["images"].append(image_info)
        
        # Assign annotations to splits based on image_id
        for annotation in self.coco_data["annotations"]:
            image_id = annotation["image_id"]
            # Find which split this image belongs to
            for split_name, split_data in splits.items():
                if any(img["id"] == image_id for img in split_data["images"]):
                    splits[split_name]["annotations"].append(annotation)
                    break
        
        # Save split files
        for split_name, split_data in splits.items():
            split_coco = {
                "info": self.coco_data["info"],
                "categories": self.coco_data["categories"],
                "images": split_data["images"],
                "annotations": split_data["annotations"]
            }
            
            output_file = self.output_dir / f"annotations_{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_coco, f, indent=2)
            
            print(f"- {split_name.capitalize()} split saved to: {output_file}")
            print(f"  Images: {len(split_data['images'])}, Annotations: {len(split_data['annotations'])}")
    
    def validate_conversion(self) -> Dict:
        """
        Validate the conversion by checking data consistency.
        
        Returns:
            Dict: Validation statistics
        """
        stats = {
            "total_images": len(self.coco_data["images"]),
            "total_annotations": len(self.coco_data["annotations"]),
            "images_with_annotations": 0,
            "images_without_annotations": 0,
            "annotation_counts": [],
            "bbox_areas": [],
            "invalid_bboxes": 0
        }
        
        # Count annotations per image
        image_annotation_count = {}
        for annotation in self.coco_data["annotations"]:
            image_id = annotation["image_id"]
            image_annotation_count[image_id] = image_annotation_count.get(image_id, 0) + 1
            
            # Check bbox validity
            bbox = annotation["bbox"]
            if bbox[2] <= 0 or bbox[3] <= 0:  # width or height <= 0
                stats["invalid_bboxes"] += 1
            else:
                stats["bbox_areas"].append(bbox[2] * bbox[3])
        
        # Count images with/without annotations
        for image_info in self.coco_data["images"]:
            image_id = image_info["id"]
            count = image_annotation_count.get(image_id, 0)
            stats["annotation_counts"].append(count)
            
            if count > 0:
                stats["images_with_annotations"] += 1
            else:
                stats["images_without_annotations"] += 1
        
        # Calculate statistics
        if stats["annotation_counts"]:
            stats["avg_annotations_per_image"] = np.mean(stats["annotation_counts"])
            stats["max_annotations_per_image"] = max(stats["annotation_counts"])
            stats["min_annotations_per_image"] = min(stats["annotation_counts"])
        
        if stats["bbox_areas"]:
            stats["avg_bbox_area"] = np.mean(stats["bbox_areas"])
            stats["median_bbox_area"] = np.median(stats["bbox_areas"])
        
        return stats


def main():
    """Main function to run the YOLO to COCO conversion."""
    print("=== YOLO to COCO Format Converter ===")
    print("Converting pineapple detection dataset...")
    
    # Initialize converter
    converter = YOLOToCOCOConverter()
    
    # Convert dataset
    try:
        annotation_file = converter.convert_dataset()
        
        # Create train/val/test splits
        converter.create_dataset_splits()
        
        # Validate conversion
        stats = converter.validate_conversion()
        
        print("\n=== Validation Results ===")
        print(f"Total images: {stats['total_images']}")
        print(f"Total annotations: {stats['total_annotations']}")
        print(f"Images with annotations: {stats['images_with_annotations']}")
        print(f"Images without annotations: {stats['images_without_annotations']}")
        print(f"Average annotations per image: {stats.get('avg_annotations_per_image', 0):.2f}")
        print(f"Max annotations per image: {stats.get('max_annotations_per_image', 0)}")
        print(f"Invalid bounding boxes: {stats['invalid_bboxes']}")
        
        if stats.get('avg_bbox_area'):
            print(f"Average bbox area: {stats['avg_bbox_area']:.2f} pixels²")
        
        print("\n✅ Conversion completed successfully!")
        print(f"📄 Main annotation file: {annotation_file}")
        print("📁 Split files created for train/val/test")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main() 