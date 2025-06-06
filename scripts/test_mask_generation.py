#!/usr/bin/env python3
"""
Test script to evaluate automatic mask generation from bounding boxes
Generates circular/elliptical segmentation polygons and visualizes results
"""

import os
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Tuple, Dict
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yolo_annotations(label_path: str) -> List[List[float]]:
    """Load YOLO format annotations from file"""
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    annotations.append([class_id, x_center, y_center, width, height])
    return annotations

def yolo_to_pixel_coords(yolo_coords: List[float], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized coordinates to pixel coordinates"""
    class_id, x_center, y_center, width, height = yolo_coords
    
    # Convert to pixel coordinates
    x_center_px = int(x_center * img_width)
    y_center_px = int(y_center * img_height)
    width_px = int(width * img_width)
    height_px = int(height * img_height)
    
    # Calculate bounding box corners
    x_min = x_center_px - width_px // 2
    y_min = y_center_px - height_px // 2
    x_max = x_center_px + width_px // 2
    y_max = y_center_px + height_px // 2
    
    return x_min, y_min, x_max, y_max

def generate_elliptical_mask(x_center: int, y_center: int, width: int, height: int, num_points: int = 36) -> List[List[float]]:
    """Generate elliptical segmentation polygon from bounding box"""
    # Use slightly smaller ellipse than bounding box for better fit
    a = width * 0.4  # Semi-major axis (horizontal)
    b = height * 0.4  # Semi-minor axis (vertical)
    
    polygon = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = x_center + a * math.cos(angle)
        y = y_center + b * math.sin(angle)
        polygon.extend([float(x), float(y)])
    
    return [polygon]  # COCO format expects list of polygons

def generate_circular_mask(x_center: int, y_center: int, width: int, height: int, num_points: int = 36) -> List[List[float]]:
    """Generate circular segmentation polygon from bounding box"""
    # Use smaller of width/height for radius
    radius = min(width, height) * 0.4
    
    polygon = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = x_center + radius * math.cos(angle)
        y = y_center + radius * math.sin(angle)
        polygon.extend([float(x), float(y)])
    
    return [polygon]  # COCO format expects list of polygons

def visualize_masks(image_path: str, annotations: List[List[float]], output_path: str):
    """Visualize original image with bounding boxes and generated masks"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Mask Generation Test: {os.path.basename(image_path)}', fontsize=16)
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Image with bounding boxes
    bbox_image = image_rgb.copy()
    for ann in annotations:
        x_min, y_min, x_max, y_max = yolo_to_pixel_coords(ann, img_width, img_height)
        cv2.rectangle(bbox_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    
    axes[0, 1].imshow(bbox_image)
    axes[0, 1].set_title(f'Bounding Boxes ({len(annotations)} annotations)')
    axes[0, 1].axis('off')
    
    # Image with circular masks
    circular_image = image_rgb.copy()
    for ann in annotations:
        x_min, y_min, x_max, y_max = yolo_to_pixel_coords(ann, img_width, img_height)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Generate circular mask
        mask_poly = generate_circular_mask(x_center, y_center, width, height)
        points = np.array(mask_poly[0]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(circular_image, [points], True, (0, 255, 0), 2)
        cv2.fillPoly(circular_image, [points], (0, 255, 0, 100))
    
    axes[1, 0].imshow(circular_image)
    axes[1, 0].set_title('Generated Circular Masks')
    axes[1, 0].axis('off')
    
    # Image with elliptical masks
    elliptical_image = image_rgb.copy()
    for ann in annotations:
        x_min, y_min, x_max, y_max = yolo_to_pixel_coords(ann, img_width, img_height)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Generate elliptical mask
        mask_poly = generate_elliptical_mask(x_center, y_center, width, height)
        points = np.array(mask_poly[0]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(elliptical_image, [points], True, (255, 165, 0), 2)
        cv2.fillPoly(elliptical_image, [points], (255, 165, 0, 100))
    
    axes[1, 1].imshow(elliptical_image)
    axes[1, 1].set_title('Generated Elliptical Masks')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {output_path}")

def test_mask_generation():
    """Test mask generation on 5 sample images"""
    logger.info("🧪 Testing Automatic Mask Generation from Bounding Boxes")
    logger.info("=" * 60)
    
    # Setup paths
    images_dir = Path("src/data/images")
    labels_dir = Path("src/data/labels")
    output_dir = Path("outputs/test_mask_generation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of available images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG"))
    if len(image_files) < 5:
        logger.error(f"Found only {len(image_files)} images, need at least 5")
        return
    
    # Select 5 random images
    random.seed(42)  # For reproducible results
    selected_images = random.sample(image_files, 5)
    
    logger.info(f"Selected {len(selected_images)} images for testing:")
    
    total_annotations = 0
    for i, image_path in enumerate(selected_images, 1):
        logger.info(f"\n📷 Image {i}: {image_path.name}")
        
        # Find corresponding label file
        label_name = image_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        if not label_path.exists():
            logger.warning(f"No label file found: {label_path}")
            continue
        
        # Load annotations
        annotations = load_yolo_annotations(str(label_path))
        logger.info(f"   📊 Annotations: {len(annotations)}")
        total_annotations += len(annotations)
        
        if len(annotations) == 0:
            logger.warning(f"   ⚠️  No annotations found in {label_path}")
            continue
        
        # Generate visualization
        output_path = output_dir / f"mask_test_{i}_{image_path.stem}.png"
        visualize_masks(str(image_path), annotations, str(output_path))
    
    logger.info(f"\n✅ Mask generation test complete!")
    logger.info(f"📊 Total annotations processed: {total_annotations}")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info(f"\n🔍 Review the generated images to evaluate mask quality:")
    for i in range(1, 6):
        result_file = output_dir / f"mask_test_{i}_*.png"
        matching_files = list(output_dir.glob(f"mask_test_{i}_*.png"))
        if matching_files:
            logger.info(f"   Image {i}: {matching_files[0]}")
    
    # Generate summary statistics
    logger.info(f"\n📈 Quality Assessment:")
    logger.info(f"   🔴 Red boxes: Original YOLO bounding boxes")
    logger.info(f"   🟢 Green masks: Circular segmentation (better for round pineapples)")
    logger.info(f"   🟠 Orange masks: Elliptical segmentation (adapts to bbox shape)")
    logger.info(f"\n💡 Recommendation: Choose the mask type that best fits your pineapples!")

if __name__ == "__main__":
    test_mask_generation() 