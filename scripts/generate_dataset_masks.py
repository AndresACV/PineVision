#!/usr/bin/env python3
"""
Generate segmentation polygons for the entire dataset
Updates COCO JSON files with circular/elliptical masks from bounding boxes
"""

import os
import json
import math
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_elliptical_mask(bbox: List[float], num_points: int = 36) -> List[List[float]]:
    """Generate elliptical segmentation polygon from COCO bounding box"""
    x, y, width, height = bbox
    
    # Center of the bounding box
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Semi-axes (slightly smaller than bbox for better fit)
    a = width * 0.4  # Semi-major axis (horizontal)
    b = height * 0.4  # Semi-minor axis (vertical)
    
    polygon = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        px = x_center + a * math.cos(angle)
        py = y_center + b * math.sin(angle)
        polygon.extend([float(px), float(py)])
    
    return [polygon]  # COCO format expects list of polygons

def generate_circular_mask(bbox: List[float], num_points: int = 36) -> List[List[float]]:
    """Generate circular segmentation polygon from COCO bounding box"""
    x, y, width, height = bbox
    
    # Center of the bounding box
    x_center = x + width / 2
    y_center = y + height / 2
    
    # Radius (smaller of width/height for circular fit)
    radius = min(width, height) * 0.4
    
    polygon = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        px = x_center + radius * math.cos(angle)
        py = y_center + radius * math.sin(angle)
        polygon.extend([float(px), float(py)])
    
    return [polygon]  # COCO format expects list of polygons

def calculate_polygon_area(polygon: List[float]) -> float:
    """Calculate area of polygon using shoelace formula"""
    # Convert flat list to coordinate pairs
    coords = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
    
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def update_annotations_with_masks(coco_data: Dict[str, Any], mask_type: str = "elliptical") -> Dict[str, Any]:
    """Update COCO annotations with generated segmentation masks"""
    logger.info(f"Generating {mask_type} masks for {len(coco_data['annotations'])} annotations...")
    
    updated_count = 0
    total_area_bbox = 0
    total_area_mask = 0
    
    for annotation in coco_data['annotations']:
        bbox = annotation['bbox']
        
        # Generate mask based on type
        if mask_type == "circular":
            segmentation = generate_circular_mask(bbox)
        else:  # elliptical
            segmentation = generate_elliptical_mask(bbox)
        
        # Update annotation
        annotation['segmentation'] = segmentation
        
        # Update area to match polygon (more accurate than bbox area)
        polygon_area = calculate_polygon_area(segmentation[0])
        total_area_bbox += annotation['area']
        total_area_mask += polygon_area
        annotation['area'] = polygon_area
        
        updated_count += 1
        
        if updated_count % 1000 == 0:
            logger.info(f"  Processed {updated_count}/{len(coco_data['annotations'])} annotations...")
    
    logger.info(f"✅ Generated {updated_count} {mask_type} masks")
    logger.info(f"📊 Average area - Bbox: {total_area_bbox/updated_count:.1f}, Mask: {total_area_mask/updated_count:.1f}")
    
    return coco_data

def process_coco_file(input_path: str, output_path: str, mask_type: str = "elliptical"):
    """Process a single COCO JSON file to add segmentation masks"""
    logger.info(f"📁 Processing: {input_path}")
    
    # Load COCO data
    with open(input_path, 'r') as f:
        coco_data = json.load(f)
    
    logger.info(f"📊 Found {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Check current segmentation status
    empty_segmentations = sum(1 for ann in coco_data['annotations'] if not ann.get('segmentation', []))
    logger.info(f"📋 Empty segmentations: {empty_segmentations}/{len(coco_data['annotations'])}")
    
    if empty_segmentations == 0:
        logger.info("✅ All annotations already have segmentation masks")
        return
    
    # Update annotations with masks
    updated_data = update_annotations_with_masks(coco_data, mask_type)
    
    # Save updated data
    with open(output_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    logger.info(f"✅ Saved updated annotations to: {output_path}")

def generate_masks_for_dataset(mask_type: str = "elliptical"):
    """Generate segmentation masks for the entire dataset"""
    logger.info("🎨 Generating Segmentation Masks for Entire Dataset")
    logger.info("=" * 60)
    
    # Setup paths
    dataset_dir = Path("outputs/dataset")
    backup_dir = Path("outputs/dataset/backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to process
    files_to_process = [
        ("annotations_train.json", f"annotations_train_{mask_type}.json"),
        ("annotations_val.json", f"annotations_val_{mask_type}.json"),
        ("annotations_test.json", f"annotations_test_{mask_type}.json")
    ]
    
    total_annotations = 0
    
    for input_file, output_file in files_to_process:
        input_path = dataset_dir / input_file
        output_path = dataset_dir / output_file
        backup_path = backup_dir / input_file
        
        if not input_path.exists():
            logger.warning(f"❌ File not found: {input_path}")
            continue
        
        # Create backup of original
        if not backup_path.exists():
            logger.info(f"💾 Creating backup: {backup_path}")
            with open(input_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
        
        # Process the file
        with open(input_path, 'r') as f:
            data = json.load(f)
            total_annotations += len(data['annotations'])
        
        process_coco_file(str(input_path), str(output_path), mask_type)
    
    logger.info(f"\n✅ Mask generation complete!")
    logger.info(f"📊 Total annotations processed: {total_annotations}")
    logger.info(f"🎨 Mask type: {mask_type}")
    logger.info(f"📁 Updated files saved with _{mask_type} suffix")
    logger.info(f"💾 Original files backed up to: {backup_dir}")
    
    return total_annotations

def validate_generated_masks(mask_type: str = "elliptical"):
    """Validate the generated masks"""
    logger.info(f"\n🔍 Validating generated {mask_type} masks...")
    
    dataset_dir = Path("outputs/dataset")
    files_to_check = [
        f"annotations_train_{mask_type}.json",
        f"annotations_val_{mask_type}.json", 
        f"annotations_test_{mask_type}.json"
    ]
    
    total_validated = 0
    
    for filename in files_to_check:
        filepath = dataset_dir / filename
        if not filepath.exists():
            logger.warning(f"❌ File not found: {filepath}")
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check segmentation masks
        annotations = data['annotations']
        empty_masks = sum(1 for ann in annotations if not ann.get('segmentation', []))
        valid_masks = len(annotations) - empty_masks
        
        logger.info(f"📁 {filename}: {valid_masks}/{len(annotations)} annotations have masks")
        total_validated += valid_masks
        
        if empty_masks > 0:
            logger.warning(f"⚠️  {empty_masks} annotations still have empty masks!")
    
    logger.info(f"✅ Total validated annotations with masks: {total_validated}")
    return total_validated

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate segmentation masks for COCO dataset")
    parser.add_argument("--mask-type", choices=["circular", "elliptical"], default="elliptical",
                       help="Type of mask to generate (default: elliptical)")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing masks, don't generate new ones")
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_generated_masks(args.mask_type)
    else:
        total_processed = generate_masks_for_dataset(args.mask_type)
        validate_generated_masks(args.mask_type)
        
        logger.info(f"\n🎉 Successfully generated {args.mask_type} masks for {total_processed} annotations!")
        logger.info(f"📋 Next steps:")
        logger.info(f"   1. Update training config to use annotations_*_{args.mask_type}.json files")
        logger.info(f"   2. Configure Mask R-CNN (MASK_ON: True)")
        logger.info(f"   3. Test training pipeline with generated masks") 