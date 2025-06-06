#!/usr/bin/env python3
"""
Visualize Pineapple Detection Results on Test Set
Shows bounding boxes and segmentation masks from trained Mask R-CNN model
"""

import os
import sys
import argparse
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_test_dataset(data_root):
    """Register test dataset for visualization"""
    test_json = os.path.join(data_root, "outputs/dataset/annotations_test_elliptical.json")
    images_dir = os.path.join(data_root, "src/data/images")
    
    if not os.path.exists(test_json):
        raise FileNotFoundError(f"Test annotations not found: {test_json}")
    
    register_coco_instances("pineapple_test_viz", {}, test_json, images_dir)
    MetadataCatalog.get("pineapple_test_viz").set(thing_classes=["pineapple"])
    
    return DatasetCatalog.get("pineapple_test_viz")

def setup_predictor(config_file, model_path, data_root):
    """Setup the predictor with trained model"""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    
    # Update model path
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold to see more detections
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure mask prediction is enabled
    cfg.MODEL.MASK_ON = True
    
    return DefaultPredictor(cfg)

def visualize_predictions(predictor, test_dataset, output_dir, max_images=10):
    """Visualize predictions on test images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metadata for visualization
    metadata = MetadataCatalog.get("pineapple_test_viz")
    
    logger.info(f"Visualizing predictions on {min(len(test_dataset), max_images)} test images...")
    
    results_summary = []
    
    for i, sample in enumerate(test_dataset[:max_images]):
        image_path = sample["file_name"]
        image_id = sample["image_id"]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            continue
            
        # Run inference
        outputs = predictor(image)
        
        # Get predictions
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)
        
        # Get ground truth count
        gt_annotations = sample["annotations"]
        gt_count = len(gt_annotations)
        
        logger.info(f"Image {i+1}: {os.path.basename(image_path)}")
        logger.info(f"  Ground Truth: {gt_count} pineapples")
        logger.info(f"  Detected: {num_detections} pineapples")
        
        # Create visualizer
        visualizer = Visualizer(
            image[:, :, ::-1],  # Convert BGR to RGB
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW
        )
        
        # Draw predictions
        vis_output = visualizer.draw_instance_predictions(instances)
        vis_image = vis_output.get_image()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(f'Test Image {i+1}: {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        
        # Original image with ground truth annotations
        gt_image = image.copy()
        gt_vis = Visualizer(
            gt_image[:, :, ::-1],
            metadata=metadata,
            scale=1.0
        )
        
        # Draw ground truth bounding boxes in GREEN
        for gt_ann in gt_annotations:
            gt_bbox = gt_ann["bbox"]  # COCO format: [x, y, width, height]
            # Convert to [x1, y1, x2, y2] format
            x1, y1, w, h = gt_bbox
            x2, y2 = x1 + w, y1 + h
            gt_vis.draw_box([x1, y1, x2, y2], edge_color="lime", line_style="-")
        
        axes[0, 0].imshow(gt_vis.output.get_image())
        axes[0, 0].set_title(f'Original + Ground Truth Labels\n({gt_count} Labeled Pineapples - GREEN boxes)', fontsize=12)
        axes[0, 0].axis('off')
        
        # Original image without annotations
        axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'Original Image\n(Clean - no annotations)', fontsize=12)
        axes[0, 1].axis('off')
        
        # Predictions with masks
        axes[1, 0].imshow(vis_image)
        axes[1, 0].set_title(f'Model Predictions with Masks\n({num_detections} Detected - COLORED masks)', fontsize=12)
        axes[1, 0].axis('off')
        
        # Comparison: GT (green) vs Predictions (red)
        comparison_image = image.copy()
        comparison_vis = Visualizer(
            comparison_image[:, :, ::-1],
            metadata=metadata,
            scale=1.0
        )
        
        # First draw ground truth in GREEN
        for gt_ann in gt_annotations:
            gt_bbox = gt_ann["bbox"]
            x1, y1, w, h = gt_bbox
            x2, y2 = x1 + w, y1 + h
            comparison_vis.draw_box([x1, y1, x2, y2], edge_color="lime", line_style="-")
        
        # Then draw predictions in RED
        for j in range(len(instances)):
            bbox = instances.pred_boxes[j].tensor.numpy()[0]
            score = instances.scores[j].item()
            comparison_vis.draw_box(bbox, edge_color="red", line_style="-")
            # Only show confidence for high-confidence detections to reduce clutter
            if score > 0.7:
                comparison_vis.draw_text(f"{score:.2f}", (bbox[0], bbox[1]-10), color="red", font_size=8)
        
        axes[1, 1].imshow(comparison_vis.output.get_image())
        axes[1, 1].set_title(f'Comparison: GT (GREEN) vs Predictions (RED)\nGT: {gt_count} | Detected: {num_detections}', fontsize=12)
        axes[1, 1].axis('off')
        
        # Save figure
        output_path = os.path.join(output_dir, f"test_result_{i+1:03d}_{os.path.basename(image_path)}.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved visualization: {output_path}")
        
        # Collect results for summary
        confidence_scores = instances.scores.numpy() if len(instances) > 0 else []
        results_summary.append({
            "image": os.path.basename(image_path),
            "ground_truth": gt_count,
            "detected": num_detections,
            "accuracy": f"{num_detections/gt_count*100:.1f}%" if gt_count > 0 else "N/A",
            "avg_confidence": f"{np.mean(confidence_scores):.3f}" if len(confidence_scores) > 0 else "N/A",
            "high_conf_detections": sum(1 for score in confidence_scores if score > 0.7)
        })
    
    return results_summary

def create_summary_report(results_summary, output_dir):
    """Create a summary report of the visualization results"""
    summary_path = os.path.join(output_dir, "detection_summary.txt")
    
    total_gt = sum(r["ground_truth"] for r in results_summary)
    total_detected = sum(r["detected"] for r in results_summary)
    
    with open(summary_path, 'w') as f:
        f.write("PINEAPPLE DETECTION RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Images Analyzed: {len(results_summary)}\n")
        f.write(f"Total Ground Truth Pineapples: {total_gt}\n")
        f.write(f"Total Detected Pineapples: {total_detected}\n")
        f.write(f"Overall Detection Rate: {total_detected/total_gt*100:.1f}%\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for result in results_summary:
            f.write(f"\nImage: {result['image']}\n")
            f.write(f"  Ground Truth: {result['ground_truth']} pineapples\n")
            f.write(f"  Detected: {result['detected']} pineapples\n")
            f.write(f"  Detection Rate: {result['accuracy']}\n")
            f.write(f"  Avg Confidence: {result['avg_confidence']}\n")
            f.write(f"  High Confidence (>0.7): {result['high_conf_detections']}\n")
    
    logger.info(f"Summary report saved: {summary_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("🍍 PINEAPPLE DETECTION RESULTS SUMMARY")
    print("="*60)
    print(f"📊 Total Images Analyzed: {len(results_summary)}")
    print(f"🎯 Total Ground Truth: {total_gt} pineapples")
    print(f"🔍 Total Detected: {total_detected} pineapples")
    print(f"📈 Overall Detection Rate: {total_detected/total_gt*100:.1f}%")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Visualize pineapple detection results")
    parser.add_argument("--config-file", default="config/pineapple_maskrcnn_clean.yaml", 
                       help="Path to config file")
    parser.add_argument("--model-path", default="outputs/models/pineapple_maskrcnn/model_final.pth",
                       help="Path to trained model")
    parser.add_argument("--data-root", default=".", help="Data root directory")
    parser.add_argument("--output-dir", default="outputs/test_visualizations", 
                       help="Output directory for visualizations")
    parser.add_argument("--max-images", type=int, default=10, 
                       help="Maximum number of test images to visualize")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    logger.info("🎯 Starting pineapple detection visualization...")
    logger.info(f"📁 Config: {args.config_file}")
    logger.info(f"🤖 Model: {args.model_path}")
    logger.info(f"📊 Output: {args.output_dir}")
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        return
    
    # Register test dataset
    test_dataset = register_test_dataset(args.data_root)
    logger.info(f"📋 Loaded {len(test_dataset)} test images")
    
    # Setup predictor
    predictor = setup_predictor(args.config_file, args.model_path, args.data_root)
    logger.info("🚀 Predictor initialized successfully")
    
    # Run visualization
    results_summary = visualize_predictions(
        predictor, test_dataset, args.output_dir, args.max_images
    )
    
    # Create summary report
    create_summary_report(results_summary, args.output_dir)
    
    logger.info("✅ Visualization completed successfully!")
    logger.info(f"📁 Check results in: {args.output_dir}")

if __name__ == "__main__":
    main() 