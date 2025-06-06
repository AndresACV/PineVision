#!/usr/bin/env python3
"""
Launch Mask R-CNN Training for Pineapple Detection
Uses generated elliptical masks and optimized configuration
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.utils.logger import setup_logger
import logging

# Setup logging
setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_datasets():
    """Register pineapple datasets with generated masks"""
    data_root = str(project_root)
    
    # Dataset paths with generated elliptical masks
    train_json = os.path.join(data_root, "outputs/dataset/annotations_train_elliptical.json")
    val_json = os.path.join(data_root, "outputs/dataset/annotations_val_elliptical.json")
    test_json = os.path.join(data_root, "outputs/dataset/annotations_test_elliptical.json")
    images_dir = os.path.join(data_root, "src/data/images")
    
    # Register datasets with masked names to match config
    datasets = [
        ("pineapple_train_masked", train_json),
        ("pineapple_val_masked", val_json),
        ("pineapple_test_masked", test_json)
    ]
    
    for dataset_name, annotation_file in datasets:
        # Clear existing registrations
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)
        
        register_coco_instances(dataset_name, {}, annotation_file, images_dir)
        
        # Set metadata
        MetadataCatalog.get(dataset_name).set(
            thing_classes=["pineapple"],
            thing_colors=[(255, 255, 0)]  # Yellow for pineapples
        )
    
    logger.info("✅ Datasets with masks registered successfully")

class MaskRCNNTrainer(DefaultTrainer):
    """Custom trainer for Mask R-CNN with evaluation"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def main():
    """Launch Mask R-CNN training"""
    logger.info("🚀 Launching Mask R-CNN Training for Pineapple Detection")
    logger.info("=" * 60)
    
    # Register datasets
    register_datasets()
    
    # Load configuration
    cfg = get_cfg()
    config_file = os.path.join(project_root, "config/pineapple_maskrcnn_clean.yaml")
    cfg.merge_from_file(config_file)
    
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    logger.info(f"📁 Configuration: {config_file}")
    logger.info(f"📂 Output directory: {cfg.OUTPUT_DIR}")
    logger.info(f"🎯 Training dataset: {cfg.DATASETS.TRAIN}")
    logger.info(f"🔍 Validation dataset: {cfg.DATASETS.TEST}")
    logger.info(f"🔢 Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"🔄 Max iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"📊 Evaluation period: {cfg.TEST.EVAL_PERIOD}")
    logger.info(f"🎭 Mask training: {cfg.MODEL.MASK_ON}")
    
    # Estimate training time
    iterations_per_epoch = 140 // cfg.SOLVER.IMS_PER_BATCH  # 35 iterations per epoch
    total_epochs = cfg.SOLVER.MAX_ITER // iterations_per_epoch  # ~286 epochs
    estimated_time_hours = (cfg.SOLVER.MAX_ITER * 2.08) / 3600  # Based on test: 2.08s per iteration
    
    logger.info(f"⏱️  Estimated training time: {estimated_time_hours:.1f} hours")
    logger.info(f"📈 Approximately {total_epochs:.0f} epochs ({iterations_per_epoch} iterations per epoch)")
    
    # Create trainer
    trainer = MaskRCNNTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    logger.info("🎯 Starting training...")
    trainer.train()
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📂 Model saved to: {cfg.OUTPUT_DIR}")
    logger.info(f"📊 TensorBoard logs: tensorboard --logdir {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    main() 