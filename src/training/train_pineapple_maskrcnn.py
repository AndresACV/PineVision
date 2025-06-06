#!/usr/bin/env python3
"""
Pineapple Detection Training Script
Optimized for high annotation density and RTX 3070 memory constraints
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import torch.multiprocessing as mp
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, HookBase
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineappleTrainer(DefaultTrainer):
    """
    Custom trainer for pineapple detection with enhanced monitoring and optimization
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build COCO evaluator for pineapple dataset"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader with custom augmentations for small dataset"""
        from detectron2.data import build_detection_train_loader
        
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.8, 1.2),
            T.RandomContrast(0.8, 1.2),
            T.RandomSaturation(0.8, 1.2),
            T.RandomLighting(0.1),
            T.RandomRotation(angle=[-15, 15], expand=False),
            T.ResizeShortestEdge(
                short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                sample_style="choice"
            ),
        ])
        return build_detection_train_loader(cfg, mapper=mapper)
    
    def build_hooks(self):
        """Build training hooks with custom monitoring"""
        hooks = super().build_hooks()
        
        # Add custom hooks for monitoring high annotation density
        hooks.append(MemoryMonitorHook())
        hooks.append(ValidationHook(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            dataset_name=self.cfg.DATASETS.TEST[0],
            cfg=self.cfg
        ))
        
        return hooks

class MemoryMonitorHook(HookBase):
    """Monitor GPU memory usage during training"""
    
    def after_step(self):
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            
            storage = get_event_storage()
            storage.put_scalar("gpu_memory/allocated_gb", memory_allocated)
            storage.put_scalar("gpu_memory/reserved_gb", memory_reserved)
            
            if memory_allocated > 7.0:  # Warning if using >7GB on RTX 3070
                logger.warning(f"High GPU memory usage: {memory_allocated:.2f}GB allocated")

class ValidationHook(HookBase):
    """Custom validation hook for monitoring overfitting"""
    
    def __init__(self, eval_period, dataset_name, cfg):
        self.eval_period = eval_period
        self.dataset_name = dataset_name
        self.cfg = cfg
        
    def after_step(self):
        storage = get_event_storage()
        if storage.iter % self.eval_period == 0 and storage.iter > 0:
            self._run_validation()
    
    def _run_validation(self):
        """Run validation and log metrics"""
        model = self.trainer.model
        model.eval()
        
        evaluator = COCOEvaluator(self.dataset_name, self.cfg, False)
        val_loader = build_detection_test_loader(self.cfg, self.dataset_name)
        
        results = inference_on_dataset(model, val_loader, evaluator)
        
        # Log key metrics
        storage = get_event_storage()
        if "bbox" in results:
            storage.put_scalar("validation/AP", results["bbox"]["AP"])
            storage.put_scalar("validation/AP50", results["bbox"]["AP50"])
            storage.put_scalar("validation/AP75", results["bbox"]["AP75"])
            storage.put_scalar("validation/APs", results["bbox"]["APs"])  # Small objects
        
        if "segm" in results:
            storage.put_scalar("validation/segm_AP", results["segm"]["AP"])
            storage.put_scalar("validation/segm_AP50", results["segm"]["AP50"])
            storage.put_scalar("validation/segm_AP75", results["segm"]["AP75"])
        
        model.train()

def register_pineapple_datasets(data_root):
    """Register pineapple datasets with Detectron2 using elliptical masks"""
    
    # Define dataset paths - using elliptical mask annotations
    train_json = os.path.join(data_root, "outputs/dataset/annotations_train_elliptical.json")
    val_json = os.path.join(data_root, "outputs/dataset/annotations_val_elliptical.json")
    test_json = os.path.join(data_root, "outputs/dataset/annotations_test_elliptical.json")
    images_dir = os.path.join(data_root, "src/data/images")
    
    # Verify files exist
    for file_path in [train_json, val_json, test_json]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Register datasets with _masked suffix to match config
    register_coco_instances("pineapple_train_masked", {}, train_json, images_dir)
    register_coco_instances("pineapple_val_masked", {}, val_json, images_dir)
    register_coco_instances("pineapple_test_masked", {}, test_json, images_dir)
    
    # Set metadata
    MetadataCatalog.get("pineapple_train_masked").set(thing_classes=["pineapple"])
    MetadataCatalog.get("pineapple_val_masked").set(thing_classes=["pineapple"])
    MetadataCatalog.get("pineapple_test_masked").set(thing_classes=["pineapple"])
    
    logger.info("✅ Pineapple datasets registered successfully")
    
    # Log dataset statistics
    train_dataset = DatasetCatalog.get("pineapple_train_masked")
    val_dataset = DatasetCatalog.get("pineapple_val_masked")
    test_dataset = DatasetCatalog.get("pineapple_test_masked")
    
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"   Training: {len(train_dataset)} images")
    logger.info(f"   Validation: {len(val_dataset)} images")
    logger.info(f"   Test: {len(test_dataset)} images")
    
    # Calculate annotation statistics
    train_annotations = sum(len(sample["annotations"]) for sample in train_dataset)
    val_annotations = sum(len(sample["annotations"]) for sample in val_dataset)
    test_annotations = sum(len(sample["annotations"]) for sample in test_dataset)
    
    logger.info(f"📋 Annotation Statistics:")
    logger.info(f"   Training: {train_annotations} annotations ({train_annotations/len(train_dataset):.1f} avg per image)")
    logger.info(f"   Validation: {val_annotations} annotations ({val_annotations/len(val_dataset):.1f} avg per image)")
    logger.info(f"   Test: {test_annotations} annotations ({test_annotations/len(test_dataset):.1f} avg per image)")

def setup_cfg(config_file, data_root):
    """Setup configuration for training"""
    cfg = get_cfg()
    
    # Load configuration file
    cfg.merge_from_file(config_file)
    
    # Update paths
    cfg.OUTPUT_DIR = os.path.join(data_root, "outputs/models/pineapple_maskrcnn")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # GPU configuration
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Adjust batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory_gb:.1f}GB")
        
        if gpu_memory_gb < 10:  # RTX 3070 has 8GB
            cfg.SOLVER.IMS_PER_BATCH = 2
            logger.info("Adjusted batch size to 2 for RTX 3070")
        else:
            cfg.SOLVER.IMS_PER_BATCH = 4
    else:
        logger.warning("CUDA not available, using CPU")
        cfg.SOLVER.IMS_PER_BATCH = 1
    
    # Enable mixed precision for memory optimization
    cfg.SOLVER.AMP.ENABLED = True
    
    return cfg

def validate_training_setup(cfg):
    """Validate that training setup is correct"""
    logger.info("🔍 Validating training setup...")
    
    # Check datasets
    try:
        train_dataset = DatasetCatalog.get("pineapple_train_masked")
        val_dataset = DatasetCatalog.get("pineapple_val_masked")
        logger.info(f"✅ Datasets accessible: {len(train_dataset)} train, {len(val_dataset)} val")
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        return False
    
    # Check GPU memory
    if torch.cuda.is_available():
        try:
            # Test memory allocation
            test_tensor = torch.randn(1000, 1000, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
            logger.info("✅ GPU memory allocation test passed")
        except Exception as e:
            logger.error(f"❌ GPU memory test failed: {e}")
            return False
    
    # Check model loading
    try:
        model = build_model(cfg)
        logger.info("✅ Model architecture validation passed")
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        return False
    
    logger.info("🎉 Training setup validation completed successfully")
    return True

def main(args):
    """Main training function"""
    
    # Setup
    data_root = args.data_root or str(project_root)
    config_file = args.config_file
    
    logger.info(f"🚀 Starting Pineapple Detection Training")
    logger.info(f"📁 Data root: {data_root}")
    logger.info(f"⚙️  Config file: {config_file}")
    
    # Register datasets
    register_pineapple_datasets(data_root)
    
    # Setup configuration
    cfg = setup_cfg(config_file, data_root)
    
    # Validate setup
    if not validate_training_setup(cfg):
        logger.error("❌ Training setup validation failed")
        return False
    
    # Create trainer
    trainer = PineappleTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Final evaluation
    logger.info("📊 Running final evaluation...")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    evaluator = COCOEvaluator("pineapple_val_masked", cfg, False, cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "pineapple_val_masked")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"📈 Final Results: {results}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pineapple Detection Model")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--data-root", help="Path to data root directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num-machines", type=int, default=1, help="Number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="Machine rank")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:9999", help="Distributed training URL")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger(name="pineapple_training")
    
    # Launch training
    if args.num_gpus > 1:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
    else:
        main(args) 