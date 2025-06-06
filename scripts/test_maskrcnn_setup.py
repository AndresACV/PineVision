#!/usr/bin/env python3
"""
Test Mask R-CNN training setup with generated segmentation masks
Validates all components before launching full training
"""

import os
import sys
import torch
import logging
import numpy as np
from pathlib import Path
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.data import build_detection_train_loader, build_detection_test_loader
import time

# Setup logging
setup_logger()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_datasets_with_masks():
    """Register datasets with generated segmentation masks"""
    logger.info("🔍 Dataset Registration with Generated Masks...")
    
    # Dataset paths with generated masks
    dataset_dir = Path("outputs/dataset")
    images_dir = Path("src/data/images")
    
    datasets = {
        "pineapple_train_masked": "annotations_train_elliptical.json",
        "pineapple_val_masked": "annotations_val_elliptical.json", 
        "pineapple_test_masked": "annotations_test_elliptical.json"
    }
    
    # Verify all files exist
    for dataset_name, annotation_file in datasets.items():
        annotation_path = dataset_dir / annotation_file
        
        if not annotation_path.exists():
            logger.error(f"❌ Annotation file not found: {annotation_path}")
            return False
            
        logger.info(f"✅ {dataset_name}: {annotation_path}")
    
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        return False
    
    logger.info(f"✅ Images directory: {images_dir}")
    
    # Register datasets
    try:
        for dataset_name, annotation_file in datasets.items():
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
            
            register_coco_instances(
                dataset_name,
                {},
                str(dataset_dir / annotation_file),
                str(images_dir)
            )
            
            # Set metadata
            MetadataCatalog.get(dataset_name).set(
                thing_classes=["pineapple"],
                thing_colors=[(255, 255, 0)]  # Yellow for pineapples
            )
        
        logger.info("✅ All datasets with masks registered successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset registration failed: {e}")
        return False

def test_dataset_loading():
    """Test loading datasets with masks"""
    logger.info("🔍 Dataset Loading with Masks...")
    
    try:
        from detectron2.data import DatasetCatalog
        
        # Test each dataset
        datasets = ["pineapple_train_masked", "pineapple_val_masked", "pineapple_test_masked"]
        
        for dataset_name in datasets:
            dataset_dicts = DatasetCatalog.get(dataset_name)
            logger.info(f"📊 {dataset_name}: {len(dataset_dicts)} images")
            
            # Check first few samples for masks
            sample_count = min(3, len(dataset_dicts))
            total_annotations = 0
            has_masks = 0
            
            for i in range(sample_count):
                sample = dataset_dicts[i]
                annotations = sample.get('annotations', [])
                total_annotations += len(annotations)
                
                # Check for segmentation masks
                for ann in annotations:
                    if 'segmentation' in ann and ann['segmentation']:
                        has_masks += 1
                        break
            
            logger.info(f"   📊 Sample {dataset_name}: {total_annotations} annotations")
            logger.info(f"   🎭 Samples with masks: {has_masks}/{sample_count}")
        
        # Calculate total statistics
        train_data = DatasetCatalog.get("pineapple_train_masked")
        val_data = DatasetCatalog.get("pineapple_val_masked")
        test_data = DatasetCatalog.get("pineapple_test_masked")
        
        total_annotations = sum(len(d.get('annotations', [])) for d in train_data + val_data + test_data)
        avg_per_image = total_annotations / (len(train_data) + len(val_data) + len(test_data))
        
        logger.info(f"📊 Total annotations with masks: {total_annotations}")
        logger.info(f"📋 Average annotations per image: {avg_per_image:.1f}")
        
        # Verify masks exist
        sample_annotations = train_data[0]['annotations']
        masks_found = sum(1 for ann in sample_annotations if ann.get('segmentation', []))
        logger.info(f"✅ Sample verification: {masks_found}/{len(sample_annotations)} annotations have masks")
        
        if masks_found == 0:
            logger.error("❌ No segmentation masks found in sample!")
            return False
            
        logger.info("✅ Dataset loading with masks successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        return False

def test_mask_rcnn_config():
    """Test Mask R-CNN configuration with generated masks"""
    logger.info("🔍 Mask R-CNN Configuration...")
    
    try:
        cfg = get_cfg()
        config_file = "config/pineapple_training_maskrcnn.yaml"
        
        if not os.path.exists(config_file):
            logger.error(f"❌ Config file not found: {config_file}")
            return False
        
        cfg.merge_from_file(config_file)
        
        # GPU detection and optimization
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🎮 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Adjust batch size for GPU memory
            if gpu_memory >= 8.0:  # RTX 3070 has 8GB
                cfg.SOLVER.IMS_PER_BATCH = 4  # Increased from 2
                logger.info(f"📏 Batch size set to {cfg.SOLVER.IMS_PER_BATCH} for {gpu_memory:.1f}GB GPU")
            else:
                cfg.SOLVER.IMS_PER_BATCH = 2
                logger.info(f"📏 Conservative batch size {cfg.SOLVER.IMS_PER_BATCH} for {gpu_memory:.1f}GB GPU")
        else:
            logger.warning("⚠️  No GPU detected, using CPU")
            cfg.MODEL.DEVICE = "cpu"
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   Mask training: {cfg.MODEL.MASK_ON}")
        logger.info(f"   Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        logger.info(f"   Max iterations: {cfg.SOLVER.MAX_ITER}")
        logger.info(f"   Learning rate: {cfg.SOLVER.BASE_LR}")
        logger.info(f"   Mixed precision: {cfg.SOLVER.AMP.ENABLED}")
        
        if not cfg.MODEL.MASK_ON:
            logger.error("❌ MASK_ON is False - Mask R-CNN requires mask training!")
            return False
            
        return cfg
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False

def test_model_creation_with_masks(cfg):
    """Test creating Mask R-CNN model with mask head"""
    logger.info("🔍 Mask R-CNN Model Creation...")
    
    try:
        # Build model
        model = build_model(cfg)
        logger.info("✅ Model architecture created successfully")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model moved to GPU")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model parameters:")
        logger.info(f"   Total: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,}")
        
        # Verify mask head exists
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'mask_head'):
            logger.info("✅ Mask head found in model")
        else:
            logger.error("❌ Mask head not found in model!")
            return False
        
        # Test model loading with pre-trained weights
        from detectron2.checkpoint import DetectionCheckpointer
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        logger.info("✅ Pre-trained weights loaded successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return False

def test_data_loading_with_masks(cfg):
    """Test data loader with mask annotations"""
    logger.info("🔍 Data Loading with Masks...")
    
    try:
        # Build data loader
        data_loader = build_detection_train_loader(cfg)
        logger.info("✅ Training data loader created")
        
        # Test loading a few batches
        logger.info("📈 Data loading performance with masks:")
        
        start_time = time.time()
        batch_count = 0
        total_instances = 0
        
        for i, batch in enumerate(data_loader):
            if i >= 3:  # Test 3 batches
                break
                
            batch_count += 1
            
            # Count instances in batch
            for item in batch:
                instances = item.get('instances', None)
                if instances is not None:
                    total_instances += len(instances)
                    
                    # Check for masks
                    if hasattr(instances, 'gt_masks'):
                        logger.info(f"   ✅ Batch {i+1}: {len(instances)} instances with masks")
                    else:
                        logger.warning(f"   ⚠️  Batch {i+1}: {len(instances)} instances WITHOUT masks")
        
        end_time = time.time()
        avg_time = (end_time - start_time) / batch_count if batch_count > 0 else 0
        avg_instances = total_instances / batch_count if batch_count > 0 else 0
        
        logger.info(f"   Loaded {batch_count} batches in {end_time - start_time:.2f}s")
        logger.info(f"   Average: {avg_time:.2f}s per batch")
        logger.info(f"   Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        logger.info(f"   Average instances per batch: {avg_instances:.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False

def test_memory_usage():
    """Test GPU memory usage"""
    logger.info("🔍 GPU Memory Usage...")
    
    try:
        if not torch.cuda.is_available():
            logger.warning("⚠️  No GPU available for memory testing")
            return True
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Check memory usage
        initial_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"🔍 GPU Memory status:")
        logger.info(f"   Initial allocated: {initial_memory:.2f}GB")
        logger.info(f"   Max allocated: {max_memory:.2f}GB")
        logger.info(f"   GPU capacity: {total_memory:.1f}GB")
        
        # Test allocation
        test_tensor = torch.randn(1000, 1000, device='cuda')
        after_test = torch.cuda.memory_allocated() / 1e9
        logger.info(f"   After test allocation: {after_test:.2f}GB")
        
        # Cleanup
        del test_tensor
        torch.cuda.empty_cache()
        after_cleanup = torch.cuda.memory_allocated() / 1e9
        logger.info(f"   After cleanup: {after_cleanup:.2f}GB")
        
        # Memory usage assessment
        estimated_usage = 4 * 1.5  # Batch size 4 * estimated 1.5GB per image
        if estimated_usage < total_memory * 0.9:
            logger.info(f"✅ Estimated usage {estimated_usage:.1f}GB within {total_memory:.1f}GB limit")
        else:
            logger.warning(f"⚠️  Estimated usage {estimated_usage:.1f}GB may exceed {total_memory:.1f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False

def test_quick_training_with_masks(cfg, model):
    """Test quick training with mask generation"""
    logger.info("🔍 Quick Mask R-CNN Training Test...")
    
    try:
        # Setup trainer
        class MaskRCNNTrainer(DefaultTrainer):
            @classmethod
            def build_train_loader(cls, cfg):
                return build_detection_train_loader(cfg)
        
        # Set up for quick test
        cfg.SOLVER.MAX_ITER = 5  # Very short test
        cfg.SOLVER.CHECKPOINT_PERIOD = 5
        cfg.TEST.EVAL_PERIOD = 5
        cfg.OUTPUT_DIR = "outputs/test_maskrcnn_training"
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Create trainer
        trainer = MaskRCNNTrainer(cfg)
        
        logger.info("🚀 Running quick training test...")
        start_time = time.time()
        
        trainer.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        avg_iter_time = training_time / cfg.SOLVER.MAX_ITER
        
        logger.info(f"✅ Quick training test completed in {training_time:.2f}s")
        logger.info(f"   Average time per iteration: {avg_iter_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick training test failed: {e}")
        return False

def main():
    """Run complete Mask R-CNN setup validation"""
    logger.info("🧪 Starting Mask R-CNN Training Setup Test with Generated Masks")
    logger.info("=" * 70)
    
    tests = [
        ("Dataset Registration", register_datasets_with_masks),
        ("Dataset Loading", test_dataset_loading),
        ("Configuration Loading", test_mask_rcnn_config),
        ("Memory Usage", test_memory_usage),
    ]
    
    results = {}
    cfg = None
    model = None
    
    # Run tests
    for test_name, test_func in tests:
        logger.info(f"\n🔍 {test_name}...")
        try:
            if test_name == "Configuration Loading":
                result = test_func()
                if result:
                    cfg = result
                    results[test_name] = True
                else:
                    results[test_name] = False
            else:
                results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Model creation test (needs cfg)
    if cfg:
        logger.info(f"\n🔍 Model Creation...")
        model = test_model_creation_with_masks(cfg)
        results["Model Creation"] = model is not False
        
        # Data loading test (needs cfg)
        logger.info(f"\n🔍 Data Loading...")
        results["Data Loading"] = test_data_loading_with_masks(cfg)
        
        # Quick training test (needs cfg and model)
        if model:
            logger.info(f"\n🔍 Quick Training...")
            results["Quick Training"] = test_quick_training_with_masks(cfg, model)
    
    # Results summary
    logger.info("\n" + "=" * 70)
    logger.info("🏁 TEST SUMMARY")
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
        if result:
            passed += 1
        total += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Mask R-CNN training setup is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python src/training/train_pineapple_maskrcnn.py --config-file config/pineapple_training_maskrcnn.yaml")
        logger.info("2. Monitor training progress in outputs/models/pineapple_maskrcnn/")
        logger.info("3. Check TensorBoard logs for both bbox and mask metrics")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 