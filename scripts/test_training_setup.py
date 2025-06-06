#!/usr/bin/env python3
"""
Test Training Setup Script
Validates Mask R-CNN training configuration before full training
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import json
import time
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.engine import default_setup
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_datasets():
    """Register pineapple datasets with generated masks for testing"""
    data_root = str(project_root)
    
    # Dataset paths with generated elliptical masks
    train_json = os.path.join(data_root, "outputs/dataset/annotations_train_elliptical.json")
    val_json = os.path.join(data_root, "outputs/dataset/annotations_val_elliptical.json")
    test_json = os.path.join(data_root, "outputs/dataset/annotations_test_elliptical.json")
    images_dir = os.path.join(data_root, "src/data/images")
    
    # Check if files exist
    files_to_check = [
        ("Training annotations (with masks)", train_json),
        ("Validation annotations (with masks)", val_json),
        ("Test annotations (with masks)", test_json),
        ("Images directory", images_dir)
    ]
    
    for name, path in files_to_check:
        if not os.path.exists(path):
            logger.error(f"❌ {name} not found: {path}")
            return False
        else:
            logger.info(f"✅ {name} found: {path}")
    
    # Register datasets with masked names to match config
    try:
        # Clear existing registrations if they exist
        for dataset_name in ["pineapple_train_masked", "pineapple_val_masked", "pineapple_test_masked"]:
            if dataset_name in DatasetCatalog.list():
                DatasetCatalog.remove(dataset_name)
                MetadataCatalog.remove(dataset_name)
        
        register_coco_instances("pineapple_train_masked", {}, train_json, images_dir)
        register_coco_instances("pineapple_val_masked", {}, val_json, images_dir)
        register_coco_instances("pineapple_test_masked", {}, test_json, images_dir)
        
        # Set metadata
        MetadataCatalog.get("pineapple_train_masked").set(
            thing_classes=["pineapple"],
            thing_colors=[(255, 255, 0)]  # Yellow for pineapples
        )
        MetadataCatalog.get("pineapple_val_masked").set(
            thing_classes=["pineapple"],
            thing_colors=[(255, 255, 0)]
        )
        MetadataCatalog.get("pineapple_test_masked").set(
            thing_classes=["pineapple"],
            thing_colors=[(255, 255, 0)]
        )
        
        logger.info("✅ Masked datasets registered successfully")
        
        # Verify masks exist in data
        train_dataset = DatasetCatalog.get("pineapple_train_masked")
        sample = train_dataset[0]
        sample_annotations = sample.get('annotations', [])
        masks_found = sum(1 for ann in sample_annotations if ann.get('segmentation', []))
        logger.info(f"🎭 Mask verification: {masks_found}/{len(sample_annotations)} annotations have masks")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset registration failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading and annotation parsing with masks"""
    logger.info("🔍 Testing masked dataset loading...")
    
    try:
        # Load masked datasets
        train_dataset = DatasetCatalog.get("pineapple_train_masked")
        val_dataset = DatasetCatalog.get("pineapple_val_masked")
        test_dataset = DatasetCatalog.get("pineapple_test_masked")
        
        logger.info(f"📊 Dataset sizes:")
        logger.info(f"   Training: {len(train_dataset)} images")
        logger.info(f"   Validation: {len(val_dataset)} images")
        logger.info(f"   Test: {len(test_dataset)} images")
        
        # Test loading first few samples and verify masks
        sample_count = min(3, len(train_dataset))
        total_annotations = 0
        total_masks = 0
        
        for i in range(sample_count):
            sample = train_dataset[i]
            annotations = sample.get("annotations", [])
            total_annotations += len(annotations)
            
            # Count masks
            masks_in_sample = sum(1 for ann in annotations if ann.get('segmentation', []))
            total_masks += masks_in_sample
            
            logger.info(f"   Sample {i}: {len(annotations)} annotations, {masks_in_sample} with masks")
        
        avg_annotations = total_annotations / sample_count if sample_count > 0 else 0
        mask_coverage = (total_masks / total_annotations * 100) if total_annotations > 0 else 0
        
        logger.info(f"📋 Average annotations per sample: {avg_annotations:.1f}")
        logger.info(f"🎭 Mask coverage: {mask_coverage:.1f}% ({total_masks}/{total_annotations})")
        
        if mask_coverage >= 99:
            logger.info("✅ Excellent mask coverage - ready for Mask R-CNN")
        elif mask_coverage >= 90:
            logger.info("✅ Good mask coverage")
        else:
            logger.warning(f"⚠️  Low mask coverage ({mask_coverage:.1f}%) - check mask generation")
        
        if avg_annotations > 50:
            logger.info("✅ High annotation density confirmed (good for training)")
        elif avg_annotations > 20:
            logger.info("✅ Moderate annotation density detected")
        else:
            logger.warning("⚠️  Low annotation density - consider checking conversion")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loading test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    logger.info("⚙️  Testing configuration loading...")
    
    config_file = os.path.join(project_root, "config/pineapple_maskrcnn_clean.yaml")
    
    if not os.path.exists(config_file):
        logger.error(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        
        # Update output directory
        cfg.OUTPUT_DIR = os.path.join(project_root, "outputs/test_training")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Test GPU configuration
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎮 GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory_gb:.1f}GB)")
            
            # Use the batch size from config (should be optimized for Mask R-CNN + masks)
            logger.info(f"📏 Using batch size {cfg.SOLVER.IMS_PER_BATCH} (config setting)")
            if cfg.SOLVER.IMS_PER_BATCH > 2 and gpu_memory_gb < 10:
                logger.info("⚠️  High batch size detected for RTX 3070 - monitoring memory usage")
        else:
            cfg.MODEL.DEVICE = "cpu"
            cfg.SOLVER.IMS_PER_BATCH = 1
            logger.warning("⚠️  No CUDA available, using CPU")
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"   Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        logger.info(f"   Max iterations: {cfg.SOLVER.MAX_ITER}")
        logger.info(f"   Learning rate: {cfg.SOLVER.BASE_LR}")
        logger.info(f"   Mixed precision: {cfg.SOLVER.AMP.ENABLED}")
        
        return cfg
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return None

def test_model_creation(cfg):
    """Test model creation and weight loading"""
    logger.info("🤖 Testing model creation...")
    
    try:
        # Build model
        model = build_model(cfg)
        logger.info("✅ Model architecture created successfully")
        
        # Test model on GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("✅ Model moved to GPU")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"📊 Model parameters:")
        logger.info(f"   Total: {total_params:,}")
        logger.info(f"   Trainable: {trainable_params:,}")
        
        # Test weight loading
        checkpointer = DetectionCheckpointer(model)
        if hasattr(cfg.MODEL, 'WEIGHTS') and cfg.MODEL.WEIGHTS:
            try:
                checkpointer.load(cfg.MODEL.WEIGHTS)
                logger.info("✅ Pre-trained weights loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Pre-trained weight loading failed: {e}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model creation failed: {e}")
        return None

def test_data_loading(cfg):
    """Test data loader creation and iteration"""
    logger.info("📦 Testing data loader...")
    
    try:
        # Create train data loader
        train_loader = build_detection_train_loader(cfg)
        logger.info("✅ Training data loader created")
        
        # Test data loading performance
        start_time = time.time()
        batch_count = 0
        
        for batch in train_loader:
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break
        
        load_time = time.time() - start_time
        logger.info(f"📈 Data loading performance:")
        logger.info(f"   Loaded {batch_count} batches in {load_time:.2f}s")
        logger.info(f"   Average: {load_time/batch_count:.2f}s per batch")
        
        # Analyze batch content
        if batch_count > 0:
            sample_batch = next(iter(train_loader))
            images = sample_batch[0]["image"]
            logger.info(f"   Batch image shape: {images.shape}")
            logger.info(f"   Batch size: {len(sample_batch)}")
            
            # Count annotations in batch
            total_instances = sum(len(sample["instances"]) for sample in sample_batch)
            logger.info(f"   Total instances in batch: {total_instances}")
            logger.info(f"   Average instances per image: {total_instances/len(sample_batch):.1f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loader test failed: {e}")
        return False

def test_memory_usage():
    """Test GPU memory usage"""
    if not torch.cuda.is_available():
        logger.info("⚠️  No CUDA available, skipping memory test")
        return True
    
    logger.info("💾 Testing GPU memory usage...")
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        logger.info(f"🔍 GPU Memory status:")
        logger.info(f"   Initial allocated: {initial_memory:.2f}GB")
        logger.info(f"   Max allocated: {max_memory:.2f}GB")
        logger.info(f"   GPU capacity: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Test memory allocation
        test_tensor = torch.randn(2000, 2000, device="cuda")
        current_memory = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"   After test allocation: {current_memory:.2f}GB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"   After cleanup: {final_memory:.2f}GB")
        
        if current_memory < 7.5:  # Safe for RTX 3070
            logger.info("✅ Memory usage within safe limits")
        else:
            logger.warning("⚠️  High memory usage detected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory test failed: {e}")
        return False

def run_quick_training_test(cfg, model):
    """Run a quick training test with few iterations"""
    logger.info("🚀 Running quick training test...")
    
    try:
        from detectron2.engine import DefaultTrainer
        
        # Create minimal config for testing
        test_cfg = cfg.clone()
        test_cfg.SOLVER.MAX_ITER = 5  # Very few iterations
        test_cfg.TEST.EVAL_PERIOD = 10  # No evaluation during test
        test_cfg.SOLVER.CHECKPOINT_PERIOD = 10  # No checkpointing
        
        # Create trainer
        class TestTrainer(DefaultTrainer):
            @classmethod
            def build_evaluator(cls, cfg, dataset_name, output_folder=None):
                return None  # Skip evaluation for quick test
        
        trainer = TestTrainer(test_cfg)
        
        # Run few training steps
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        logger.info(f"✅ Quick training test completed in {training_time:.2f}s")
        logger.info(f"   Average time per iteration: {training_time/5:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick training test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🧪 Starting Pineapple Detection Training Setup Test")
    logger.info("=" * 60)
    
    # Test steps
    tests = [
        ("Dataset Registration", register_datasets),
        ("Dataset Loading", test_dataset_loading),
        ("Configuration Loading", test_config_loading),
        ("Memory Usage", test_memory_usage),
    ]
    
    # Run tests
    results = {}
    cfg = None
    model = None
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 {test_name}...")
        try:
            result = test_func()
            if test_name == "Configuration Loading":
                cfg = result
                result = cfg is not None
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Additional tests that require cfg
    if cfg is not None:
        logger.info(f"\n🔍 Model Creation...")
        model = test_model_creation(cfg)
        results["Model Creation"] = model is not None
        
        logger.info(f"\n🔍 Data Loading...")
        data_result = test_data_loading(cfg)
        results["Data Loading"] = data_result
        
        if model is not None and data_result:
            logger.info(f"\n🔍 Quick Training Test...")
            training_result = run_quick_training_test(cfg, model)
            results["Quick Training"] = training_result
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 TEST SUMMARY")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Training setup is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Run: python src/training/train_pineapple_maskrcnn.py --config-file config/pineapple_training_simple.yaml")
        logger.info("2. Monitor training progress in outputs/models/pineapple_maskrcnn/")
        logger.info("3. Check TensorBoard logs for metrics")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix issues before training.")
        return False

if __name__ == "__main__":
    setup_logger()
    success = main()
    sys.exit(0 if success else 1) 