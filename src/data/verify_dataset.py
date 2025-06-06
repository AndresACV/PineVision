"""
Dataset Verification Script

Verifies the pineapple dataset structure and samples annotations
before YOLO to COCO conversion.
"""

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict


def verify_dataset_structure():
    """Verify the basic dataset structure and file counts."""
    print("=== Dataset Structure Verification ===")
    
    # Check directories
    images_dir = Path("src/data/images")
    labels_dir = Path("src/data/labels")
    
    print(f"Images directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return False
    
    # Count files
    image_files = list(images_dir.glob("*.jpg"))
    label_files = list(labels_dir.glob("*.txt"))
    
    print(f"✅ Found {len(image_files)} image files (.jpg)")
    print(f"✅ Found {len(label_files)} label files (.txt)")
    
    # Check for matching files
    missing_labels = []
    missing_images = []
    
    for img_file in image_files:
        label_file = labels_dir / (img_file.stem + ".txt")
        if not label_file.exists():
            missing_labels.append(img_file.name)
    
    for label_file in label_files:
        img_file = images_dir / (label_file.stem + ".jpg")
        if not img_file.exists():
            missing_images.append(label_file.name)
    
    if missing_labels:
        print(f"⚠️  {len(missing_labels)} images missing corresponding labels:")
        for missing in missing_labels[:5]:  # Show first 5
            print(f"   - {missing}")
        if len(missing_labels) > 5:
            print(f"   ... and {len(missing_labels) - 5} more")
    
    if missing_images:
        print(f"⚠️  {len(missing_images)} labels missing corresponding images:")
        for missing in missing_images[:5]:  # Show first 5
            print(f"   - {missing}")
        if len(missing_images) > 5:
            print(f"   ... and {len(missing_images) - 5} more")
    
    matched_pairs = len(image_files) - len(missing_labels)
    print(f"✅ {matched_pairs} image-label pairs matched")
    
    return len(missing_labels) == 0 and len(missing_images) == 0


def check_image_dimensions():
    """Check image dimensions and consistency."""
    print("\n=== Image Dimensions Check ===")
    
    images_dir = Path("src/data/images")
    image_files = list(images_dir.glob("*.jpg"))
    
    if not image_files:
        print("❌ No image files found")
        return False
    
    dimensions = defaultdict(int)
    sample_images = image_files[:10]  # Check first 10 images
    
    for img_file in sample_images:
        img = cv2.imread(str(img_file))
        if img is not None:
            height, width, channels = img.shape
            dimensions[(width, height)] += 1
            print(f"📸 {img_file.name}: {width}x{height}x{channels}")
    
    print(f"\nDimension distribution (from {len(sample_images)} samples):")
    for (width, height), count in dimensions.items():
        print(f"  {width}x{height}: {count} images")
    
    # Check if all images have expected dimensions
    expected_width, expected_height = 1368, 912
    if (expected_width, expected_height) in dimensions:
        print(f"✅ Found expected dimensions {expected_width}x{expected_height}")
    else:
        print(f"⚠️  Expected dimensions {expected_width}x{expected_height} not found")
    
    return True


def sample_annotations():
    """Sample and display some annotations to verify format."""
    print("\n=== Annotation Format Check ===")
    
    labels_dir = Path("src/data/labels")
    label_files = list(labels_dir.glob("*.txt"))
    
    if not label_files:
        print("❌ No label files found")
        return False
    
    # Sample first few label files
    sample_files = label_files[:5]
    total_annotations = 0
    annotation_counts = []
    
    for label_file in sample_files:
        print(f"\n📄 {label_file.name}:")
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            annotations = [line.strip() for line in lines if line.strip()]
            annotation_counts.append(len(annotations))
            total_annotations += len(annotations)
            
            print(f"   Annotations: {len(annotations)}")
            
            for i, annotation in enumerate(annotations[:3]):  # Show first 3
                parts = annotation.split()
                if len(parts) == 5:
                    class_id, center_x, center_y, width, height = parts
                    print(f"   [{i+1}] Class: {class_id}, Center: ({center_x}, {center_y}), Size: ({width}, {height})")
                else:
                    print(f"   [{i+1}] ⚠️  Invalid format: {annotation}")
            
            if len(annotations) > 3:
                print(f"   ... and {len(annotations) - 3} more annotations")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    if annotation_counts:
        avg_annotations = sum(annotation_counts) / len(annotation_counts)
        print(f"\n📊 Annotation Statistics (from {len(sample_files)} files):")
        print(f"   Total annotations: {total_annotations}")
        print(f"   Average per file: {avg_annotations:.2f}")
        print(f"   Min per file: {min(annotation_counts)}")
        print(f"   Max per file: {max(annotation_counts)}")
    
    return True


def verify_filename_pattern():
    """Verify the filename pattern matches expected format."""
    print("\n=== Filename Pattern Check ===")
    
    images_dir = Path("src/data/images")
    image_files = list(images_dir.glob("*.jpg"))
    
    # Expected pattern: 100_0945_0001_JPG.rf.{hash}.jpg
    pattern_matches = 0
    sample_names = []
    
    for img_file in image_files[:10]:  # Check first 10
        name = img_file.name
        sample_names.append(name)
        
        # Check if it contains expected components
        if "JPG.rf." in name and name.endswith(".jpg"):
            pattern_matches += 1
    
    print(f"Sample filenames:")
    for name in sample_names[:5]:
        print(f"  📁 {name}")
    
    print(f"\n✅ {pattern_matches}/{len(sample_names)} files match expected pattern")
    
    if pattern_matches == len(sample_names):
        print("✅ All sampled files follow expected naming convention")
    else:
        print("⚠️  Some files don't follow expected naming convention")
    
    return True


def main():
    """Run all verification checks."""
    print("🔍 Starting Dataset Verification...")
    print("=" * 50)
    
    checks = [
        ("Dataset Structure", verify_dataset_structure),
        ("Image Dimensions", check_image_dimensions),
        ("Annotation Format", sample_annotations),
        ("Filename Pattern", verify_filename_pattern)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results[check_name] = False
    
    print("\n" + "=" * 50)
    print("🏁 Verification Summary:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! Dataset is ready for conversion.")
        print("💡 Next step: Run 'python src/data/yolo_to_coco.py' to convert to COCO format")
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
    
    return all_passed


if __name__ == "__main__":
    main() 