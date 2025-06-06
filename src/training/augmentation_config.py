"""
Data Augmentation Configuration for Pineapple Detection
Optimized for small dataset (176 images) with high annotation density (73+ per image)
"""

from detectron2.data import transforms as T
import numpy as np

class PineappleAugmentationConfig:
    """
    Comprehensive augmentation strategy for pineapple detection
    Designed to maximize data diversity while preserving annotation integrity
    """
    
    @staticmethod
    def get_training_augmentations():
        """
        Get training augmentations optimized for pineapple detection
        
        Key considerations:
        - Small dataset (176 images) needs aggressive augmentation
        - High annotation density (73+ per image) requires careful geometric transforms
        - Pineapples are roughly circular/oval - rotations are beneficial
        - Agricultural imagery has consistent lighting patterns
        """
        
        augmentations = [
            # Geometric transformations - preserve annotation relationships
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(
                angle=[-20, 20], 
                expand=False,  # Don't expand to preserve image dimensions
                center=None,   # Random center rotation for variety
                sample_style="range"
            ),
            
            # Multi-scale training for robustness
            T.ResizeShortestEdge(
                short_edge_length=[800, 900, 1000, 1100, 1200],
                max_size=1368,  # Preserve native width
                sample_style="choice"
            ),
            
            # Color space augmentations - agricultural imagery specific
            T.RandomBrightness(intensity_min=0.7, intensity_max=1.3),
            T.RandomContrast(intensity_min=0.8, intensity_max=1.2),
            T.RandomSaturation(intensity_min=0.8, intensity_max=1.2),
            T.RandomLighting(scale=0.1),
            
            # Blur augmentation for varying focus conditions
            T.RandomApply(
                T.GaussianBlur(sigma=[0.1, 2.0]),
                prob=0.2
            ),
            
            # Cutout augmentation for robustness to occlusion
            # Careful parameters due to high annotation density
            T.RandomApply(
                T.RandomCrop(
                    crop_type="relative",
                    crop_size=[0.9, 0.9]  # Mild cropping to avoid losing too many annotations
                ),
                prob=0.1
            ),
        ]
        
        return augmentations
    
    @staticmethod
    def get_validation_augmentations():
        """
        Minimal augmentations for validation - mainly resizing
        """
        return [
            T.ResizeShortestEdge(
                short_edge_length=1000,
                max_size=1368,
                sample_style="choice"
            ),
        ]
    
    @staticmethod
    def get_test_augmentations():
        """
        Test-time augmentations for inference
        """
        return [
            T.ResizeShortestEdge(
                short_edge_length=1000,
                max_size=1368,
                sample_style="choice"
            ),
        ]

class AdvancedAugmentationConfig:
    """
    Advanced augmentation techniques for challenging scenarios
    Use when baseline augmentations are insufficient
    """
    
    @staticmethod
    def get_heavy_augmentations():
        """
        Heavy augmentation strategy for extreme data scarcity
        WARNING: May degrade annotation quality - use with caution
        """
        
        augmentations = [
            # Basic geometric
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomRotation(angle=[-30, 30], expand=False),
            
            # Aggressive multi-scale
            T.ResizeShortestEdge(
                short_edge_length=[600, 700, 800, 900, 1000, 1100, 1200, 1300],
                max_size=1400,
                sample_style="choice"
            ),
            
            # Strong color augmentations
            T.RandomBrightness(intensity_min=0.5, intensity_max=1.5),
            T.RandomContrast(intensity_min=0.6, intensity_max=1.4),
            T.RandomSaturation(intensity_min=0.6, intensity_max=1.4),
            T.RandomLighting(scale=0.2),
            
            # Noise and blur
            T.RandomApply(T.GaussianBlur(sigma=[0.1, 3.0]), prob=0.3),
            
            # Elastic deformation (mild)
            T.RandomApply(
                T.RandomCrop(crop_type="relative", crop_size=[0.85, 0.85]),
                prob=0.15
            ),
        ]
        
        return augmentations

class SmallObjectAugmentationConfig:
    """
    Specialized augmentations for small object detection
    Optimized for pineapples with average size 148.63 pixels²
    """
    
    @staticmethod
    def get_small_object_augmentations():
        """
        Augmentations specifically designed for small object detection
        """
        
        augmentations = [
            # Preserve small objects during geometric transforms
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            
            # Conservative rotation to preserve small object integrity
            T.RandomRotation(angle=[-15, 15], expand=False),
            
            # Multi-scale training with emphasis on higher resolutions
            T.ResizeShortestEdge(
                short_edge_length=[1000, 1100, 1200, 1300],  # Higher resolutions for small objects
                max_size=1500,
                sample_style="choice"
            ),
            
            # Mild color augmentations to preserve small object visibility
            T.RandomBrightness(intensity_min=0.8, intensity_max=1.2),
            T.RandomContrast(intensity_min=0.9, intensity_max=1.1),
            T.RandomSaturation(intensity_min=0.9, intensity_max=1.1),
            
            # NO cropping or heavy geometric transforms that might eliminate small objects
        ]
        
        return augmentations

# Utility function to create custom augmentation pipeline
def create_custom_augmentation_pipeline(
    flip_prob=0.5,
    rotation_range=20,
    brightness_range=(0.7, 1.3),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    blur_prob=0.2,
    scale_range=(800, 1200),
    max_size=1368
):
    """
    Create a custom augmentation pipeline with specified parameters
    
    Args:
        flip_prob: Probability of horizontal flip
        rotation_range: Range of rotation angles in degrees
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment
        saturation_range: Range for saturation adjustment
        blur_prob: Probability of applying Gaussian blur
        scale_range: Range of scales for multi-scale training
        max_size: Maximum image size
    
    Returns:
        List of augmentation transforms
    """
    
    augmentations = []
    
    # Geometric transforms
    if flip_prob > 0:
        augmentations.append(T.RandomFlip(prob=flip_prob, horizontal=True, vertical=False))
    
    if rotation_range > 0:
        augmentations.append(T.RandomRotation(
            angle=[-rotation_range, rotation_range],
            expand=False
        ))
    
    # Scale augmentation
    if isinstance(scale_range, tuple) and len(scale_range) == 2:
        scale_list = list(range(scale_range[0], scale_range[1] + 1, 100))
        augmentations.append(T.ResizeShortestEdge(
            short_edge_length=scale_list,
            max_size=max_size,
            sample_style="choice"
        ))
    
    # Color augmentations
    if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
        augmentations.append(T.RandomBrightness(
            intensity_min=brightness_range[0],
            intensity_max=brightness_range[1]
        ))
    
    if contrast_range[0] != 1.0 or contrast_range[1] != 1.0:
        augmentations.append(T.RandomContrast(
            intensity_min=contrast_range[0],
            intensity_max=contrast_range[1]
        ))
    
    if saturation_range[0] != 1.0 or saturation_range[1] != 1.0:
        augmentations.append(T.RandomSaturation(
            intensity_min=saturation_range[0],
            intensity_max=saturation_range[1]
        ))
    
    # Blur augmentation
    if blur_prob > 0:
        augmentations.append(T.RandomApply(
            T.GaussianBlur(sigma=[0.1, 2.0]),
            prob=blur_prob
        ))
    
    return augmentations

# Preset configurations for different training phases
AUGMENTATION_PRESETS = {
    "baseline": PineappleAugmentationConfig.get_training_augmentations(),
    "small_objects": SmallObjectAugmentationConfig.get_small_object_augmentations(),
    "heavy": AdvancedAugmentationConfig.get_heavy_augmentations(),
    "validation": PineappleAugmentationConfig.get_validation_augmentations(),
    "test": PineappleAugmentationConfig.get_test_augmentations(),
    
    # Conservative preset for high annotation density
    "conservative": create_custom_augmentation_pipeline(
        flip_prob=0.5,
        rotation_range=10,
        brightness_range=(0.9, 1.1),
        contrast_range=(0.95, 1.05),
        saturation_range=(0.95, 1.05),
        blur_prob=0.1,
        scale_range=(900, 1100),
        max_size=1368
    ),
    
    # Aggressive preset for maximum data diversity
    "aggressive": create_custom_augmentation_pipeline(
        flip_prob=0.7,
        rotation_range=25,
        brightness_range=(0.6, 1.4),
        contrast_range=(0.7, 1.3),
        saturation_range=(0.7, 1.3),
        blur_prob=0.3,
        scale_range=(700, 1300),
        max_size=1400
    ),
} 