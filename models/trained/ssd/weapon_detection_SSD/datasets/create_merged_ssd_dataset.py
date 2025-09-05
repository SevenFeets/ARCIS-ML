#!/usr/bin/env python3
import os
import sys
import shutil
import glob
import yaml
import cv2
import numpy as np
from pathlib import Path
import argparse
import random
import hashlib

def create_merged_ssd_dataset():
    """
    Create a merged dataset from all 5 datasets specifically formatted for SSD architecture.
    This function handles:
    1. Merging and harmonizing class names across datasets
    2. Creating train/val/test splits (80/10/10 by default)
    3. Ensuring all labels are formatted for SSD architecture
    4. Creating a data.yaml file with SSD-specific metadata
    """
    print("Creating merged dataset specifically for SSD architecture...")
    
    # Define paths to source datasets
    source_datasets = [
        "newDatasets/dataset_1",
        "newDatasets/dataset_2", 
        "newDatasets/dataset_3",
        "newDatasets/dataset_4",
        "newDatasets/dataset_5"
    ]
    
    # Define output path for merged dataset
    output_dir = Path("merged_ssd_dataset")
    
    # Create output directories
    train_img_dir = output_dir / "images" / "train"
    val_img_dir = output_dir / "images" / "val"
    test_img_dir = output_dir / "images" / "test"
    train_label_dir = output_dir / "labels" / "train"
    val_label_dir = output_dir / "labels" / "val"
    test_label_dir = output_dir / "labels" / "test"
    
    # Clean up existing directory if it exists (to avoid accumulating files on repeated runs)
    if output_dir.exists():
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Define a unified class mapping
    # This maps specific class names from each dataset to a common set
    # The class IDs will be reindexed to match SSD expectations
    unified_classes = {
        "pistol": 0,
        "handgun": 0, 
        "Handgun": 0,
        "Pistol": 0,
        "gun": 0,
        "Gun": 0,
        "weapon": 0,  # Generic weapon will be mapped to pistol for simplicity
        
        "rifle": 1,
        "Rifle": 1,
        "shot-gun": 1,
        "submachine-gun": 1,
        "heavy-weapon": 1,
        
        "knife": 2,
        "Knife": 2,
        "knife_attacker": 2,
        "blunt object": 2,  # For simplicity, mapping blunt objects to knife category
        
        "Grenade": 3,
        
        # Person is kept separate if we want to do person-with-weapon detection
        "person": 4,
        "Gunmen": 4,
    }
    
    # Simplified class names for our unified dataset
    final_class_names = ["pistol", "rifle", "knife", "grenade", "person"]
    
    # Track statistics
    total_images = 0
    train_count = 0
    val_count = 0
    test_count = 0
    class_counts = {name: 0 for name in final_class_names}
    
    # Process each dataset
    for dataset_path in source_datasets:
        dataset_path = Path(dataset_path)
        print(f"\nProcessing dataset: {dataset_path}")
        
        # Load dataset config
        yaml_path = dataset_path / "data.yaml"
        if not yaml_path.exists():
            print(f"Warning: data.yaml not found in {dataset_path}, skipping")
            continue
            
        with open(yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Get original class names
        orig_class_names = dataset_config.get('names', [])
        print(f"Original classes: {orig_class_names}")
        
        # Process train, val, and test sets
        for split in ['train', 'valid', 'test']:
            # Some datasets use 'val' and others use 'valid'
            split_dir_name = 'val' if split == 'valid' and not (dataset_path / split).exists() else split
            
            # Define source directories
            src_img_dir = dataset_path / split_dir_name / 'images'
            src_label_dir = dataset_path / split_dir_name / 'labels'
            
            if not src_img_dir.exists() or not src_label_dir.exists():
                print(f"Warning: {split} directories not found in {dataset_path}, skipping")
                continue
            
            # Get all image files
            img_files = list(src_img_dir.glob('*.jpg')) + list(src_img_dir.glob('*.jpeg')) + list(src_img_dir.glob('*.png'))
            
            print(f"Found {len(img_files)} images in {split} set")
            total_images += len(img_files)
            
            # Process each image and its corresponding label
            for img_file in img_files:
                # Determine destination directories based on split and random assignment
                # For the original dataset, maintain the split
                # For simplicity, we'll use 80/10/10 for the final merged dataset
                if split == 'train':
                    if random.random() < 0.9:  # 90% of train goes to train
                        dest_img_dir = train_img_dir
                        dest_label_dir = train_label_dir
                        train_count += 1
                    else:  # 10% of train goes to val
                        dest_img_dir = val_img_dir
                        dest_label_dir = val_label_dir
                        val_count += 1
                elif split == 'valid' or split == 'val':
                    if random.random() < 0.5:  # 50% of val goes to val
                        dest_img_dir = val_img_dir
                        dest_label_dir = val_label_dir
                        val_count += 1
                    else:  # 50% of val goes to test
                        dest_img_dir = test_img_dir
                        dest_label_dir = test_label_dir
                        test_count += 1
                else:  # test
                    dest_img_dir = test_img_dir
                    dest_label_dir = test_label_dir
                    test_count += 1
                
                # Generate a short unique filename using hash to avoid long filenames
                # Use dataset name + original filename + random suffix to ensure uniqueness
                hash_input = f"{dataset_path.name}_{img_file.stem}_{random.randint(1000, 9999)}"
                file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]  # 12 chars is sufficient for uniqueness
                
                # Create short filenames
                new_img_filename = f"{dataset_path.name.split('_')[1]}_{file_hash}{img_file.suffix}"
                new_label_filename = f"{dataset_path.name.split('_')[1]}_{file_hash}.txt"
                
                # Copy image file
                try:
                    shutil.copy(img_file, dest_img_dir / new_img_filename)
                except (OSError, shutil.Error) as e:
                    print(f"Error copying image {img_file}: {e}")
                    continue  # Skip this file and continue with the next
                
                # Process and copy label file
                label_file = src_label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.read().strip().split('\n')
                        
                        new_lines = []
                        for line in lines:
                            if line.strip():
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    # Get original class ID and map to new unified class ID
                                    orig_class_id = int(parts[0])
                                    if orig_class_id < len(orig_class_names):
                                        orig_class_name = orig_class_names[orig_class_id]
                                        if orig_class_name in unified_classes:
                                            new_class_id = unified_classes[orig_class_name]
                                            # Count objects by class
                                            if new_class_id < len(final_class_names):
                                                class_counts[final_class_names[new_class_id]] += 1
                                            
                                            # Create new label line with mapped class ID
                                            new_line = f"{new_class_id} {' '.join(parts[1:])}"
                                            new_lines.append(new_line)
                        
                        # Write new label file
                        with open(dest_label_dir / new_label_filename, 'w') as f:
                            f.write('\n'.join(new_lines))
                    except Exception as e:
                        print(f"Error processing label {label_file}: {e}")
                        # Remove the corresponding image if we couldn't process the label
                        try:
                            os.remove(dest_img_dir / new_img_filename)
                        except:
                            pass
    
    # Create data.yaml file for the merged dataset
    merged_yaml = {
        'train': '../images/train',
        'val': '../images/val',
        'test': '../images/test',
        'nc': len(final_class_names),
        'names': final_class_names,
        
        # Add SSD-specific metadata
        'ssd_config': {
            'image_size': 320,
            'feature_map_sizes': [10, 20, 40],  # Feature maps for 320x320 input: 10x10, 20x20, 40x40
            'anchor_sizes': [[0.1, 0.2], [0.2, 0.4], [0.4, 0.8]],  # Anchor box sizes relative to image size
            'aspect_ratios': [1.0, 2.0, 0.5],  # Default aspect ratios for SSD
            'total_boxes': 6300,  # Total predicted boxes (3 anchors per cell on each feature map)
            'box_variance': [0.1, 0.1, 0.2, 0.2]  # Box variance for SSD
        }
    }
    
    # Ensure the YAML file is properly saved
    try:
        yaml_path = output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(merged_yaml, f)
        print(f"Dataset configuration saved to: {yaml_path}")
    except Exception as e:
        print(f"Error saving data.yaml: {e}")
        sys.exit(1)
    
    # Verify dataset creation
    train_images = len(list(train_img_dir.glob('*.*')))
    val_images = len(list(val_img_dir.glob('*.*')))
    test_images = len(list(test_img_dir.glob('*.*')))
    
    # Print summary
    print("\n=== Merged SSD Dataset Summary ===")
    print(f"Total processed images: {total_images}")
    print(f"Images successfully copied:")
    print(f"  Training images: {train_images}")
    print(f"  Validation images: {val_images}")
    print(f"  Test images: {test_images}")
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} objects")
    
    print(f"\nDataset saved to: {output_dir}")
    print("The dataset is now formatted specifically for SSD architecture with appropriate feature map sizes.")
    
    # Return success only if we have images in all splits and the YAML file exists
    if train_images > 0 and val_images > 0 and test_images > 0 and yaml_path.exists():
        return True
    else:
        print("Warning: Dataset creation may be incomplete.")
        return False

if __name__ == "__main__":
    success = create_merged_ssd_dataset()
    if not success:
        sys.exit(1) 