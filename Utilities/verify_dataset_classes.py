#!/usr/bin/env python3
import os
import yaml
from pathlib import Path
from collections import Counter

def verify_dataset_classes(dataset_path):
    """
    Verify that all label files in the dataset only contain the specified classes (0, 1, 2, 3)
    """
    # Load the class names from data.yaml
    with open(Path(dataset_path) / "data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config["names"]
    print(f"Dataset: {dataset_path}")
    print(f"Expected classes: {class_names}")
    
    # Initialize counters
    valid_classes = set(range(len(class_names)))  # Should be {0, 1, 2, 3}
    class_counts = Counter()
    total_labels = 0
    invalid_files = []
    
    # Check each split
    for split in ["train", "val", "test"]:
        labels_dir = Path(dataset_path) / "labels" / split
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist")
            continue
        
        print(f"\nChecking {split} split...")
        
        # Process all label files
        label_files = list(labels_dir.glob("*.txt"))
        print(f"Found {len(label_files)} label files")
        
        # Process each label file
        for label_file in label_files:
            with open(label_file, "r") as f:
                lines = f.readlines()
            
            # Check each class in the label file
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_idx = int(parts[0])
                    total_labels += 1
                    
                    # Check if the class index is valid
                    if class_idx not in valid_classes:
                        invalid_files.append((str(label_file), class_idx))
                    else:
                        class_counts[class_idx] += 1
                        
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {label_file}: {line.strip()}")
    
    # Report results
    print("\nResults:")
    print(f"Total objects/labels found: {total_labels}")
    print("\nClass distribution:")
    for class_idx in sorted(class_counts.keys()):
        class_name = class_names[class_idx]
        count = class_counts[class_idx]
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"  Class {class_idx} ({class_name}): {count} instances ({percentage:.2f}%)")
    
    # Report any invalid files
    if invalid_files:
        print(f"\nWARNING: Found {len(invalid_files)} instances with invalid class indices!")
        print("First 10 instances with invalid classes:")
        for i, (file_path, class_idx) in enumerate(invalid_files[:10]):
            print(f"  {file_path}: invalid class index {class_idx}")
    else:
        print("\nAll classes are valid! Dataset contains only the 4 specified classes.")
    
    return len(invalid_files) == 0

if __name__ == "__main__":
    print("=== Verifying 80/10/10 Dataset ===")
    valid_80_10_10 = verify_dataset_classes("merged_dataset")
    
    print("\n\n=== Verifying 75/15/15 Dataset ===")
    valid_75_15_15 = verify_dataset_classes("merged_dataset_75_15")
    
    if valid_80_10_10 and valid_75_15_15:
        print("\n✅ SUCCESS: Both datasets contain only the 4 specified classes.")
    else:
        print("\n❌ ERROR: One or both datasets contain invalid classes!") 