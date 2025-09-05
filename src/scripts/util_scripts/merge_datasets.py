#!/usr/bin/env python3
import os
import shutil
import yaml
import random
from pathlib import Path
from collections import defaultdict
import re

# Define paths
base_dir = Path("newDatasets")
output_dir = Path("merged_dataset")
output_dir_75_15 = Path("merged_dataset_75_15")

# Create output directories
os.makedirs(output_dir / "images" / "train", exist_ok=True)
os.makedirs(output_dir / "images" / "val", exist_ok=True)
os.makedirs(output_dir / "images" / "test", exist_ok=True)
os.makedirs(output_dir / "labels" / "train", exist_ok=True)
os.makedirs(output_dir / "labels" / "val", exist_ok=True)
os.makedirs(output_dir / "labels" / "test", exist_ok=True)

os.makedirs(output_dir_75_15 / "images" / "train", exist_ok=True)
os.makedirs(output_dir_75_15 / "images" / "val", exist_ok=True)
os.makedirs(output_dir_75_15 / "images" / "test", exist_ok=True)
os.makedirs(output_dir_75_15 / "labels" / "train", exist_ok=True)
os.makedirs(output_dir_75_15 / "labels" / "val", exist_ok=True)
os.makedirs(output_dir_75_15 / "labels" / "test", exist_ok=True)

# Define class mappings and classes to remove
class_mappings = {
    "dataset_1": {
        "classes_to_remove": ["person"],
        "class_mapping": {}
    },
    "dataset_2": {
        "classes_to_remove": [],
        "class_mapping": {"Handgun": "weapon"}
    },
    "dataset_3": {
        "classes_to_remove": ["blunt object", "knife_attacker", "person", "Gunmen"],
        "class_mapping": {"submachine-gun": "weapon", "shot-gun": "weapon"}
    },
    "dataset_4": {
        "classes_to_remove": [],
        "class_mapping": {"heavy-weapon": "weapon", "gun": "weapon"}
    },
    "dataset_5": {
        "classes_to_remove": ["Grenade"],
        "class_mapping": {"Gun": "weapon", "handgun": "weapon"}
    },
    "dataset_6": {
        "classes_to_remove": ["person"],
        "class_mapping": {}
    },
    "dataset_7": {
        "classes_to_remove": [],
        "class_mapping": {}
    },
    "dataset_8": {
        "classes_to_remove": ["d"],
        "class_mapping": {}
    },
    "dataset_9": {
        "classes_to_remove": [],
        "class_mapping": {"gun": "weapon"}
    },
    "dataset_10": {
        "classes_to_remove": ["alcohol", "gambling_element", "mariuhanna", "sexual-content", "smoking"],
        "class_mapping": {}
    }
}

# Define the unified class list
unified_classes = ["Knife", "Pistol", "weapon", "rifle"]

# Initialize a list to keep track of all files to be included
all_files = []

# Process each dataset
for dataset_num in range(1, 11):
    dataset_name = f"dataset_{dataset_num}"
    dataset_path = base_dir / dataset_name
    
    print(f"Processing {dataset_name}...")
    
    # Load data.yaml
    try:
        with open(dataset_path / "data.yaml", "r") as f:
            data_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: data.yaml not found for {dataset_name}, skipping")
        continue
    
    class_names = data_config["names"]
    
    # Create a mapping from original class indices to new unified class indices
    class_idx_mapping = {}
    classes_to_remove = class_mappings[dataset_name]["classes_to_remove"]
    
    for i, class_name in enumerate(class_names):
        # Check if class should be removed
        if class_name.lower() in [c.lower() for c in classes_to_remove]:
            class_idx_mapping[i] = -1  # Mark for removal
            continue
        
        # Apply class mapping if needed
        if class_name in class_mappings[dataset_name]["class_mapping"]:
            mapped_name = class_mappings[dataset_name]["class_mapping"][class_name]
        else:
            mapped_name = class_name
        
        # Convert class name to lowercase for case-insensitive matching
        mapped_name_lower = mapped_name.lower()
        
        # Map to unified class index
        found = False
        for j, unified_class in enumerate(unified_classes):
            if unified_class.lower() == mapped_name_lower:
                class_idx_mapping[i] = j
                found = True
                break
                
        if not found:
            # Special case for "rifle" which might be capitalized differently
            if mapped_name_lower == "rifle":
                class_idx_mapping[i] = unified_classes.index("rifle")
            # Special case for "knife" which might be capitalized differently
            elif mapped_name_lower == "knife":
                class_idx_mapping[i] = unified_classes.index("Knife")
            # Special case for "pistol" which might be capitalized differently
            elif mapped_name_lower in ["pistol", "handgun"]:
                class_idx_mapping[i] = unified_classes.index("Pistol")
            else:
                # Map any weapon-related class to "weapon"
                if any(w in mapped_name_lower for w in ["gun", "weapon"]):
                    class_idx_mapping[i] = unified_classes.index("weapon")
                else:
                    class_idx_mapping[i] = -1  # Mark class for removal if not found
    
    # Process train, validation, and test sets
    for split in ["train", "valid", "test"]:
        # Handle the case where dataset_9 only has train and dataset_10 doesn't have test
        if (dataset_name == "dataset_9" and split != "train") or (dataset_name == "dataset_10" and split == "test"):
            print(f"Skipping {split} split for {dataset_name} (directory not expected to exist)")
            continue
            
        if not os.path.exists(dataset_path / split / "images"):
            print(f"Skipping {split} split for {dataset_name} (directory not found)")
            continue
            
        # Get all image files
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        
        # Check if label directory exists
        if not os.path.exists(label_dir):
            print(f"Warning: Labels directory not found for {dataset_name}/{split}")
            continue
        
        for img_file in os.listdir(image_dir):
            if not (img_file.endswith(".jpg") or img_file.endswith(".jpeg") or img_file.endswith(".png")):
                continue
                
            # Get corresponding label file
            label_file = img_file.rsplit(".", 1)[0] + ".txt"
            label_path = label_dir / label_file
            
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {img_file} in {dataset_name}/{split}")
                continue
            
            # Read label file
            with open(label_path, "r") as f:
                label_lines = f.readlines()
            
            # Process label lines and check if any objects should be kept
            new_label_lines = []
            keep_file = False
            
            for line in label_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_idx = int(parts[0])
                
                # Skip this object if class should be removed
                if class_idx not in class_idx_mapping or class_idx_mapping[class_idx] == -1:
                    continue
                
                # Remap class index
                new_class_idx = class_idx_mapping[class_idx]
                new_line = f"{new_class_idx} {' '.join(parts[1:])}\n"
                new_label_lines.append(new_line)
                keep_file = True
            
            # Only keep files that have at least one valid object
            if keep_file:
                all_files.append({
                    "dataset": dataset_name,
                    "split": split,
                    "image": str(image_dir / img_file),
                    "label": str(label_path),
                    "new_label_lines": new_label_lines
                })

print(f"Total files to include: {len(all_files)}")

# Shuffle files for random distribution
random.shuffle(all_files)

# Split files for 80/10/10 distribution
total_files = len(all_files)
train_split = int(0.8 * total_files)
val_split = int(0.1 * total_files)

train_files = all_files[:train_split]
val_files = all_files[train_split:train_split + val_split]
test_files = all_files[train_split + val_split:]

print(f"80/10/10 Split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Split files for 75/15/15 distribution
train_split_75 = int(0.75 * total_files)
val_split_75 = int(0.15 * total_files)

train_files_75 = all_files[:train_split_75]
val_files_75 = all_files[train_split_75:train_split_75 + val_split_75]
test_files_75 = all_files[train_split_75 + val_split_75:]

print(f"75/15/15 Split - Train: {len(train_files_75)}, Val: {len(val_files_75)}, Test: {len(test_files_75)}")

# Process each split for 80/10/10
file_index = 0
for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for file_info in files:
        # Copy image
        src_img = file_info["image"]
        img_ext = os.path.splitext(src_img)[1]
        dst_img = output_dir / "images" / split_name / f"{file_index:06d}{img_ext}"
        shutil.copy2(src_img, dst_img)
        
        # Save new label
        dst_label = output_dir / "labels" / split_name / f"{file_index:06d}.txt"
        with open(dst_label, "w") as f:
            f.writelines(file_info["new_label_lines"])
            
        file_index += 1

# Process each split for 75/15/15
file_index = 0
for split_name, files in [("train", train_files_75), ("val", val_files_75), ("test", test_files_75)]:
    for file_info in files:
        # Copy image
        src_img = file_info["image"]
        img_ext = os.path.splitext(src_img)[1]
        dst_img = output_dir_75_15 / "images" / split_name / f"{file_index:06d}{img_ext}"
        shutil.copy2(src_img, dst_img)
        
        # Save new label
        dst_label = output_dir_75_15 / "labels" / split_name / f"{file_index:06d}.txt"
        with open(dst_label, "w") as f:
            f.writelines(file_info["new_label_lines"])
            
        file_index += 1

# Create data.yaml for 80/10/10
data_yaml_80_10_10 = {
    "path": str(output_dir),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(unified_classes),
    "names": unified_classes
}

with open(output_dir / "data.yaml", "w") as f:
    yaml.dump(data_yaml_80_10_10, f, default_flow_style=False)

# Create data.yaml for 75/15/15
data_yaml_75_15_15 = {
    "path": str(output_dir_75_15),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(unified_classes),
    "names": unified_classes
}

with open(output_dir_75_15 / "data.yaml", "w") as f:
    yaml.dump(data_yaml_75_15_15, f, default_flow_style=False)

print("\nDataset merging completed!")
print(f"Merged dataset (80/10/10) saved to: {output_dir}")
print(f"Merged dataset (75/15/15) saved to: {output_dir_75_15}")

# Count classes in each split for 80/10/10
class_counts = defaultdict(lambda: defaultdict(int))
for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for file_info in files:
        for line in file_info["new_label_lines"]:
            class_idx = int(line.strip().split()[0])
            class_name = unified_classes[class_idx]
            class_counts[split_name][class_name] += 1

print("\nClass distribution for 80/10/10 split:")
for split_name in ["train", "val", "test"]:
    print(f"\n{split_name.upper()} set:")
    for class_name in unified_classes:
        count = class_counts[split_name][class_name]
        print(f"  {class_name}: {count} instances")

# Count classes in each split for 75/15/15
class_counts_75 = defaultdict(lambda: defaultdict(int))
for split_name, files in [("train", train_files_75), ("val", val_files_75), ("test", test_files_75)]:
    for file_info in files:
        for line in file_info["new_label_lines"]:
            class_idx = int(line.strip().split()[0])
            class_name = unified_classes[class_idx]
            class_counts_75[split_name][class_name] += 1

print("\nClass distribution for 75/15/15 split:")
for split_name in ["train", "val", "test"]:
    print(f"\n{split_name.upper()} set:")
    for class_name in unified_classes:
        count = class_counts_75[split_name][class_name]
        print(f"  {class_name}: {count} instances") 