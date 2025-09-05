# Data Directory

This directory contains all datasets used in the ARCIS project.

## Structure

```
data/
├── raw/                    # Original, unprocessed datasets
│   └── dataset_1/          # Individual source datasets
├── processed/              # Processed and merged datasets
│   ├── merged_dataset/     # 80/10/10 split (122,985 images)
│   ├── merged_dataset_75_15/ # 75/15/10 split (125,497 images)
│   ├── merged_dataset_80_10_10_FULL/ # Full dataset (248,374 images)
│   └── merged_ssd_dataset/ # SSD-specific dataset
└── external/               # External data sources
    └── calibration_image_sample_data_20x128x128x3_float32.npy
```

## Dataset Information

### Primary Datasets

1. **merged_dataset_80_10_10_FULL** (Primary)
   - Total Images: 248,374
   - Classes: 19 (automatic_rifle, bazooka, civilian_aircraft, etc.)
   - Split: ~80% train, ~10% val, ~10% test
   - Structure: Nested directories with train/val/test/images and labels

2. **merged_dataset** (Standardized)
   - Total Images: 122,985
   - Classes: 4 (Knife, Pistol, weapon, rifle)
   - Split: ~73% train, ~13% val, ~13% test
   - Structure: Flat images/train, images/val, images/test

3. **merged_dataset_75_15** (Alternative)
   - Total Images: 125,497
   - Classes: 4 (Knife, Pistol, weapon, rifle)
   - Split: ~67% train, ~20% val, ~13% test
   - Structure: Flat images/train, images/val, images/test

## Usage

```python
# Load dataset configuration
from ultralytics import YOLO

# Use the primary dataset
model = YOLO('yolov8n.pt')
model.train(data='data/processed/merged_dataset_80_10_10_FULL/data.yaml')

# Use standardized dataset
model.train(data='data/processed/merged_dataset/data.yaml')
```

## Data Processing

Scripts for dataset processing and merging are located in:
- `tools/scripts/data_processing/` - Data processing utilities
- `src/scripts/merge_datasets.py` - Dataset merging script

## Notes

- All datasets are in YOLO format
- Label files (.txt) contain normalized bounding box coordinates
- Images are in JPG/PNG format
- Calibration data is stored in external/ for model optimization
