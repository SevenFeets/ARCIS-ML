# Merged Weapons Dataset

This dataset is a merger of 5 different weapon detection datasets with the following modifications:

## Class Standardization

The merged dataset contains 4 standardized classes:
- `Knife`
- `Pistol`
- `weapon` (general category for various weapons)
- `rifle`

## Modifications Applied

1. **Dataset 1**: 
   - Removed class `person` and all related images/labels

2. **Dataset 2**:
   - Changed `Handgun` to `weapon`

3. **Dataset 3**:
   - Removed classes: `blunt object`, `knife_attacker`, `person`, `Gunmen`
   - Changed classes: `submachine-gun`, `shot-gun` to `weapon`

4. **Dataset 4**:
   - Changed `heavy-weapon`, `gun` to `weapon`

5. **Dataset 5**:
   - Changed `Gun`, `handgun` to `weapon`
   - Removed class `Grenade` and all related images/labels

## Dataset Splits

Two versions of the merged dataset were created with different train/validation/test splits:

1. **80/10/10 Split**
   - 80% training data
   - 10% validation data
   - 10% testing data

2. **75/15/10 Split**
   - 75% training data
   - 15% validation data
   - 10% testing data

## Directory Structure

```
merged_dataset/            # 80/10/10 split
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

merged_dataset_75_15/      # 75/15/10 split
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Usage

To use the dataset with YOLOv8:

```python
from ultralytics import YOLO

# Train a model with the 80/10/10 split
model = YOLO('yolov8n.pt')
model.train(data='merged_dataset/data.yaml', epochs=100)

# Or train with the 75/15/10 split
model = YOLO('yolov8n.pt')
model.train(data='merged_dataset_75_15/data.yaml', epochs=100)
```

## Actual On-Disk Splits (Counts and Percentages)

The following reflect the counts currently present in this repository.

- merged_dataset_80_10_10_FULL (80/10/10 target)
  - images: train=198,699; val=24,837; test=24,838; total=248,374
  - split: train≈80.00%, val≈10.00%, test≈10.00%

- merged_dataset (80/10/10 target)
  - images: train=90,349; val=16,316; test=16,320; total=122,985
  - split: train≈73.49%, val≈13.27%, test≈13.28%

- merged_dataset_75_15 (75/15/10 target)
  - images: train=84,702; val=24,475; test=16,320; total=125,497
  - split: train≈67.49%, val≈19.51%, test≈13.01%

Notes:
- The FULL dataset closely matches the intended 80/10/10 split and includes nested `train/`, `val/`, and `test/` directories with `images/` and (where applicable) `labels/`.
- The other two datasets reflect earlier merges and may deviate from the intended ratios due to source set balances. If you need exact ratios (e.g., 75/15/10), we can re-generate splits to match precisely.