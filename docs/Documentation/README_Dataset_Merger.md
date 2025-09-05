# ARCIS Model Dataset Merger

This tool merges multiple weapon detection, military vehicle, aircraft, and crime datasets into a unified dataset with standardized class mappings for the ARCIS model.

## Features

- **Comprehensive Class Mapping**: Automatically maps various weapon types, military vehicles, aircraft, and crime-related classes to standardized categories
- **Multiple Split Options**: Creates datasets with different train/validation/test ratios (80/10/10, 70/15/15, 75/12.5/12.5)
- **Intelligent Defaults**: Uses fallback mapping rules when specific weapon/vehicle types aren't detected
- **YOLO Format Compatible**: Maintains YOLO annotation format throughout the process
- **Metadata Tracking**: Generates detailed metadata about class distributions and source datasets

## Supported Datasets

The merger automatically processes the following datasets from your `datasets/` directory:

### Weapon Datasets
- `weapon_detection_balanced&unbalanced/` - General weapon detection
- `gun_holding_person_21_12.v2i.yolov8/` - Gun holding, criminal behavior (punch, slap, balaclava)
- `knife-detection.v2i.yolov8/` - Knife detection
- `70k Guns.v5-main.yolov8/` - Large gun dataset
- `various_weapons_by_type.yolov8/` - Specific weapon types (Automatic Rifle, Bazooka, Grenade Launcher)
- `crime.v10i.yolov8/` - Criminal behavior detection

### Military Vehicle Datasets
- `tanks.v1i.yolov8/` - Tank detection
- `ilitary vehicles of numerous classes.yolov8/` - Multiple military vehicle types

### Other Datasets
- `Fire_Detection.v8i.yolov8/` - Fire and smoke detection

## Class Mapping Strategy

### Weapons
- **Specific Types**: `gun`, `handgun`, `knife`, `automatic_rifle`, `bazooka`, `grenade_launcher`
- **Default Fallback**: `weapon` (for unrecognized weapon types)

### Military Vehicles
- **Specific Types**: `military_tank`, `military_truck`, `military_aircraft`, `military_helicopter`
- **Default Fallback**: `military_vehicle` (for unrecognized military vehicle types)

### Aircraft
- **Specific Types**: `military_aircraft`, `military_helicopter`, `civilian_aircraft`
- **Default Fallback**: `aircraft` (for unrecognized aircraft types)

### High Warning Classes
- **Criminal Behavior**: `punch`, `slap`, `balaclava`, `criminal` → `high_warning`

### Other Classes
- `person`, `civilian_car`, `fire`, `smoke`

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your datasets are organized in the `datasets/` directory with the expected structure.

## Usage

### Basic Usage
```bash
python dataset_merger.py
```

This will create three merged datasets:
- `merged_dataset_80_10_10/` (80% train, 10% val, 10% test)
- `merged_dataset_70_15_15/` (70% train, 15% val, 15% test)  
- `merged_dataset_75_12_12/` (75% train, 12.5% val, 12.5% test)

### Custom Usage
```bash
python dataset_merger.py --datasets_dir /path/to/datasets --output_dir /path/to/output --split_ratios 0.8 0.1 0.1
```

### Parameters
- `--datasets_dir`: Directory containing source datasets (default: `datasets`)
- `--output_dir`: Output directory for merged dataset (default: `merged_dataset`)
- `--split_ratios`: Three float values for train/val/test ratios (must sum to 1.0)

## Output Structure

Each merged dataset contains:

```
merged_dataset_80_10_10/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml          # YOLO dataset configuration
└── metadata.json      # Detailed dataset information
```

### data.yaml
Standard YOLO format configuration file with:
- Path to train/val/test directories
- Number of classes (`nc`)
- Class names list

### metadata.json
Comprehensive dataset information including:
- Total classes and class mapping
- Class distribution counts
- Actual split ratios achieved
- Source datasets used

## Class Distribution Analysis

The tool provides detailed statistics during processing:

```
Total samples collected: 45,230
Class distribution:
  weapon: 15,420
  gun: 8,930
  high_warning: 3,240
  military_tank: 2,100
  person: 8,540
  ...
```

## Recommendations

### Split Ratio Selection

1. **80/10/10 Split**: Best for balanced training with sufficient validation data
2. **70/15/15 Split**: Better for models requiring more validation/test data
3. **75/12.5/12.5 Split**: Recommended for very large datasets (>50k samples)

### Additional Suggestions

1. **Data Augmentation**: Consider applying augmentation techniques to minority classes
2. **Class Balancing**: Monitor class distribution and apply balancing techniques if needed
3. **Quality Control**: Review merged samples to ensure proper class mapping
4. **Incremental Training**: Start with high-priority datasets and gradually add more

## Troubleshooting

### Common Issues

1. **Missing Datasets**: The tool will skip missing datasets and continue processing
2. **Invalid Annotations**: Malformed YOLO labels are automatically skipped
3. **Memory Issues**: For very large datasets, consider processing in batches

### Validation

After merging, validate your dataset by:
1. Checking the `metadata.json` for expected class counts
2. Reviewing sample images and labels
3. Testing with a small training run

## Advanced Configuration

### Custom Class Mapping

To modify class mappings, edit the `class_mapping` dictionary in `DatasetMerger.__init__()`:

```python
self.class_mapping = {
    'your_custom_class': 'standardized_name',
    # ... other mappings
}
```

### Dataset Priority

Datasets are processed with different priorities (high/medium/low) which can be adjusted in the `dataset_configs` dictionary.

## Performance Notes

- Processing time depends on dataset size and number of annotations
- Expect ~1-5 minutes for datasets with 10k-50k images
- Large datasets (>100k images) may take 10-30 minutes
- Progress is displayed during processing

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Verify dataset directory structure matches expected format
3. Ensure all required dependencies are installed
4. Review the metadata.json file for processing results 