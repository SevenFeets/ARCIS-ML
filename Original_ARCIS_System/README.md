# Original ARCIS System

This folder contains the standalone ARCIS (Advanced Reconnaissance and Combat Intelligence System) weapon detection system files.

## 📁 Contents

### Core Detection Scripts
- `train_weapon_detection.py` - **Main ARCIS tactical detection system**
  - Military threat classification (CRITICAL, HIGH, MEDIUM, LOW)
  - IMX415 distance estimation
  - Tactical engagement recommendations
  - Mission logging and SITREP generation
  - Jetson Nano optimized training and inference
  - Audio alerts for critical threats

- `train_weapon_detection_gps.py` - **GPS-Enhanced ARCIS system**
  - All features from main system PLUS:
  - L76K GPS module integration
  - MGRS military coordinate system
  - GPS-tagged mission logging
  - Real-time position tracking
  - Enhanced SITREP with location data

- `webcam_inference.py` - **Simple webcam inference script**
  - Basic weapon detection using webcam
  - Real-time inference display
  - Confidence-based bounding box colors

- `verify_setup.py` - **System verification script**
  - Checks all file paths and dependencies
  - Verifies model and dataset availability
  - Tests audio and YOLO functionality
  - Provides setup troubleshooting guidance

- `test_dataset_selection.py` - **Dataset selection demo**
  - Shows available dataset splits and statistics
  - Displays image counts and ratios for each split
  - Helps choose the best dataset for your needs

### Requirements
- `requirements_gps.txt` - Dependencies for GPS-enhanced system

## 🚀 Quick Start

### Verify Setup (Recommended First Step)
```bash
cd Original_ARCIS_System
python verify_setup.py
```

### Standard ARCIS System
```bash
cd Original_ARCIS_System
python train_weapon_detection.py
```

### GPS-Enhanced System
```bash
cd Original_ARCIS_System
pip install -r requirements_gps.txt
python train_weapon_detection_gps.py
```

## 🎯 Key Features

### Military-Grade Threat Classification
- **CRITICAL (RED)**: Immediate danger - tanks, aircraft, heavy weapons
- **HIGH (ORANGE)**: Significant threat - guns, rifles, weapons
- **MEDIUM (YELLOW)**: Potential threat - handguns, knives
- **LOW (GREEN)**: Minimal threat - other objects

### Tactical Intelligence
- Real-time distance estimation using IMX415 sensor
- Bearing calculation to detected threats
- Engagement recommendations (ENGAGE, AVOID, TAKE_COVER)
- Mission timer and threat counting
- After-action report generation

### Field Operation Support
- Audio alerts for critical threats
- Tactical screenshot capture
- Mission logging with JSON format
- SITREP generation for command
- Operator and mission ID tracking

### Jetson Nano Optimization
- Optimized for 2.9" 2K 1440x1440 display
- Support for 1080p laptop screens
- Memory-efficient training parameters
- ONNX and TensorRT export formats
- Performance monitoring

### Dataset Selection
- **3 Dataset Splits Available**: 80/10/10, 70/15/15, 75/12.5/12.5
- **Interactive Selection**: Choose dataset split at runtime
- **Automatic Verification**: Checks dataset availability and completeness
- **Usage Recommendations**: Guided selection based on training goals
- **Statistics Display**: Shows image counts and ratios for each split

## 🔧 Configuration Options

### Display Configurations
- **2K Square**: 2.9 inch 2K 1440x1440 IPS 120hz display
- **1080p**: Standard 1080p laptop screen

### Training Options
- Jetson Nano optimized (416px inference, smaller batches)
- Desktop/Server optimized (640px inference, larger batches)
- Custom epoch and model size selection

### Mission Parameters
- Custom mission ID and operator ID
- Configurable confidence thresholds
- Audio alert enable/disable
- Distance estimation toggle

## 📊 System Requirements

### Hardware
- **Jetson Nano** (4GB recommended) OR desktop/laptop
- **IMX415 Camera** with 2.8mm lens (for distance estimation)
- **Audio output** (for danger alerts)
- **Display** (2K square or 1080p)

### Software
- Python 3.8+
- CUDA support (for GPU acceleration)
- OpenCV 4.5+
- Ultralytics YOLO 8.0+

## 🆚 System Comparison

| Feature | Standard ARCIS | GPS-Enhanced | Redis-Integrated |
|---------|---------------|--------------|------------------|
| Threat Detection | ✅ | ✅ | ✅ |
| Military Classification | ✅ | ✅ | ✅ |
| Distance Estimation | ✅ | ✅ | ✅ |
| Mission Logging | ✅ | ✅ | ✅ |
| GPS Tracking | ❌ | ✅ | ✅ |
| MGRS Coordinates | ❌ | ✅ | ✅ |
| Cloud Processing | ❌ | ❌ | ✅ |
| Multi-device Communication | ❌ | ❌ | ✅ |
| Raspberry Pi Alerts | ❌ | ❌ | ✅ |
| Website Integration | ❌ | ❌ | ✅ |

## 📚 Related Documentation

- See `../Documentation/` for setup guides and manuals
- See `../ARCIS_Redis_System/` for distributed system version
- See `../Dataset_Tools/` for dataset preparation tools

### Documentation
- `DATASET_SELECTION_GUIDE.md` - **Comprehensive dataset selection guide**
  - Detailed comparison of 3 available dataset splits
  - Recommendations for different use cases
  - Best practices and workflow suggestions

---

**Note**: This is the original, standalone ARCIS system. For distributed operations with cloud processing and multi-device communication, see the `ARCIS_Redis_System` folder. 