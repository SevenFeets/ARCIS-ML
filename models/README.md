# Models Directory

This directory contains all model artifacts for the ARCIS project.

## Structure

```
models/
├── pretrained/             # Pre-trained models
│   ├── yolo11n.pt         # YOLOv11 nano
│   ├── yolov8n.pt         # YOLOv8 nano
│   └── yolov8n-ultralight.yaml # Ultralight configuration
├── trained/                # Trained models
│   ├── yolo/               # YOLO trained models
│   ├── mobilenet/          # MobileNet models
│   │   └── mobilenet_ssd_model/
│   └── ssd/                # SSD models
│       └── weapon_detection_SSD/
└── exported/               # Exported models
    ├── onnx/               # ONNX format
    ├── tflite/             # TensorFlow Lite format
    └── tensorrt/            # TensorRT format
```

## Model Performance

### YOLO Models (Trained)
- **YOLOv8x Ultra High**: mAP50: 0.942 (Enterprise)
- **YOLOv8l Cloud**: mAP50: 0.932 (Cloud/Server)
- **YOLOv8m Mobile**: mAP50: 0.925 (Mobile)
- **YOLOv8n Quantized**: mAP50: 0.878 (Edge)

### MobileNet Models
- **MobileNet-SSD**: Optimized for Raspberry Pi
- **Quantized TFLite**: Edge deployment ready

### SSD Models
- **Weapon Detection SSD**: Alternative architecture
- **Custom trained**: Optimized for specific use cases

## Usage

### Loading Pre-trained Models
```python
from ultralytics import YOLO

# Load YOLOv8 nano
model = YOLO('models/pretrained/yolov8n.pt')

# Load trained model
model = YOLO('models/trained/yolo/best.pt')
```

### Model Conversion
```python
# Convert to ONNX
model.export(format='onnx')

# Convert to TensorFlow Lite
model.export(format='tflite')
```

## Model Training

Training scripts are located in:
- `src/scripts/train.py` - Main training script
- `tools/scripts/model_conversion/` - Model conversion utilities

## Deployment

Deployment configurations are in:
- `deployment/edge/` - Edge device deployment
- `deployment/cloud/` - Cloud deployment
- `deployment/docker/` - Containerized deployment

## Notes

- All models are trained on weapon detection datasets
- Models support multiple deployment formats
- Performance metrics are documented in `docs/reports/training_progress_report.txt`
