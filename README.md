# ARCIS - Advanced Real-time Computer Intelligence System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## Overview

ARCIS is a comprehensive weapon detection system designed for real-time deployment across multiple platforms including edge devices (Raspberry Pi, Jetson Nano), mobile devices, cloud infrastructure, and enterprise environments. The system utilizes state-of-the-art YOLO models with custom optimizations for different deployment scenarios.

## Key Features

- **Multi-Architecture Support**: YOLOv8 (nano, small, medium, large, extra-large)
- **Multi-Platform Deployment**: Edge, mobile, cloud, and enterprise configurations
- **Advanced Optimization**: Quantization, pruning, and platform-specific optimizations
- **Comprehensive Dataset**: Merged weapon detection datasets with 4-19 classes
- **Real-time Inference**: Optimized for low-latency detection
- **Scalable Architecture**: Redis-based distributed system support

## Performance Highlights

| Model | mAP50 | Platform | Use Case |
|-------|-------|----------|----------|
| YOLOv8x Ultra High | 0.942 | Enterprise | Maximum accuracy |
| YOLOv8l Cloud | 0.932 | Cloud/Server | High-performance cloud |
| YOLOv8m Mobile | 0.925 | Mobile | Balanced mobile performance |
| YOLOv8n Quantized | 0.878 | Edge | Optimized edge deployment |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arcis.git
cd arcis

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment (if using)
source arcis_venv/bin/activate
```

### Basic Usage

```python
from ultralytics import YOLO

# Load a trained model
model = YOLO('models/trained/yolo/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Training

```bash
# Train on merged dataset
python src/scripts/train.py --dataset merged_dataset_80_10_10_FULL --model yolov8n --epochs 100

# Train for specific platform
python src/scripts/train.py --platform raspberry_pi --model yolov8n --quantize
```

## Project Structure

```
arcis/
├── src/                    # Source code
│   ├── arcis/             # Main package
│   └── scripts/           # Executable scripts
├── data/                  # Datasets
│   ├── raw/               # Original datasets
│   └── processed/         # Processed datasets
├── models/                # Model artifacts
│   ├── pretrained/        # Pre-trained models
│   ├── trained/           # Trained models
│   └── exported/          # Exported models
├── experiments/           # Training experiments
├── deployment/            # Deployment configurations
├── docs/                  # Documentation
├── tests/                 # Test suite
└── tools/                 # Development tools
```

## Datasets

The project includes three merged weapon detection datasets:

- **merged_dataset_80_10_10_FULL**: 248,374 images, 19 classes (primary dataset)
- **merged_dataset**: 122,985 images, 4 classes (standardized)
- **merged_dataset_75_15**: 125,497 images, 4 classes (alternative split)

## Deployment Platforms

### Edge Devices
- **Raspberry Pi**: Optimized YOLOv8n with quantization
- **Jetson Nano**: CUDA-optimized inference
- **Mobile**: TensorFlow Lite models

### Cloud Infrastructure
- **AWS**: Containerized deployment with auto-scaling
- **GCP**: Cloud Run and Vertex AI integration
- **Azure**: Container Instances and ML Services

### Enterprise
- **On-premise**: Docker containers with Redis clustering
- **Hybrid**: Multi-cloud deployment with load balancing

## Training Results

The system has been trained with 15 different configurations achieving mAP50 scores from 0.875 to 0.942. Detailed training progress and results are documented in [docs/reports/training_progress_report.txt](docs/reports/training_progress_report.txt).

## Documentation

- [Installation Guide](docs/guides/installation.md)
- [Training Guide](docs/guides/training.md)
- [Deployment Guide](docs/guides/deployment.md)
- [API Reference](docs/api/)
- [Tutorials](docs/tutorials/)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is provided for educational and research purposes only. Please ensure compliance with all applicable laws and regulations when using weapon detection technology.

## Support

For questions and support:
- Create an [issue](https://github.com/yourusername/arcis/issues)
- Check the [documentation](docs/)
- Review [troubleshooting guide](docs/guides/troubleshooting.md)

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [TensorFlow](https://tensorflow.org) for mobile optimization
- [OpenCV](https://opencv.org) for computer vision utilities
- Contributors and the open-source community
