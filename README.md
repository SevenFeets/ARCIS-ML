# ARCIS - Advanced Reconnaissance and Combat Intelligence System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)
[![Jetson Nano](https://img.shields.io/badge/Jetson-Nano-green.svg)](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)

##  Overview

ARCIS (Advanced Reconnaissance and Combat Intelligence System) is a comprehensive, military-grade weapon detection and threat assessment system designed for tactical field operations. The system combines real-time computer vision, GPS tracking, cloud intelligence, and distributed communication to provide enhanced situational awareness and threat classification.

##  Key Features

###  Military-Grade Threat Classification
- **CRITICAL (RED)**: Immediate danger - tanks, aircraft, heavy weapons
- **HIGH (ORANGE)**: Significant threat - guns, rifles, weapons  
- **MEDIUM (YELLOW)**: Potential threat - handguns, knives
- **LOW (GREEN)**: Minimal threat - other objects

###  Multi-Platform Deployment
- **Jetson Nano**: High-performance edge computing with TensorRT optimization
- **Raspberry Pi**: Lightweight field deployment with TensorFlow Lite
- **Desktop/Server**: Full-featured training and inference capabilities
- **Cloud Integration**: Google Cloud Vision for enhanced threat analysis

###  Distributed Architecture
- **Redis Message Broker**: Real-time threat distribution and caching
- **Multi-Device Communication**: Raspberry Pi field alerts and coordination
- **Cloud Processing**: Enhanced threat analysis with Google Cloud Vision
- **Website Integration**: Automatic data upload and dashboard connectivity

###  Tactical Intelligence
- **Distance Estimation**: IMX415 sensor-based threat range calculation
- **GPS Integration**: L76K GPS module with MGRS military coordinates
- **Mission Logging**: Comprehensive SITREP generation and after-action reports
- **Audio Alerts**: Critical threat notifications with customizable sounds
- **Real-time Tracking**: Live threat monitoring and engagement recommendations

##  Project Structure

```
ARCIS-ML/
├── Original_ARCIS_System/          # Standalone ARCIS system
│   ├── train_weapon_detection.py   # Main tactical detection system
│   ├── train_weapon_detection_gps.py # GPS-enhanced version
│   ├── webcam_inference.py         # Simple webcam inference
│   └── verify_setup.py             # System verification
├── ARCIS_Redis_System/             # Distributed Redis-integrated system
│   ├── train_weapon_detection_redis.py # Main detection with Redis
│   ├── arcis_redis_integration.py  # Redis manager
│   ├── arcis_cloud_service.py      # Google Cloud Vision service
│   ├── arcis_api_service.py        # FastAPI service
│   ├── raspberry_pi_client.py      # Field alert client
│   └── docker-compose.yml          # Multi-container deployment
├── multi_edge_deployment/          # Edge device optimization
│   ├── jetson_nano/               # Jetson Nano specific tools
│   ├── raspberry_pi/              # Raspberry Pi specific tools
│   └── check_onnx.py              # ONNX model validation
├── mobilenet_ssd_model/           # MobileNet-SSD models for Pi
├── merged_dataset/                # Unified weapon detection dataset
├── Documentation/                 # Comprehensive guides
├── Audio_Assets/                  # Threat alert sounds
└── Utilities/                     # Helper tools and scripts
```

##  Quick Start

### 1. System Verification (Recommended First Step)
```bash
cd Original_ARCIS_System
python verify_setup.py
```

### 2. Standalone ARCIS System
```bash
cd Original_ARCIS_System
python train_weapon_detection.py
```

### 3. GPS-Enhanced System
```bash
cd Original_ARCIS_System
pip install -r requirements_gps.txt
python train_weapon_detection_gps.py
```

### 4. Distributed Redis System
```bash
cd ARCIS_Redis_System
cp env.example .env
# Edit .env with your configuration
docker-compose up -d
```

##  System Variants

### Original ARCIS System
**Location**: `Original_ARCIS_System/`
-  Standalone operation
-  Local processing only
-  Direct camera interface
-  Mission logging and SITREP generation
-  Tactical interface with threat classification
-  Audio alerts for critical threats
-  Distance estimation with IMX415 sensor

### GPS-Enhanced ARCIS System
**Location**: `Original_ARCIS_System/train_weapon_detection_gps.py`
-  All original features PLUS:
-  L76K GPS module integration
-  MGRS military coordinate system
-  GPS-tagged mission logging
-  Real-time position tracking
-  Enhanced SITREP with location data

### Redis-Integrated ARCIS System
**Location**: `ARCIS_Redis_System/`
-  All original features PLUS:
-  Distributed processing architecture
-  Redis message broker and caching
-  Google Cloud Vision integration
-  Real-time multi-device communication
-  Automatic website data upload
-  Raspberry Pi field alerts
-  Docker containerization
-  Scalable microservices architecture

## System Architecture

### Standalone Architecture
```
┌─────────────────┐
│   Camera        │
│   (IMX415)      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   ARCIS Core    │
│   (YOLOv8)      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Threat        │
│   Classification│
└─────────────────┘
```

### Distributed Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Jetson Nano   │    │      Redis      │    │  Cloud Service  │
│  (Detection)    │◄──►│   (Message      │◄──►│ (Google Vision) │
│                 │    │    Broker)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Service   │    │  Raspberry Pi   │    │    Website      │
│  (Monitoring)   │◄──►│   (Alerts)      │    │  (Dashboard)    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Threat Detection Classes

The system detects and classifies the following threat categories:

### Weapons
- `gun` - Handguns and pistols
- `handgun` - Specific handgun detection
- `knife` - Knives and bladed weapons
- `automatic_rifle` - Automatic rifles and assault weapons
- `bazooka` - Rocket launchers and heavy weapons
- `grenade_launcher` - Grenade launchers
- `weapon` - General weapon fallback

### Military Vehicles
- `military_tank` - Tanks and armored vehicles
- `military_truck` - Military transport vehicles
- `military_aircraft` - Military aircraft
- `military_helicopter` - Military helicopters
- `military_vehicle` - General military vehicle fallback

### Other Objects
- `person` - Human detection
- `fire` - Fire and smoke detection
- `other` - Miscellaneous objects

## Hardware Requirements

### Jetson Nano (Recommended)
- **GPU**: 128-core Maxwell GPU
- **CPU**: Quad-core ARM Cortex-A57
- **RAM**: 4GB LPDDR4 (recommended)
- **Storage**: 32GB+ microSD card
- **Camera**: IMX415 with 2.8mm lens
- **Display**: 2.9" 2K 1440x1440 IPS display (optional)

### Raspberry Pi 4
- **CPU**: Quad-core ARM Cortex-A72
- **RAM**: 4GB+ (recommended)
- **Storage**: 32GB+ microSD card
- **Camera**: Raspberry Pi Camera Module
- **Accelerator**: Coral USB Accelerator (optional)

### Desktop/Server
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **CPU**: Multi-core processor
- **RAM**: 8GB+ (16GB+ recommended for training)
- **Storage**: 100GB+ available space

## Performance Benchmarks

| Device | Model | Format | Resolution | Precision | FPS |
|--------|-------|--------|------------|-----------|-----|
| Jetson Nano | YOLOv8n | PyTorch | 640x640 | FP32 | ~3 FPS |
| Jetson Nano | YOLOv8n | TensorRT | 256x256 | FP16 | ~10-12 FPS |
| Jetson Nano | YOLOv8n-ultralight | TensorRT | 256x256 | FP16 | ~12-15 FPS |
| Raspberry Pi 4 | YOLOv8n | TFLite | 320x320 | INT8 | ~2-3 FPS |
| Raspberry Pi 4 | YOLOv8n | OpenVINO | 320x320 | FP16 | ~3-4 FPS |
| Desktop (RTX 3080) | YOLOv8m | PyTorch | 640x640 | FP32 | ~60+ FPS |

## Installation

### Prerequisites
```bash
# Python 3.8+
python --version

# CUDA (for GPU acceleration)
nvidia-smi

# Docker (for distributed system)
docker --version
docker-compose --version
```

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/SevenFeets/ARCIS-ML.git
cd ARCIS-ML

# Create virtual environment
python -m venv arcis_venv
source arcis_venv/bin/activate  # Linux/Mac
# or
arcis_venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Jetson Nano Setup
```bash
# Install JetPack SDK
# Follow NVIDIA's JetPack installation guide

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python

# Install TensorRT (for optimization)
sudo apt install tensorrt
```

### Raspberry Pi Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-opencv
pip3 install tensorflow-lite ultralytics

# For Coral USB Accelerator (optional)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install libedgetpu1-std
```

## Usage Examples

### Basic Weapon Detection
```bash
# Run with webcam
python Original_ARCIS_System/train_weapon_detection.py --source 0

# Run with video file
python Original_ARCIS_System/train_weapon_detection.py --source video.mp4

# Run with custom model
python Original_ARCIS_System/train_weapon_detection.py --model custom_model.pt
```

### GPS-Enhanced Detection
```bash
# Run with GPS integration
python Original_ARCIS_System/train_weapon_detection_gps.py --source 0 --enable_gps
```

### Distributed System
```bash
# Start all services
cd ARCIS_Redis_System
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Edge Device Deployment
```bash
# Train for Jetson Nano
python multi_edge_deployment/jetson_nano/train_for_jetson.py

# Train for Raspberry Pi
python multi_edge_deployment/raspberry_pi/train_for_raspi.py

# Export to ONNX
python multi_edge_deployment/export_to_onnx.py --model best.pt
```

## Documentation

### Core Documentation
- [`Original_ARCIS_System/README.md`](Original_ARCIS_System/README.md) - Standalone system guide
- [`ARCIS_Redis_System/README.md`](ARCIS_Redis_System/README.md) - Distributed system guide
- [`ARCIS_Redis_System/DEPLOYMENT_GUIDE.md`](ARCIS_Redis_System/DEPLOYMENT_GUIDE.md) - Complete deployment guide

### Specialized Guides
- [`Documentation/JETSON_OPTIMIZATION.md`](Documentation/JETSON_OPTIMIZATION.md) - Jetson Nano optimization
- [`Documentation/AMD_GPU_TRAINING.md`](Documentation/AMD_GPU_TRAINING.md) - AMD GPU training with ROCm
- [`Documentation/L76K_GPS_Setup_Guide.md`](Documentation/L76K_GPS_Setup_Guide.md) - GPS module setup
- [`Documentation/README_Dataset_Merger.md`](Documentation/README_Dataset_Merger.md) - Dataset preparation

### Edge Deployment
- [`multi_edge_deployment/README.md`](multi_edge_deployment/README.md) - Edge device optimization
- [`mobilenet_ssd_model/README.md`](mobilenet_ssd_model/README.md) - MobileNet-SSD for Raspberry Pi

## Configuration

### Environment Variables (Redis System)
```bash
# Google Cloud Vision
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Website Integration
WEBSITE_API_URL=https://your-website.com/api
WEBSITE_API_KEY=your-api-key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your-password

# Raspberry Pi Configuration
ARCIS_PI_ID=pi_field_01
ARCIS_API_URL=http://jetson-ip:8080
```

### Model Configuration
```yaml
# data.yaml example
path: ../merged_dataset
train: images/train
val: images/val
test: images/test

nc: 8  # number of classes
names: ['gun', 'handgun', 'knife', 'automatic_rifle', 'bazooka', 'grenade_launcher', 'military_tank', 'military_vehicle']
```

## Training

### Dataset Preparation
```bash
# Merge multiple datasets
python merge_datasets.py --input datasets/ --output merged_dataset/

# Verify dataset
python Utilities/verify_dataset_classes.py --dataset merged_dataset/
```

### Model Training
```bash
# Standard training
python Original_ARCIS_System/train_weapon_detection.py --epochs 100 --imgsz 640

# Jetson Nano optimized
python multi_edge_deployment/jetson_nano/train_for_jetson.py --epochs 50 --imgsz 416

# Raspberry Pi optimized
python multi_edge_deployment/raspberry_pi/train_for_raspi.py --epochs 100 --imgsz 320

# AMD GPU training
python rocm\ optimized\ train/train_with_rocm_fixed.py --epochs 50 --imgsz 320
```

## Security Features

- **Redis TTL**: Automatic data expiration and cleanup
- **API Authentication**: Bearer token support for secure communication
- **Encrypted Communication**: HTTPS/WSS support for production deployments
- **Access Logging**: Comprehensive audit trails for all operations
- **Credential Management**: Secure environment variable handling
- **Mission Logging**: Encrypted mission data with GPS coordinates

## System Comparison

| Feature | Original ARCIS | GPS-Enhanced | Redis-Integrated |
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
| Docker Deployment | ❌ | ❌ | ✅ |
| Redis Caching | ❌ | ❌ | ✅ |

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Detected**
   ```bash
   # Check camera permissions
   ls /dev/video*
   
   # Test camera with OpenCV
   python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   ```

2. **CUDA Out of Memory**
   ```bash
   # Reduce batch size and image size
   python train_weapon_detection.py --batch 4 --imgsz 416
   ```

3. **Redis Connection Failed**
   ```bash
   # Check Redis container status
   docker-compose ps redis
   
   # Test Redis connection
   docker-compose exec redis redis-cli ping
   ```

4. **Google Cloud Vision Errors**
   ```bash
   # Verify credentials
   export GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
   
   # Test API access
   gcloud auth application-default print-access-token
   ```

### Debug Commands
```bash
# System verification
python Original_ARCIS_System/verify_setup.py

# Check model validity
python multi_edge_deployment/check_onnx.py

# Monitor system resources
docker stats

# View container logs
docker-compose logs -f [service_name]
```

## Monitoring and Analytics

### Health Checks
```bash
# API Service health
curl http://localhost:8080/health

# System statistics
curl http://localhost:8080/api/threats/statistics

# Redis statistics
curl http://localhost:8080/api/system/status
```

### Real-time Monitoring
- **Redis Commander**: http://localhost:8081 (if enabled)
- **WebSocket**: ws://localhost:8080/ws/threats
- **API Endpoints**: http://localhost:8080/api/

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is provided for educational and research purposes only. Users are responsible for ensuring compliance with all applicable laws and regulations when using weapon detection technology. The authors and contributors are not responsible for any misuse of this software.

## Support

For support and questions:
1. Check the documentation in the `Documentation/` folder
2. Review the troubleshooting section above
3. Open an issue on GitHub
4. Check container logs for distributed system issues

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [Redis](https://redis.io/) for message brokering and caching
- [Google Cloud Vision](https://cloud.google.com/vision) for enhanced threat analysis
- [NVIDIA Jetson](https://developer.nvidia.com/embedded/jetson-nano-developer-kit) for edge computing platform

---

**ARCIS - Advanced Reconnaissance and Combat Intelligence System**  
*Enhancing tactical awareness through intelligent threat detection*
