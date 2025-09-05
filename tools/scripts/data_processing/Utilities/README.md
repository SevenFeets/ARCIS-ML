# Utilities

This folder contains utility scripts and tools for testing, training, and maintaining the ARCIS weapon detection system.

## 📁 Contents

### Training Utilities
- `train_yolo.py` - **Basic YOLO training script**
  - Simple YOLO model training
  - Standard configuration options
  - Basic performance monitoring

- `default_train.py` - **Default training configuration**
  - Pre-configured training parameters
  - Standard dataset handling
  - Basic model evaluation

### Testing Tools
- `test_model.py` - **Model testing and evaluation**
  - Model performance testing
  - Accuracy metrics calculation
  - Inference speed benchmarking
  - Validation dataset evaluation

- `test_gpu.py` - **GPU functionality test**
  - CUDA availability check
  - GPU memory testing
  - Performance benchmarking
  - Hardware compatibility verification

### File Management
- `delete_duplicate_files.py` - **Duplicate file removal**
  - Identifies duplicate images
  - Safe file deletion
  - Storage optimization
  - Dataset cleanup

## 🚀 Quick Start

### Test GPU Setup
```bash
cd Utilities
python test_gpu.py
```

### Basic Model Training
```bash
python train_yolo.py --data path/to/data.yaml --epochs 100
```

### Test Trained Model
```bash
python test_model.py --model path/to/model.pt --data path/to/test/data
```

### Clean Duplicate Files
```bash
python delete_duplicate_files.py --directory path/to/dataset
```

## 🔧 Tool Descriptions

### train_yolo.py
Basic YOLO training script with standard configurations:
- Supports YOLOv8 models
- Configurable epochs and batch sizes
- Basic data augmentation
- Standard evaluation metrics

### default_train.py
Pre-configured training script with optimized settings:
- Default hyperparameters for weapon detection
- Automatic dataset validation
- Progress monitoring
- Model checkpointing

### test_model.py
Comprehensive model testing utility:
- **Performance Metrics**: Precision, Recall, mAP
- **Speed Testing**: Inference time measurement
- **Memory Usage**: GPU/CPU memory monitoring
- **Batch Testing**: Multiple image processing

### test_gpu.py
Hardware compatibility checker:
- **CUDA Detection**: Checks CUDA availability
- **GPU Information**: Device specifications
- **Memory Testing**: Available GPU memory
- **Performance Test**: Basic computation benchmark

### delete_duplicate_files.py
File management utility:
- **Hash-based Detection**: MD5/SHA256 comparison
- **Safe Deletion**: Backup before removal
- **Progress Reporting**: Real-time status updates
- **Selective Cleanup**: Filter by file type/size

## 📊 Usage Examples

### GPU Testing
```bash
python test_gpu.py
# Output:
# CUDA Available: True
# GPU Device: NVIDIA GeForce RTX 3080
# GPU Memory: 10240 MB
# Performance Score: 95.2
```

### Model Evaluation
```bash
python test_model.py --model ../Models/yolov8n.pt --data ../ARCIS_Dataset_80_10_10/data.yaml
# Output:
# mAP@0.5: 0.892
# mAP@0.5:0.95: 0.654
# Inference Speed: 12.3ms
```

### Duplicate Cleanup
```bash
python delete_duplicate_files.py --directory ../datasets --dry-run
# Output:
# Found 1,247 duplicate files
# Total space to save: 2.3 GB
# Use --confirm to proceed with deletion
```

## ⚙️ Configuration Options

### Training Parameters
```python
# In train_yolo.py
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
LEARNING_RATE = 0.01
```

### Testing Thresholds
```python
# In test_model.py
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1000
```

### Cleanup Settings
```python
# In delete_duplicate_files.py
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
MIN_FILE_SIZE = 1024  # bytes
BACKUP_ENABLED = True
```

## 🛠️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: SSD recommended for faster I/O

### Software
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **PyTorch**: 1.13+
- **OpenCV**: 4.5+

## 📈 Performance Optimization

### Training Optimization
- Use GPU acceleration when available
- Optimize batch size based on GPU memory
- Enable mixed precision training
- Use data loading workers

### Testing Optimization
- Batch processing for multiple images
- GPU memory management
- Efficient data loading
- Parallel processing where possible

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Not Available**
   ```bash
   # Check CUDA installation
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of Memory Errors**
   ```bash
   # Reduce batch size in training scripts
   # Monitor GPU memory usage
   ```

3. **Slow Training**
   ```bash
   # Check data loading bottlenecks
   # Optimize dataset storage (SSD vs HDD)
   # Increase number of workers
   ```

### Debug Commands
```bash
# Check system resources
python test_gpu.py --verbose

# Validate model architecture
python test_model.py --model path/to/model.pt --validate-only

# Monitor training progress
python train_yolo.py --data data.yaml --epochs 1 --verbose
```

## 📚 Related Tools

- **Original ARCIS**: `../Original_ARCIS_System/` - Main detection system
- **Dataset Tools**: `../Dataset_Tools/` - Dataset preparation
- **Models**: `../Models/` - Pre-trained model files
- **Documentation**: `../Documentation/` - Setup guides

---

**Note**: These utilities are designed to support the ARCIS weapon detection system development and maintenance. Use them for testing, optimization, and troubleshooting. 