# ARCIS Project Reorganization Plan

## Current Issues with Folder Structure

### 1. **Root Directory Clutter**
- Multiple loose files in root (`yolo11n.pt`, `yolov8n.pt`, `calibration_image_sample_data_20x128x128x3_float32.npy`)
- Mixed purposes and technologies in root
- No clear main README.md

### 2. **Inconsistent Naming Conventions**
- `rocm optimized train/` (spaces, inconsistent casing)
- `newDatasets/` (camelCase vs snake_case)
- `Original_ARCIS_System/` (mixed naming)

### 3. **Scattered Related Files**
- Training scripts spread across multiple directories
- Dataset files in root instead of dedicated folder
- Model files mixed with scripts

### 4. **Missing Professional Structure**
- No clear separation of concerns
- No standardized project layout
- Missing essential project files

## Recommended Professional Structure

```
arcis/
├── README.md                          # Main project documentation
├── LICENSE                           # Project license
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Main dependencies
├── setup.py                          # Package installation
├── pyproject.toml                    # Modern Python project config
├── docker-compose.yml                # Development environment
├── Dockerfile                        # Main container
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── arcis/                        # Main package
│   │   ├── __init__.py
│   │   ├── core/                     # Core functionality
│   │   │   ├── __init__.py
│   │   │   ├── detection.py
│   │   │   ├── training.py
│   │   │   └── inference.py
│   │   ├── models/                   # Model definitions
│   │   │   ├── __init__.py
│   │   │   ├── yolo.py
│   │   │   ├── mobilenet.py
│   │   │   └── ssd.py
│   │   ├── utils/                    # Utilities
│   │   │   ├── __init__.py
│   │   │   ├── data_processing.py
│   │   │   ├── visualization.py
│   │   │   └── metrics.py
│   │   └── deployment/               # Deployment modules
│   │       ├── __init__.py
│   │       ├── edge.py
│   │       ├── cloud.py
│   │       └── mobile.py
│   └── scripts/                     # Executable scripts
│       ├── train.py
│       ├── inference.py
│       ├── evaluate.py
│       └── deploy.py
│
├── data/                             # Data management
│   ├── raw/                          # Original datasets
│   │   ├── dataset_1/
│   │   ├── dataset_2/
│   │   └── ...
│   ├── processed/                    # Processed datasets
│   │   ├── merged_dataset/
│   │   ├── merged_dataset_75_15/
│   │   └── merged_dataset_80_10_10_FULL/
│   ├── external/                     # External data sources
│   └── README.md                     # Data documentation
│
├── models/                           # Model artifacts
│   ├── pretrained/                   # Pre-trained models
│   │   ├── yolo11n.pt
│   │   ├── yolov8n.pt
│   │   └── yolov8n-ultralight.yaml
│   ├── trained/                      # Trained models
│   │   ├── yolo/
│   │   ├── mobilenet/
│   │   └── ssd/
│   ├── exported/                     # Exported models
│   │   ├── onnx/
│   │   ├── tflite/
│   │   └── tensorrt/
│   └── README.md                     # Model documentation
│
├── experiments/                      # Training experiments
│   ├── runs/                         # Training runs
│   │   ├── detect/
│   │   ├── calibration_data/
│   │   └── ...
│   ├── configs/                      # Training configurations
│   │   ├── yolo_configs/
│   │   ├── mobilenet_configs/
│   │   └── ssd_configs/
│   ├── logs/                         # Training logs
│   └── README.md                     # Experiment documentation
│
├── deployment/                       # Deployment configurations
│   ├── docker/                       # Docker configurations
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.cloud
│   │   ├── Dockerfile.jetson
│   │   └── docker-compose.yml
│   ├── edge/                         # Edge deployment
│   │   ├── raspberry_pi/
│   │   ├── jetson_nano/
│   │   └── mobile/
│   ├── cloud/                        # Cloud deployment
│   │   ├── aws/
│   │   ├── gcp/
│   │   └── azure/
│   └── README.md                     # Deployment documentation
│
├── tests/                            # Test suite
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   ├── performance/                  # Performance tests
│   ├── conftest.py                   # Pytest configuration
│   └── README.md                     # Testing documentation
│
├── docs/                             # Documentation
│   ├── api/                          # API documentation
│   ├── tutorials/                    # Tutorials
│   ├── guides/                       # User guides
│   │   ├── dataset_merger.md
│   │   ├── edge_training.md
│   │   ├── jetson_optimization.md
│   │   └── gps_setup.md
│   ├── reports/                      # Project reports
│   │   └── training_progress_report.txt
│   └── README.md                     # Documentation index
│
├── tools/                            # Development tools
│   ├── scripts/                      # Utility scripts
│   │   ├── data_processing/
│   │   ├── model_conversion/
│   │   └── deployment/
│   ├── notebooks/                    # Jupyter notebooks
│   └── README.md                     # Tools documentation
│
├── assets/                           # Static assets
│   ├── audio/                        # Audio files
│   │   └── danger_alert.mp3
│   ├── images/                       # Images
│   │   └── test_image.jpg
│   └── README.md                     # Assets documentation
│
└── .github/                          # GitHub workflows
    ├── workflows/
    │   ├── ci.yml                    # Continuous integration
    │   ├── train.yml                 # Training pipeline
    │   └── deploy.yml                 # Deployment pipeline
    └── ISSUE_TEMPLATE/
```

## Migration Steps

### Phase 1: Create New Structure
1. Create new directory structure
2. Move files to appropriate locations
3. Update import paths
4. Create missing configuration files

### Phase 2: Consolidate and Clean
1. Merge duplicate functionality
2. Remove deprecated code
3. Standardize naming conventions
4. Add proper documentation

### Phase 3: Professional Setup
1. Add CI/CD pipelines
2. Create proper package structure
3. Add comprehensive testing
4. Update documentation

## Benefits of Reorganization

### 1. **Professional Appearance**
- Clear separation of concerns
- Standard Python project structure
- Industry-standard conventions

### 2. **Better Maintainability**
- Logical file organization
- Easier to find and modify code
- Clear dependencies

### 3. **Improved Collaboration**
- Standard structure familiar to developers
- Clear documentation
- Proper version control

### 4. **Deployment Ready**
- Containerized development environment
- Clear deployment configurations
- Automated testing and deployment

### 5. **Scalability**
- Modular architecture
- Easy to add new features
- Clear extension points

## Immediate Actions Needed

1. **Create main README.md** - Missing project overview
2. **Add .gitignore** - Essential for version control
3. **Consolidate training scripts** - Currently scattered
4. **Organize datasets** - Move to dedicated data/ folder
5. **Standardize naming** - Fix inconsistent naming conventions
6. **Add configuration files** - requirements.txt, setup.py, etc.

## Priority Levels

### High Priority (Immediate)
- Create main README.md
- Add .gitignore
- Consolidate training scripts
- Organize datasets

### Medium Priority (Next Sprint)
- Create proper package structure
- Add configuration files
- Standardize naming conventions
- Add basic testing

### Low Priority (Future)
- Full CI/CD setup
- Comprehensive documentation
- Advanced deployment configurations
- Performance optimization tools

## Conclusion

Your current structure shows good functionality but lacks professional organization. The proposed structure follows industry standards and will make your project more maintainable, collaborative, and deployment-ready. The migration can be done incrementally without disrupting current functionality.
