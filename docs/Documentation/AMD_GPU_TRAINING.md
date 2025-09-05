# Training with AMD GPU using ROCm

This guide explains how to set up and train machine learning models using an AMD GPU with ROCm.

## Hardware Used
- AMD Radeon RX 6700 XT

## Software Setup

### 1. Create a Python Virtual Environment
```bash
python3 -m venv arcis_venv
source arcis_venv/bin/activate
```

### 2. Install PyTorch with ROCm Support
```bash
pip install torch==2.2.2+rocm5.6 torchvision==0.17.2+rocm5.6 torchaudio==2.2.2+rocm5.6 --index-url https://download.pytorch.org/whl/rocm5.6
```

### 3. Install Ultralytics for YOLO Training
```bash
pip install ultralytics
```

### 4. Install Compatible NumPy Version
```bash
pip install numpy==1.26.4
```

## Key Configuration for AMD GPUs

For AMD GPUs like the RX 6700 XT, you may need to set the `HSA_OVERRIDE_GFX_VERSION` environment variable for compatibility:

```python
import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
```

This should be set at the beginning of your training script.

## Training Script

The training script (`train_with_rocm_fixed.py`) includes:

1. Setting the HSA_OVERRIDE_GFX_VERSION environment variable
2. Loading the YOLO model
3. Configuring training parameters
4. Running training on the GPU

## Training Parameters

For optimal performance with AMD GPUs, consider these settings:

- **Image Size**: 416 (reduced from 640 for better compatibility)
- **Batch Size**: 8 (adjusted for GPU memory)
- **Workers**: 2-4 (depending on system capabilities)
- **AMP**: Disabled (automatic mixed precision can cause issues with some ROCm versions)

## Monitoring GPU Usage

You can monitor GPU usage during training with:

```bash
rocm-smi
```

This will show GPU utilization, temperature, and memory usage.

## Troubleshooting

If you encounter segmentation faults or other GPU errors:

1. Try different ROCm versions (5.4, 5.5, 5.6)
2. Adjust the HSA_OVERRIDE_GFX_VERSION value (10.3.0 works for many cards)
3. Reduce batch size or image size
4. Disable automatic mixed precision (amp=False)

## References

- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/rocm.html)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [ROCm Documentation](https://rocm.docs.amd.com/) 