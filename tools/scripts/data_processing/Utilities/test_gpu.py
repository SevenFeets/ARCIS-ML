import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("No GPU detected. Please check your CUDA installation.") 