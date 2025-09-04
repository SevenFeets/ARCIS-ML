#!/usr/bin/env python3
import torch
import time

def test_gpu_simple():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test with a very small tensor
        print("\nTesting GPU with small tensor operations...")
        
        # Create a small tensor on CPU
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        print(f"CPU tensor: {a}")
        
        try:
            # Move to GPU
            print("Moving tensor to GPU...")
            b = a.to("cuda")
            print(f"GPU tensor: {b}")
            
            # Simple operation
            print("Performing operation on GPU...")
            c = b * 2
            print(f"Result on GPU: {c}")
            
            # Move back to CPU
            print("Moving result back to CPU...")
            d = c.cpu()
            print(f"Result on CPU: {d}")
            
            print("\nGPU test successful!")
        except Exception as e:
            print(f"\nGPU test failed with error: {e}")
    else:
        print("No GPU available. Please check your ROCm installation.")

if __name__ == "__main__":
    test_gpu_simple() 