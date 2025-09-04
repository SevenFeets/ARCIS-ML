#!/usr/bin/env python3
import torch
import time

def test_gpu():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        
        # Test GPU performance with a simple matrix multiplication
        size = 5000
        print(f"\nTesting GPU performance with {size}x{size} matrix multiplication...")
        
        # CPU test
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        print(f"CPU time: {cpu_time:.2f} seconds")
        
        # GPU test
        start_time = time.time()
        a_gpu = torch.randn(size, size, device="cuda")
        b_gpu = torch.randn(size, size, device="cuda")
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        gpu_setup_time = time.time() - start_time
        
        start_time = time.time()
        c_gpu = torch.matmul(a_gpu, b_gpu)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        gpu_compute_time = time.time() - start_time
        
        print(f"GPU setup time: {gpu_setup_time:.2f} seconds")
        print(f"GPU compute time: {gpu_compute_time:.2f} seconds")
        print(f"Total GPU time: {gpu_setup_time + gpu_compute_time:.2f} seconds")
        print(f"Speed improvement: {cpu_time / gpu_compute_time:.2f}x faster computation")
        
        # Verify results match
        c_gpu_on_cpu = c_gpu.cpu()
        max_diff = torch.max(torch.abs(c_cpu - c_gpu_on_cpu)).item()
        print(f"Maximum difference between CPU and GPU results: {max_diff}")
    else:
        print("No GPU available. Please check your ROCm installation.")

if __name__ == "__main__":
    test_gpu() 