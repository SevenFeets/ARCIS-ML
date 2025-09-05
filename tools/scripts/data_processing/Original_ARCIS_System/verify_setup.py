#!/usr/bin/env python3
"""
ARCIS Setup Verification Script
Verifies that all paths and dependencies are working correctly after folder organization.
"""

import os
import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists and print result"""
    exists = os.path.exists(path)
    status = "✅ FOUND" if exists else "❌ MISSING"
    print(f"{status}: {description} - {path}")
    return exists

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ IMPORT OK: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ IMPORT FAILED: {module_name} - {e}")
        return False

def main():
    print("=== ARCIS SETUP VERIFICATION ===\n")
    
    all_good = True
    
    # Check dataset paths
    print("📁 DATASET PATHS:")
    datasets = [
        ("../ARCIS_Dataset_80_10_10", "80/10/10 Split (Recommended)"),
        ("../ARCIS_Dataset_70_15_15", "70/15/15 Split (More validation)"),
        ("../ARCIS_Dataset_75_12_12", "75/12.5/12.5 Split (Balanced)")
    ]
    
    available_datasets = 0
    for dataset_path, dataset_name in datasets:
        data_yaml = f"{dataset_path}/data.yaml"
        train_images = f"{dataset_path}/train/images"
        val_images = f"{dataset_path}/val/images"
        test_images = f"{dataset_path}/test/images"
        
        print(f"\n  {dataset_name}:")
        dataset_ok = True
        if not check_file_exists(data_yaml, "  Configuration file"):
            dataset_ok = False
        if not check_file_exists(train_images, "  Training images"):
            dataset_ok = False
        if not check_file_exists(val_images, "  Validation images"):
            dataset_ok = False
        if not check_file_exists(test_images, "  Test images"):
            dataset_ok = False
        
        if dataset_ok:
            available_datasets += 1
            print(f"  ✅ {dataset_name} - Complete")
        else:
            print(f"  ❌ {dataset_name} - Incomplete")
            all_good = False
    
    print(f"\n📊 Available datasets: {available_datasets}/3")
    
    print("\n🤖 MODEL PATHS:")
    model_checks = [
        ("../Models/yolov8n.pt", "YOLOv8 Nano model"),
        ("../Models/yolo11n.pt", "YOLO11 Nano model"),
    ]
    
    for path, desc in model_checks:
        if not check_file_exists(path, desc):
            all_good = False
    
    print("\n🔊 AUDIO ASSETS:")
    audio_checks = [
        ("../Audio_Assets/danger_alert.mp3", "Danger alert sound"),
    ]
    
    for path, desc in audio_checks:
        if not check_file_exists(path, desc):
            all_good = False
    
    print("\n📦 PYTHON DEPENDENCIES:")
    required_modules = [
        "ultralytics",
        "torch", 
        "cv2",
        "pygame",
        "numpy",
        "matplotlib",
        "yaml",
        "json",
        "pathlib"
    ]
    
    for module in required_modules:
        if not check_import(module):
            all_good = False
    
    print("\n🧪 FUNCTIONALITY TESTS:")
    
    # Test YOLO model loading
    try:
        from ultralytics import YOLO
        model = YOLO("../Models/yolov8n.pt")
        print("✅ YOLO MODEL LOADING: Success")
    except Exception as e:
        print(f"❌ YOLO MODEL LOADING: Failed - {e}")
        all_good = False
    
    # Test audio system
    try:
        import pygame
        pygame.mixer.init()
        if os.path.exists("../Audio_Assets/danger_alert.mp3"):
            sound = pygame.mixer.Sound("../Audio_Assets/danger_alert.mp3")
            print("✅ AUDIO SYSTEM: Success")
        else:
            print("❌ AUDIO SYSTEM: Audio file not found")
            all_good = False
    except Exception as e:
        print(f"❌ AUDIO SYSTEM: Failed - {e}")
        all_good = False
    
    # Test dataset loading
    try:
        import yaml
        with open("../ARCIS_Dataset_80_10_10/data.yaml", 'r') as f:
            data = yaml.safe_load(f)
        class_names = data.get('names', [])
        print(f"✅ DATASET LOADING: Success - {len(class_names)} classes found")
    except Exception as e:
        print(f"❌ DATASET LOADING: Failed - {e}")
        all_good = False
    
    print("\n" + "="*50)
    if all_good:
        print("🎉 ALL CHECKS PASSED! ARCIS system is ready to run.")
        print("\nYou can now run:")
        print("  python train_weapon_detection.py")
    else:
        print("⚠️  SOME CHECKS FAILED! Please fix the issues above.")
        print("\nCommon solutions:")
        print("  - Run dataset merger: cd ../Dataset_Tools && python dataset_merger.py")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check file paths and folder organization")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 