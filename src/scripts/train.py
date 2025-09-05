#!/usr/bin/env python3
"""
ARCIS Training Script

This script provides a unified interface for training weapon detection models
across different architectures and deployment targets.

Usage:
    python train.py --model yolov8n --dataset merged_dataset_80_10_10_FULL --epochs 100
    python train.py --platform raspberry_pi --model yolov8n --quantize
    python train.py --model yolov8x --dataset merged_dataset --epochs 300 --batch 64
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


class ARCISTrainer:
    """ARCIS Model Trainer"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data" / "processed"
        self.models_dir = self.project_root / "models"
        self.experiments_dir = self.project_root / "experiments"
        
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get dataset configuration path"""
        dataset_path = self.data_dir / dataset_name / "data.yaml"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        return str(dataset_path)
    
    def get_model_path(self, model_name: str) -> str:
        """Get model path"""
        if model_name.startswith('yolo'):
            model_path = self.models_dir / "pretrained" / f"{model_name}.pt"
        else:
            model_path = self.models_dir / "pretrained" / model_name
            
        if not model_path.exists():
            # Use default YOLO model
            return model_name
            
        return str(model_path)
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific training configuration"""
        configs = {
            'raspberry_pi': {
                'imgsz': 320,
                'batch': 8,
                'workers': 2,
                'device': 'cpu',
                'quantize': True
            },
            'jetson_nano': {
                'imgsz': 416,
                'batch': 16,
                'workers': 4,
                'device': '0',
                'quantize': False
            },
            'mobile': {
                'imgsz': 320,
                'batch': 16,
                'workers': 4,
                'device': 'cpu',
                'quantize': True
            },
            'cloud': {
                'imgsz': 640,
                'batch': 64,
                'workers': 8,
                'device': '0',
                'quantize': False
            },
            'enterprise': {
                'imgsz': 640,
                'batch': 128,
                'workers': 16,
                'device': '0',
                'quantize': False
            }
        }
        return configs.get(platform, {})
    
    def train(self, args):
        """Train model with given arguments"""
        try:
            # Get dataset path
            dataset_path = self.get_dataset_path(args.dataset)
            
            # Get model path
            model_path = self.get_model_path(args.model)
            
            # Load model
            model = YOLO(model_path)
            
            # Get platform configuration
            platform_config = self.get_platform_config(args.platform)
            
            # Prepare training arguments
            train_args = {
                'data': dataset_path,
                'epochs': args.epochs,
                'imgsz': args.imgsz or platform_config.get('imgsz', 640),
                'batch': args.batch or platform_config.get('batch', 16),
                'workers': args.workers or platform_config.get('workers', 8),
                'device': args.device or platform_config.get('device', '0'),
                'project': str(self.experiments_dir / "runs"),
                'name': f"{args.model}_{args.platform}_{args.dataset}",
                'exist_ok': True,
                'verbose': True
            }
            
            # Add quantization if requested
            if args.quantize or platform_config.get('quantize', False):
                train_args['quantize'] = True
            
            # Add data augmentation
            if args.augment:
                train_args.update({
                    'mixup': 0.15,
                    'mosaic': 1.0,
                    'degrees': 10,
                    'translate': 0.1,
                    'scale': 0.5,
                    'shear': 2.0,
                    'perspective': 0.0,
                    'flipud': 0.0,
                    'fliplr': 0.5,
                    'hsv_h': 0.015,
                    'hsv_s': 0.7,
                    'hsv_v': 0.4,
                    'copy_paste': 0.3
                })
            
            print(f"Training {args.model} on {args.dataset} for {args.platform}")
            print(f"Configuration: {train_args}")
            
            # Train model
            results = model.train(**train_args)
            
            print(f"Training completed successfully!")
            print(f"Results saved to: {self.experiments_dir / 'runs' / train_args['name']}")
            
            return results
            
        except Exception as e:
            print(f"Training failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='ARCIS Model Training')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='yolov8n',
                       help='Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    parser.add_argument('--dataset', type=str, default='merged_dataset_80_10_10_FULL',
                       help='Dataset name (merged_dataset, merged_dataset_75_15, merged_dataset_80_10_10_FULL)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=None,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of workers')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cpu, 0, 1, etc.)')
    
    # Platform arguments
    parser.add_argument('--platform', type=str, default='cloud',
                       choices=['raspberry_pi', 'jetson_nano', 'mobile', 'cloud', 'enterprise'],
                       help='Target deployment platform')
    
    # Optimization arguments
    parser.add_argument('--quantize', action='store_true',
                       help='Enable quantization')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ARCISTrainer()
    
    # Train model
    results = trainer.train(args)
    
    if results is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
