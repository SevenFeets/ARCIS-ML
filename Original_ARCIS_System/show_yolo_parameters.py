#!/usr/bin/env python3
"""
Simple YOLO Parameter Guide for ARCIS
"""

def show_parameters():
    print("=== YOLO HYPERPARAMETER GUIDE ===\n")
    
    print("LEARNING RATE PARAMETERS:")
    print("  lr0: Initial learning rate (0.001-0.1)")
    print("    - Higher: Faster learning, risk of instability")
    print("    - Lower: Stable learning, slower convergence")
    print("  lrf: Final learning rate (lr0 * lrf)")
    print("    - Controls learning rate decay")
    
    print("\nOPTIMIZATION PARAMETERS:")
    print("  momentum: SGD momentum (0.8-0.99)")
    print("    - Higher: Better convergence, more stable")
    print("  weight_decay: L2 regularization (0.0001-0.01)")
    print("    - Higher: Prevents overfitting, may reduce accuracy")
    
    print("\nWARMUP PARAMETERS:")
    print("  warmup_epochs: Gradual learning rate increase (1-10)")
    print("    - More epochs: Stable start, slower initial training")
    print("  warmup_momentum: Initial momentum during warmup")
    print("  warmup_bias_lr: Bias learning rate during warmup")
    
    print("\nLOSS FUNCTION WEIGHTS:")
    print("  box: Bounding box loss weight (1-20)")
    print("    - Higher: Better localization, important for weapons")
    print("  cls: Classification loss weight (0.1-2)")
    print("    - Higher: Better class prediction accuracy")
    print("  dfl: Distribution focal loss weight (0.5-5)")
    print("    - Higher: Better bounding box regression")
    
    print("\nDATA AUGMENTATION:")
    print("  hsv_h/s/v: Color space augmentation (0-1)")
    print("  degrees: Rotation augmentation (0-45)")
    print("  translate: Translation augmentation (0-0.5)")
    print("  scale: Scale augmentation (0-1)")
    print("  fliplr: Horizontal flip probability (0-1)")
    print("  mosaic: Mosaic augmentation probability (0-1)")
    print("  mixup: Mixup augmentation probability (0-1)")

def show_presets():
    print("\n=== AVAILABLE PRESETS ===\n")
    
    print("1. DEFAULT YOLO SETTINGS")
    print("   - Standard parameters for general use")
    print("   - lr0=0.01, momentum=0.937, box=7.5, cls=0.5")
    
    print("\n2. WEAPON DETECTION OPTIMIZED")
    print("   - Optimized for small weapon detection")
    print("   - lr0=0.008, box=10.0, cls=0.8, reduced augmentation")
    
    print("\n3. JETSON NANO OPTIMIZED")
    print("   - Speed-focused for Jetson deployment")
    print("   - lr0=0.012, faster convergence, balanced weights")
    
    print("\n4. HIGH ACCURACY (SLOW)")
    print("   - Maximum accuracy for research")
    print("   - lr0=0.005, box=15.0, minimal augmentation")

if __name__ == "__main__":
    show_parameters()
    show_presets() 