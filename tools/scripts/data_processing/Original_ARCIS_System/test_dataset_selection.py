#!/usr/bin/env python3
"""
Test script to demonstrate dataset selection functionality
"""

import os
from pathlib import Path

def main():
    print("=== ARCIS DATASET SELECTION DEMO ===")
    
    # Dataset selection (same as in main script)
    print("\n📊 AVAILABLE DATASETS:")
    datasets = {
        "1": {"path": "../ARCIS_Dataset_80_10_10", "name": "80/10/10 Split", "desc": "80% train, 10% val, 10% test (Recommended for most training)"},
        "2": {"path": "../ARCIS_Dataset_70_15_15", "name": "70/15/15 Split", "desc": "70% train, 15% val, 15% test (More validation data)"},
        "3": {"path": "../ARCIS_Dataset_75_12_12", "name": "75/12.5/12.5 Split", "desc": "75% train, 12.5% val, 12.5% test (Balanced approach)"}
    }
    
    for key, dataset in datasets.items():
        status = "✅" if os.path.exists(dataset["path"]) else "❌"
        print(f"{key}. {status} {dataset['name']} - {dataset['desc']}")
    
    # Show dataset statistics
    print("\n📈 DATASET STATISTICS:")
    for key, dataset in datasets.items():
        if os.path.exists(dataset["path"]):
            try:
                # Count images in each split
                train_path = Path(dataset["path"]) / "train" / "images"
                val_path = Path(dataset["path"]) / "val" / "images"
                test_path = Path(dataset["path"]) / "test" / "images"
                
                train_count = len(list(train_path.glob("*"))) if train_path.exists() else 0
                val_count = len(list(val_path.glob("*"))) if val_path.exists() else 0
                test_count = len(list(test_path.glob("*"))) if test_path.exists() else 0
                total_count = train_count + val_count + test_count
                
                print(f"\n  {dataset['name']}:")
                print(f"    📁 Train: {train_count:,} images")
                print(f"    📁 Val:   {val_count:,} images")
                print(f"    📁 Test:  {test_count:,} images")
                print(f"    📊 Total: {total_count:,} images")
                
                if total_count > 0:
                    train_pct = (train_count / total_count) * 100
                    val_pct = (val_count / total_count) * 100
                    test_pct = (test_count / total_count) * 100
                    print(f"    📈 Ratio: {train_pct:.1f}% / {val_pct:.1f}% / {test_pct:.1f}%")
                
            except Exception as e:
                print(f"    ❌ Error reading dataset: {e}")
        else:
            print(f"\n  {dataset['name']}: ❌ Not found")
    
    print("\n" + "="*60)
    print("💡 DATASET SELECTION GUIDE:")
    print("   1. 80/10/10 - Best for most training scenarios (more training data)")
    print("   2. 70/15/15 - Better for model validation and hyperparameter tuning")
    print("   3. 75/12.5/12.5 - Balanced approach between training and validation")
    print("\n🚀 To use in training, run: python train_weapon_detection.py")

if __name__ == "__main__":
    main() 