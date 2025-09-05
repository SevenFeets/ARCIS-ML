# ARCIS Dataset Selection Guide

This guide helps you choose the best dataset split for your specific training needs.

## 📊 Available Dataset Splits

You have 3 different dataset splits available, all containing the same **248,374 images** with **442,399 annotations** across **19 threat classes**:

### 1. 80/10/10 Split (Recommended for Most Cases)
- **Training**: 198,699 images (80%)
- **Validation**: 24,837 images (10%)
- **Testing**: 24,838 images (10%)
- **Best for**: General training, production models, maximum training data

### 2. 70/15/15 Split (Best for Model Development)
- **Training**: 173,861 images (70%)
- **Validation**: 37,256 images (15%)
- **Testing**: 37,257 images (15%)
- **Best for**: Hyperparameter tuning, model comparison, research

### 3. 75/12.5/12.5 Split (Balanced Approach)
- **Training**: 186,280 images (75%)
- **Validation**: 31,046 images (12.5%)
- **Testing**: 31,048 images (12.5%)
- **Best for**: Balanced training and validation, general purpose

## 🎯 When to Use Each Split

### Use 80/10/10 Split When:
- ✅ **Production deployment** - You need the best possible model accuracy
- ✅ **Limited training time** - More training data = faster convergence
- ✅ **Stable model architecture** - You know your model works and just need to train it
- ✅ **Resource constraints** - Less validation data = faster training epochs
- ✅ **Final model training** - You've already done hyperparameter tuning

### Use 70/15/15 Split When:
- ✅ **Model development** - You're experimenting with different architectures
- ✅ **Hyperparameter tuning** - You need more validation data for reliable metrics
- ✅ **Model comparison** - Comparing different YOLO versions or configurations
- ✅ **Research projects** - You need robust validation for publications
- ✅ **Early stopping** - More validation data gives better early stopping decisions

### Use 75/12.5/12.5 Split When:
- ✅ **General purpose training** - Good balance for most scenarios
- ✅ **Uncertain requirements** - You're not sure which approach is best
- ✅ **Medium-scale projects** - Balance between training and validation needs
- ✅ **Educational purposes** - Good for learning and experimentation

## 🚀 How to Select Dataset in Training

When you run the training script, you'll see this menu:

```bash
cd Original_ARCIS_System
python train_weapon_detection.py
```

```
📊 AVAILABLE DATASETS:
1. ✅ 80/10/10 Split - 80% train, 10% val, 10% test (Recommended for most training)
2. ✅ 70/15/15 Split - 70% train, 15% val, 15% test (More validation data)
3. ✅ 75/12.5/12.5 Split - 75% train, 12.5% val, 12.5% test (Balanced approach)

Select dataset (1-3, press Enter for default 80/10/10):
```

Simply enter the number (1, 2, or 3) or press Enter for the default.

## 📈 Performance Expectations

### Training Speed
- **80/10/10**: Fastest training (more training data, less validation overhead)
- **70/15/15**: Slower training (less training data, more validation overhead)
- **75/12.5/12.5**: Medium training speed

### Model Accuracy
- **80/10/10**: Potentially highest accuracy (more training data)
- **70/15/15**: Good accuracy with better validation reliability
- **75/12.5/12.5**: Balanced accuracy and validation

### Validation Reliability
- **80/10/10**: Less reliable validation metrics (smaller validation set)
- **70/15/15**: Most reliable validation metrics (larger validation set)
- **75/12.5/12.5**: Good validation reliability

## 🔧 Advanced Considerations

### For Jetson Nano Deployment
- **Recommended**: 80/10/10 split
- **Reason**: Maximum training data for best model accuracy on resource-constrained device

### For Research and Development
- **Recommended**: 70/15/15 split
- **Reason**: Better validation for hyperparameter tuning and model comparison

### For Production Systems
- **Recommended**: 80/10/10 split for final training, 70/15/15 for development
- **Workflow**: Develop with 70/15/15, then final training with 80/10/10

## 🛠️ Switching Between Datasets

You can easily switch between datasets for different experiments:

```bash
# Train with 70/15/15 for hyperparameter tuning
python train_weapon_detection.py
# Select: 2 (70/15/15 split)
# Select: 1 (Train for Jetson)

# Then train final model with 80/10/10
python train_weapon_detection.py  
# Select: 1 (80/10/10 split)
# Select: 1 (Train for Jetson)
```

## 📊 Dataset Quality Verification

To check all datasets are properly set up:

```bash
python verify_setup.py
```

To see detailed statistics:

```bash
python test_dataset_selection.py
```

## 💡 Best Practices

1. **Start with 70/15/15** for initial experiments and hyperparameter tuning
2. **Use 80/10/10** for final production model training
3. **Use 75/12.5/12.5** when you're unsure or for general-purpose training
4. **Always verify** your dataset choice matches your training goals
5. **Document** which dataset split you used for reproducibility

## 🔄 Workflow Recommendations

### Research/Development Workflow
```
1. Use 70/15/15 split for initial experiments
2. Try different model architectures and hyperparameters
3. Select best configuration based on validation metrics
4. Final training with 80/10/10 split for production model
```

### Production Workflow
```
1. Use 80/10/10 split for maximum training data
2. Train with proven hyperparameters
3. Deploy to Jetson Nano or production environment
4. Monitor real-world performance
```

### Educational Workflow
```
1. Start with 75/12.5/12.5 split for balanced learning
2. Experiment with different training options
3. Compare results across different splits
4. Understand impact of data distribution on model performance
```

---

**Note**: All dataset splits contain identical images and annotations, just distributed differently across train/val/test sets. The choice depends on your specific training objectives and validation requirements. 