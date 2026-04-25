#!/bin/bash
set -e

echo "=========================================="
echo "  Oral Cancer AI — Full Training Pipeline"
echo "=========================================="

# Step 1: Split datasets (run once after download)
python -c "
from src.preprocess.split import split_dataset

# Adjust label_map folder names to match your actual downloaded folders
split_dataset('data/raw/kaggle_shivam', 'data/processed/shivam',
              {'cancer': 1, 'normal': 0})
split_dataset('data/raw/kaggle_zaidpy', 'data/processed/zaidpy',
              {'cancer': 1, 'normal': 0})
split_dataset('data/raw/mendeley',      'data/processed/mendeley',
              {'cancer': 1, 'normal': 0})
"

echo "--- Training Model 1: EfficientNet-B4 (Shivam dataset) ---"
python src/train/train_model1.py

echo "--- Training Model 2: ResNet-50 (Zaidpy dataset) ---"
python src/train/train_model2.py

echo "--- Training Model 3: DenseNet-121 (Mendeley dataset) ---"
python src/train/train_model3.py

echo "--- Training Ensemble Meta-Learner ---"
python src/train/train_ensemble.py

echo "--- Starting Flask Web App ---"
cd src && python app/app.py
