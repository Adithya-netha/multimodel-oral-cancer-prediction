import os
import subprocess
import sys
from src.preprocess.split import split_dataset

def main():
    print("==========================================")
    print("  Oral Cancer AI — Full Training Pipeline")
    print("==========================================")

    print("\n--- Step 1: Splitting Datasets ---")
    
    shivam_src = os.path.join('data', 'raw', 'kaggle_shivam', 'OralCancer')
    shivam_dst = os.path.join('data', 'processed', 'shivam')
    print(f"Splitting Shivam dataset from {shivam_src}...")
    split_dataset(shivam_src, shivam_dst, 
                  {'cancer': 'cancer', 'normal': 'non-cancer'})

    zaidpy_src = os.path.join('data', 'raw', 'kaggle_zaidpy', 'Oral Cancer', 'Oral Cancer Dataset')
    zaidpy_dst = os.path.join('data', 'processed', 'zaidpy')
    print(f"Splitting Zaidpy dataset from {zaidpy_src}...")
    split_dataset(zaidpy_src, zaidpy_dst, 
                  {'cancer': 'CANCER', 'normal': 'NON CANCER'})

    mendeley_src = os.path.join('data', 'raw', 'mendeley', 'First Set')
    mendeley_dst = os.path.join('data', 'processed', 'mendeley')
    print(f"Splitting Mendeley dataset from {mendeley_src}...")
    split_dataset(mendeley_src, mendeley_dst, 
                  {'cancer': '100x OSCC Histopathological Images', 
                   'normal': '100x Normal Oral Cavity Histopathological Images'})

    print("\n--- Step 2: Training Models ---")
    
    scripts_to_run = [
        ("Model 1 (EfficientNet-B4)", os.path.join("src", "train", "train_model1.py")),
        ("Model 2 (ResNet-50)", os.path.join("src", "train", "train_model2.py")),
        ("Model 3 (DenseNet-121)", os.path.join("src", "train", "train_model3.py")),
        ("Ensemble Meta-Learner", os.path.join("src", "train", "train_ensemble.py"))
    ]

    for name, script in scripts_to_run:
        print(f"\n--- Training {name} ---")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Error occurred while training {name}. Exiting.")
            sys.exit(result.returncode)

    print("\n==========================================")
    print("  Pipeline Completed Successfully!")
    print("==========================================")
    print("To start the web app, run: python src/app/app.py")

if __name__ == '__main__':
    main()
