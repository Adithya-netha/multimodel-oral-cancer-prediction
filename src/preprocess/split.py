import os
import shutil
import random
from pathlib import Path

def split_dataset(src_root, dst_root, label_map,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  seed=42):
    """
    Splits images into train/val/test folders.
    src_root: path containing subfolders (e.g. cancer/, normal/)
    dst_root: output root (train/, val/, test/ created here)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)

    for split in ['train', 'val', 'test']:
        for folder in label_map.keys():
            Path(os.path.join(dst_root, split, folder)).mkdir(
                parents=True, exist_ok=True)

    for target_folder, source_folder in label_map.items():
        src = os.path.join(src_root, source_folder)
        if not os.path.exists(src):
            continue
        files = [f for f in os.listdir(src)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': files[:n_train],
            'val':   files[n_train:n_train + n_val],
            'test':  files[n_train + n_val:]
        }

        for split, split_files in splits.items():
            for f in split_files:
                shutil.copy2(
                    os.path.join(src, f),
                    os.path.join(dst_root, split, target_folder, f)
                )
        print(f"{target_folder} (from {source_folder}): train={len(splits['train'])}, "
              f"val={len(splits['val'])}, test={len(splits['test'])}")

# ── Usage ──────────────────────────────────────────────────────────────────
# Run once for each dataset with the correct label_map for its folder names.
# Example:
#
# split_dataset(
#     src_root='data/raw/kaggle_shivam',
#     dst_root='data/processed/shivam',
#     label_map={'cancer': 1, 'normal': 0}
# )
