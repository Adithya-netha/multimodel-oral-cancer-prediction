import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class OralCancerDataset(Dataset):
    """
    Generic dataset class for all three datasets.
    Expected folder layout:
        root/
          cancer/   (or malignant/, positive/, etc.)
          normal/   (or benign/, negative/, etc.)
    Pass label_map to handle different folder names.
    """
    def __init__(self, root_dir, transform=None,
                 label_map=None):
        """
        label_map: dict mapping folder name -> int label
                   e.g. {'cancer': 1, 'normal': 0}
                   If None, auto-detects two folders (0 and 1).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        if label_map is None:
            folders = sorted([
                f for f in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, f))
            ])
            label_map = {f: i for i, f in enumerate(folders)}

        for folder, label in label_map.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found, skipping.")
                continue
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append(
                        (os.path.join(folder_path, fname), label)
                    )

        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"Label map: {label_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label
