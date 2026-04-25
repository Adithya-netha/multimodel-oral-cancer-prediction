import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# ── local imports ──────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.preprocess.dataset import OralCancerDataset
from src.preprocess.transforms import get_train_transforms, get_val_transforms
from src.models.model1_shivam import EfficientNetModel
from src.evaluate.metrics import compute_metrics

# ── Config ─────────────────────────────────────────────────────────────────
DATA_ROOT   = 'data/processed/shivam'
SAVE_PATH   = 'saved_models/model1_efficientnet.pth'
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 3e-4
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP   = {'cancer': 1, 'normal': 0}   # adjust to your folder names

def get_class_weights(dataset):
    labels = [s[1] for s in dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total = len(labels)
    weights = torch.tensor([total / (2 * n_neg),
                            total / (2 * n_pos)], dtype=torch.float)
    return weights

def train():
    os.makedirs('saved_models', exist_ok=True)

    # Datasets
    train_ds = OralCancerDataset(
        os.path.join(DATA_ROOT, 'train'),
        transform=get_train_transforms(),
        label_map=LABEL_MAP
    )
    val_ds = OralCancerDataset(
        os.path.join(DATA_ROOT, 'val'),
        transform=get_val_transforms(),
        label_map=LABEL_MAP
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

    # Model
    model = EfficientNetModel(num_classes=2, pretrained=True).to(DEVICE)

    # Weighted cross entropy to handle class imbalance
    class_weights = get_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader,
                                   desc=f'Epoch {epoch}/{EPOCHS} [Train]'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        scheduler.step()

        print(f"Epoch {epoch:02d} | "
              f"TrainLoss: {train_loss/len(train_loader):.4f} | "
              f"ValLoss: {val_loss/len(val_loader):.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"AUC: {metrics['auc']:.4f}")

        if metrics['f1'] > best_val_f1:
            best_val_f1 = metrics['f1']
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  > Best model saved (F1={best_val_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_val_f1:.4f}")

if __name__ == '__main__':
    train()
