import os, sys, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.preprocess.dataset import OralCancerDataset
from src.preprocess.transforms import get_train_transforms, get_val_transforms
from src.models.model2_zaidpy import ResNetModel
from src.evaluate.metrics import compute_metrics

DATA_ROOT  = 'data/processed/zaidpy'
SAVE_PATH  = 'saved_models/model2_resnet50.pth'
BATCH_SIZE = 32
EPOCHS     = 25
LR         = 0.01
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP  = {'cancer': 1, 'normal': 0}   # adjust to your folder names

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
    train_ds = OralCancerDataset(
        os.path.join(DATA_ROOT, 'train'),
        transform=get_train_transforms(), label_map=LABEL_MAP)
    val_ds = OralCancerDataset(
        os.path.join(DATA_ROOT, 'val'),
        transform=get_val_transforms(), label_map=LABEL_MAP)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4)

    model = ResNetModel(num_classes=2, pretrained=True).to(DEVICE)
    class_weights = get_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = SGD(model.parameters(), lr=LR,
                    momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = OneCycleLR(optimizer, max_lr=LR,
                           steps_per_epoch=len(train_loader), epochs=EPOCHS)
    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images)
                probs = torch.softmax(out, dim=1)[:, 1]
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        m = compute_metrics(all_labels, all_preds, all_probs)
        print(f"Epoch {epoch:02d} | Acc={m['accuracy']:.4f} "
              f"F1={m['f1']:.4f} AUC={m['auc']:.4f}")
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  > Saved (F1={best_f1:.4f})")

if __name__ == '__main__':
    train()
