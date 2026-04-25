import os, sys, torch
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.preprocess.dataset import OralCancerDataset
from src.preprocess.transforms import get_val_transforms
from src.models.model1_shivam  import EfficientNetModel
from src.models.model2_zaidpy  import ResNetModel
from src.models.model3_mendeley import DenseNetModel
from src.models.ensemble        import EnsembleClassifier
from src.evaluate.metrics       import compute_metrics, print_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LABEL_MAP = {'cancer': 1, 'normal': 0}   # adjust per dataset if needed

def load_model(cls, path, **kwargs):
    m = cls(**kwargs)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE)

def main():
    # ── Load trained base models ─────────────────────────────────────────
    model1 = load_model(EfficientNetModel,
                        'saved_models/model1_efficientnet.pth',
                        num_classes=2, pretrained=False)
    model2 = load_model(ResNetModel,
                        'saved_models/model2_resnet50.pth',
                        num_classes=2, pretrained=False)
    model3 = load_model(DenseNetModel,
                        'saved_models/model3_densenet121.pth',
                        num_classes=2, pretrained=False)

    # ── Build combined val & test loaders ────────────────────────────────
    val_datasets = [
        OralCancerDataset('data/processed/shivam/val',
                          get_val_transforms(), LABEL_MAP),
        OralCancerDataset('data/processed/zaidpy/val',
                          get_val_transforms(), LABEL_MAP),
        OralCancerDataset('data/processed/mendeley/val',
                          get_val_transforms(), LABEL_MAP),
    ]
    test_datasets = [
        OralCancerDataset('data/processed/shivam/test',
                          get_val_transforms(), LABEL_MAP),
        OralCancerDataset('data/processed/zaidpy/test',
                          get_val_transforms(), LABEL_MAP),
        OralCancerDataset('data/processed/mendeley/test',
                          get_val_transforms(), LABEL_MAP),
    ]

    val_loader  = DataLoader(ConcatDataset(val_datasets),
                             batch_size=32, shuffle=False)
    test_loader = DataLoader(ConcatDataset(test_datasets),
                             batch_size=32, shuffle=False)

    # ── Train ensemble ───────────────────────────────────────────────────
    ensemble = EnsembleClassifier(model1, model2, model3, device=DEVICE)
    ensemble.fit_meta_learner(val_loader)
    ensemble.save('saved_models/meta_learner.pkl')

    # ── Evaluate on test set ─────────────────────────────────────────────
    preds, probs, labels = ensemble.predict(test_loader)
    metrics = compute_metrics(labels, preds, probs)
    print_report(metrics)

if __name__ == '__main__':
    main()
