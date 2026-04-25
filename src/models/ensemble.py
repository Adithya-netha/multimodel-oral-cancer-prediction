import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import joblib

class EnsembleClassifier:
    """
    Combines 3 base CNNs via probability stacking + Logistic Regression.
    """
    def __init__(self, model1, model2, model3, device='cpu'):
        self.models = [model1, model2, model3]
        self.device = device
        self.meta_clf = None

    def _get_stacked_probs(self, loader):
        """
        Returns stacked probabilities from all 3 models: shape (N, 6)
        """
        all_stacked = []
        all_labels  = []

        for model in self.models:
            model.eval()

        for images, labels in loader:
            images = images.to(self.device)
            batch_probs = []
            with torch.no_grad():
                for model in self.models:
                    out = model(images)
                    probs = torch.softmax(out, dim=1).cpu().numpy()  # (B, 2)
                    batch_probs.append(probs)

            stacked = np.concatenate(batch_probs, axis=1)   # (B, 6)
            all_stacked.append(stacked)
            all_labels.extend(labels.numpy())

        return np.vstack(all_stacked), np.array(all_labels)

    def fit_meta_learner(self, val_loader):
        """Train meta-learner on validation predictions."""
        print("Collecting stacked probabilities for meta-learner training...")
        X_val, y_val = self._get_stacked_probs(val_loader)

        base_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs'
        )
        # Platt scaling for calibrated probabilities
        self.meta_clf = CalibratedClassifierCV(base_clf, cv=5)
        self.meta_clf.fit(X_val, y_val)

        print("Meta-learner trained.")

    def predict(self, loader):
        X, y = self._get_stacked_probs(loader)
        preds = self.meta_clf.predict(X)
        probs = self.meta_clf.predict_proba(X)[:, 1]
        return preds, probs, y

    def predict_single(self, image_tensor):
        """
        image_tensor: (1, 3, H, W) already on self.device
        Returns (predicted_class, cancer_probability)
        """
        image_tensor = image_tensor.to(self.device)
        batch_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                out = model(image_tensor)
                probs = torch.softmax(out, dim=1).cpu().numpy()
                batch_probs.append(probs)

        stacked = np.concatenate(batch_probs, axis=1)   # (1, 6)
        pred  = self.meta_clf.predict(stacked)[0]
        prob  = self.meta_clf.predict_proba(stacked)[0, 1]
        return int(pred), float(prob)

    def save(self, path='saved_models/meta_learner.pkl'):
        joblib.dump(self.meta_clf, path)
        print(f"Meta-learner saved to {path}")

    def load(self, path='saved_models/meta_learner.pkl'):
        self.meta_clf = joblib.load(path)
        print(f"Meta-learner loaded from {path}")
