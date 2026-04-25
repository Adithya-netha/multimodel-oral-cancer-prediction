import torch
import torch.nn as nn
import timm

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,          # remove head
            global_pool='avg'
        )
        feat_dim = self.backbone.num_features  # 1792

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        """Returns penultimate features for ensemble."""
        return self.backbone(x)
