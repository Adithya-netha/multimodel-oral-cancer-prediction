import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.35, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        feat_dim = backbone.classifier.in_features  # 1024

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_features(self, x):
        x = self.features(x)
        return self.pool(x).squeeze(-1).squeeze(-1)
