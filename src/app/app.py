import os, sys, io, base64
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import joblib
import cv2

# Ensure root path is added
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model1_shivam   import EfficientNetModel
from src.models.model2_zaidpy   import ResNetModel
from src.models.model3_mendeley import DenseNetModel
from src.models.ensemble        import EnsembleClassifier
from src.explain.xai            import ExplainabilityModule
from src.preprocess.transforms  import get_val_transforms

app = Flask(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_model_path(filename):
    return os.path.join(BASE_DIR, 'saved_models', filename)

# ── Load models at startup ──────────────────────────────────────────────────
def load_model(cls, path, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    m = cls(**kwargs)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()

#  Load models with correct paths
model1 = load_model(
    EfficientNetModel,
    get_model_path('model1_efficientnet.pth'),
    num_classes=2,
    pretrained=False
)

model2 = load_model(
    ResNetModel,
    get_model_path('model2_resnet50.pth'),
    num_classes=2,
    pretrained=False
)

model3 = load_model(
    DenseNetModel,
    get_model_path('model3_densenet121.pth'),
    num_classes=2,
    pretrained=False
)

ensemble = EnsembleClassifier(model1, model2, model3, device=DEVICE)
ensemble.load(get_model_path('meta_learner.pkl'))

# Grad-CAM on ResNet-50 (Superior localization for medical images)
xai = ExplainabilityModule(
    model2,
    target_layer=model2.features[7],
    device=DEVICE
)

transforms = get_val_transforms()

# ── Helpers ─────────────────────────────────────────────────────────────────
def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    raw = np.array(img.resize((224, 224)))
    aug = transforms(image=raw)
    tensor = aug['image'].unsqueeze(0)
    return tensor, raw

def img_to_b64(arr):
    _, buf = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode('utf-8')

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file_bytes = request.files['image'].read()
    tensor, raw = preprocess_image(file_bytes)

    # Ensemble prediction
    pred_class, cancer_prob = ensemble.predict_single(tensor)

    # Grad-CAM
    cam_map = xai.gradcam(tensor)
    overlay = xai.overlay_cam(raw, cam_map)
    cam_b64 = img_to_b64(overlay)

    # Individual model confidences
    individual = {}
    for name, m in [
        ('EfficientNet-B4', model1),
        ('ResNet-50', model2),
        ('DenseNet-121', model3)
    ]:
        with torch.no_grad():
            out = m(tensor.to(DEVICE))
            p = torch.softmax(out, dim=1)[0, 1].item()
        individual[name] = round(p * 100, 2)

    result = {
        'prediction': 'Cancer' if pred_class == 1 else 'Normal',
        'cancer_probability': round(cancer_prob * 100, 2),
        'risk_level': (
            'High'   if cancer_prob >= 0.7 else
            'Medium' if cancer_prob >= 0.4 else
            'Low'
        ),
        'individual_models': individual,
        'gradcam_image': cam_b64
    }
    return jsonify(result)

if __name__ == '__main__':
    print("Model path check:")
    print(get_model_path('model1_efficientnet.pth'))
    print("Exists:", os.path.exists(get_model_path('model1_efficientnet.pth')))
    
    app.run(debug=True, host='0.0.0.0', port=5000)