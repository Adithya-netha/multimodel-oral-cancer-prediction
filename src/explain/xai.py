import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import shap

class ExplainabilityModule:
    """
    Provides Grad-CAM visualizations and SHAP explanations.
    """
    def __init__(self, model, target_layer, device='cpu'):
        """
        model       : trained PyTorch model
        target_layer: the conv layer to hook (e.g. model.backbone.conv_head)
        device      : 'cuda' or 'cpu'
        """
        self.model = model.eval().to(device)
        self.device = device
        self.cam = GradCAM(model=model, target_layers=[target_layer])

    def gradcam(self, image_tensor, class_idx=None, save_path=None):
        """
        image_tensor : (1, 3, H, W) float tensor (NOT normalized for display)
        raw_image    : (H, W, 3) uint8 np array for overlay
        """
        image_tensor = image_tensor.to(self.device)
        # Always target class 1 (Cancer) for more intuitive medical visualizations
        targets = [ClassifierOutputTarget(class_idx if class_idx is not None else 1)]
        
        grayscale_cam = self.cam(
            input_tensor=image_tensor,
            targets=targets
        )
        return grayscale_cam[0]   # (H, W)

    def overlay_cam(self, raw_image_rgb, cam_map, save_path=None):
        """
        raw_image_rgb : (H, W, 3) uint8 numpy array
        cam_map       : (H, W) float array from gradcam()
        """
        rgb_float = raw_image_rgb.astype(np.float32) / 255.0
        visualization = show_cam_on_image(rgb_float, cam_map, use_rgb=True)
        if save_path:
            cv2.imwrite(save_path,
                        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        return visualization

    def shap_explain(self, background_loader, test_images, num_samples=50):
        """
        Uses DeepSHAP to explain model predictions.
        background_loader : DataLoader with background samples
        test_images       : (N, 3, H, W) tensor
        """
        # Collect background samples
        background = []
        for imgs, _ in background_loader:
            background.append(imgs)
            if len(background) * imgs.shape[0] >= num_samples:
                break
        background = torch.cat(background)[:num_samples].to(self.device)

        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(
            test_images.to(self.device)
        )
        return shap_values   # list of arrays, one per class

    def plot_shap(self, shap_values, test_images, class_idx=1,
                  max_display=4, save_path=None):
        """Plot SHAP image explanations."""
        # shap_values[class_idx] shape: (N, C, H, W)
        sv = shap_values[class_idx][:max_display]
        imgs = test_images[:max_display].cpu().numpy()

        shap.image_plot(
            [sv],
            imgs.transpose(0, 2, 3, 1),
            show=False
        )
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()
