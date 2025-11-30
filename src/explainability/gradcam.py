"""
Grad-CAM (Gradient-weighted Class Activation Mapping) utilities for CNN explainability.

This module provides tools to visualize which regions of spectrograms the model
focuses on when making predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
from typing import Optional, Tuple, Dict
from pathlib import Path


class GradCAMExplainer:
    """
    Wrapper class for Grad-CAM analysis on CNN models.
    
    This class handles:
    - Grad-CAM heatmap generation
    - Visualization of heatmaps on spectrograms
    - Batch processing of multiple samples
    - Comparison of clean vs. adversarial examples
    """
    
    def __init__(self, model, target_layer, device=None):
        """
        Initialize the Grad-CAM explainer.
        
        Args:
            model: PyTorch CNN model
            target_layer: Target convolutional layer for Grad-CAM
                         (typically the last conv layer)
            device: Device to run on (cuda/cpu). If None, auto-detects.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        
        # Initialize Grad-CAM
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])
    
    def generate_heatmap(
        self, 
        input_tensor: torch.Tensor, 
        target_class: int
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single input.
        
        Args:
            input_tensor: Input tensor (1, C, H, W) or (C, H, W)
            target_class: Target class index for Grad-CAM
            
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
        """
        # Ensure batch dimension
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Move to device
        input_tensor = input_tensor.to(self.device)
        
        # Generate heatmap
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        
        # Extract heatmap for single image
        heatmap = grayscale_cam[0, :]
        
        return heatmap
    
    def generate_batch_heatmaps(
        self,
        input_batch: torch.Tensor,
        target_classes: np.ndarray
    ) -> list:
        """
        Generate Grad-CAM heatmaps for a batch of inputs.
        
        Args:
            input_batch: Batch of input tensors (N, C, H, W)
            target_classes: Array of target class indices (N,)
            
        Returns:
            List of heatmaps
        """
        heatmaps = []
        
        for i in range(len(input_batch)):
            heatmap = self.generate_heatmap(
                input_batch[i:i+1],
                target_classes[i]
            )
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def analyze_sample(
        self,
        spectrogram: np.ndarray,
        true_label: int,
        label_names: Dict[int, str],
        get_prediction: bool = True
    ) -> Dict:
        """
        Perform complete Grad-CAM analysis on a single sample.
        
        Args:
            spectrogram: Input spectrogram (H, W, 1) or (H, W)
            true_label: True class label
            label_names: Dictionary mapping class IDs to names
            get_prediction: Whether to get model prediction
            
        Returns:
            Dictionary containing:
                - heatmap: Grad-CAM heatmap
                - prediction: Predicted class ID
                - confidence: Prediction confidence
                - predicted_label: Predicted class name
                - true_label_name: True class name
        """
        # Prepare input tensor
        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=-1)
        
        input_tensor = torch.from_numpy(spectrogram).permute(2, 0, 1).unsqueeze(0).float()
        input_tensor = input_tensor.to(self.device)
        
        # Get prediction if requested
        if get_prediction:
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0, prediction].item()
        else:
            prediction = true_label
            confidence = None
        
        # Generate Grad-CAM heatmap
        heatmap = self.generate_heatmap(input_tensor, prediction)
        
        return {
            'heatmap': heatmap,
            'prediction': prediction,
            'confidence': confidence,
            'predicted_label': label_names[prediction],
            'true_label_name': label_names[true_label]
        }


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram to [0, 1] range for visualization.
    
    Args:
        spec: Input spectrogram
        
    Returns:
        Normalized spectrogram
    """
    spec_norm = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
    return spec_norm


def visualize_gradcam(
    spectrogram: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Grad-CAM Analysis",
    prediction: Optional[str] = None,
    true_label: Optional[str] = None,
    confidence: Optional[float] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (18, 5)
) -> plt.Figure:
    """
    Visualize Grad-CAM heatmap overlaid on spectrogram.
    
    Creates a three-panel figure showing:
    1. Original spectrogram
    2. Grad-CAM heatmap
    3. Overlay of heatmap on spectrogram
    
    Args:
        spectrogram: Input spectrogram (H, W, 1) or (H, W)
        heatmap: Grad-CAM heatmap (H, W)
        title: Main plot title
        prediction: Predicted class name
        true_label: True class name
        confidence: Prediction confidence
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Prepare spectrogram
    if spectrogram.ndim == 3:
        spectrogram = spectrogram.squeeze()
    
    # Normalize spectrogram for visualization
    spec_norm = normalize_spectrogram(spectrogram)
    
    # Convert to RGB for overlay
    spec_rgb = np.stack([spec_norm] * 3, axis=-1)
    
    # Resize heatmap to match spectrogram if needed
    if heatmap.shape != spectrogram.shape:
        heatmap = cv2.resize(heatmap, (spectrogram.shape[1], spectrogram.shape[0]))
    
    # Create overlay
    visualization = show_cam_on_image(spec_rgb, heatmap, use_rgb=True)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original spectrogram
    axes[0].imshow(spec_norm, cmap='viridis', aspect='auto', origin='lower')
    axes[0].set_title('Original Spectrogram', fontsize=12)
    axes[0].set_xlabel('Time', fontsize=10)
    axes[0].set_ylabel('Mel Frequency', fontsize=10)
    
    # Grad-CAM heatmap
    im = axes[1].imshow(heatmap, cmap='jet', aspect='auto', origin='lower')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].set_xlabel('Time', fontsize=10)
    axes[1].set_ylabel('Mel Frequency', fontsize=10)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Activation', fontsize=10)
    
    # Overlay
    axes[2].imshow(visualization, aspect='auto', origin='lower')
    axes[2].set_title('Grad-CAM Overlay', fontsize=12)
    axes[2].set_xlabel('Time', fontsize=10)
    axes[2].set_ylabel('Mel Frequency', fontsize=10)
    
    # Add overall title with prediction info
    if prediction and true_label:
        if confidence is not None:
            suptitle = f"{title}\nTrue: {true_label} | Predicted: {prediction} (confidence: {confidence:.3f})"
        else:
            suptitle = f"{title}\nTrue: {true_label} | Predicted: {prediction}"
    else:
        suptitle = title
    
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig


def compare_gradcam_side_by_side(
    spec1: np.ndarray,
    heatmap1: np.ndarray,
    spec2: np.ndarray,
    heatmap2: np.ndarray,
    title1: str = "Clean",
    title2: str = "Adversarial",
    overall_title: str = "Grad-CAM Comparison",
    pred1: Optional[str] = None,
    pred2: Optional[str] = None,
    true_label: Optional[str] = None,
    conf1: Optional[float] = None,
    conf2: Optional[float] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    Create side-by-side comparison of two Grad-CAM visualizations.
    
    Useful for comparing:
    - Clean vs. adversarial examples
    - Correct vs. incorrect predictions
    - Different model architectures
    
    Args:
        spec1, spec2: Spectrograms to compare
        heatmap1, heatmap2: Corresponding Grad-CAM heatmaps
        title1, title2: Titles for each side
        overall_title: Overall figure title
        pred1, pred2: Predicted class names
        true_label: True class name
        conf1, conf2: Prediction confidences
        save_path: Optional path to save figure
        show_plot: Whether to display the plot
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure object
    """
    # Prepare spectrograms
    if spec1.ndim == 3:
        spec1 = spec1.squeeze()
    if spec2.ndim == 3:
        spec2 = spec2.squeeze()
    
    # Normalize spectrograms
    spec1_norm = normalize_spectrogram(spec1)
    spec2_norm = normalize_spectrogram(spec2)
    
    # Convert to RGB
    spec1_rgb = np.stack([spec1_norm] * 3, axis=-1)
    spec2_rgb = np.stack([spec2_norm] * 3, axis=-1)
    
    # Resize heatmaps if needed
    if heatmap1.shape != spec1.shape:
        heatmap1 = cv2.resize(heatmap1, (spec1.shape[1], spec1.shape[0]))
    if heatmap2.shape != spec2.shape:
        heatmap2 = cv2.resize(heatmap2, (spec2.shape[1], spec2.shape[0]))
    
    # Create overlays
    vis1 = show_cam_on_image(spec1_rgb, heatmap1, use_rgb=True)
    vis2 = show_cam_on_image(spec2_rgb, heatmap2, use_rgb=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # First row: Clean/Original
    axes[0, 0].imshow(spec1_norm, cmap='viridis', aspect='auto', origin='lower')
    axes[0, 0].set_title(f'{title1} - Spectrogram', fontsize=12)
    axes[0, 0].set_ylabel('Mel Frequency', fontsize=10)
    
    im0 = axes[0, 1].imshow(heatmap1, cmap='jet', aspect='auto', origin='lower')
    axes[0, 1].set_title(f'{title1} - Heatmap', fontsize=12)
    cbar0 = plt.colorbar(im0, ax=axes[0, 1])
    cbar0.set_label('Activation', fontsize=10)
    
    axes[0, 2].imshow(vis1, aspect='auto', origin='lower')
    axes[0, 2].set_title(f'{title1} - Overlay', fontsize=12)
    
    # Second row: Adversarial/Comparison
    axes[1, 0].imshow(spec2_norm, cmap='viridis', aspect='auto', origin='lower')
    axes[1, 0].set_title(f'{title2} - Spectrogram', fontsize=12)
    axes[1, 0].set_xlabel('Time', fontsize=10)
    axes[1, 0].set_ylabel('Mel Frequency', fontsize=10)
    
    im1 = axes[1, 1].imshow(heatmap2, cmap='jet', aspect='auto', origin='lower')
    axes[1, 1].set_title(f'{title2} - Heatmap', fontsize=12)
    axes[1, 1].set_xlabel('Time', fontsize=10)
    cbar1 = plt.colorbar(im1, ax=axes[1, 1])
    cbar1.set_label('Activation', fontsize=10)
    
    axes[1, 2].imshow(vis2, aspect='auto', origin='lower')
    axes[1, 2].set_title(f'{title2} - Overlay', fontsize=12)
    axes[1, 2].set_xlabel('Time', fontsize=10)
    
    # Add overall title
    if pred1 and pred2 and true_label:
        suptitle = f"{overall_title}\nTrue: {true_label}\n"
        suptitle += f"{title1}: {pred1}"
        if conf1 is not None:
            suptitle += f" (conf: {conf1:.3f})"
        suptitle += f" | {title2}: {pred2}"
        if conf2 is not None:
            suptitle += f" (conf: {conf2:.3f})"
    else:
        suptitle = overall_title
    
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Show plot
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return fig
