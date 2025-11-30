"""
Adversarial attack implementations using Adversarial Robustness Toolbox (ART).
"""

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier


def prepare_classifier(model, device=None, input_shape=(1, 128, 130), num_classes=10):
    """
    Wrap PyTorch model with ART's PyTorchClassifier.
    
    Args:
        model: Trained PyTorch model
        device: Device to run on (cuda/cpu). If None, auto-detects.
        input_shape: Shape of input spectrograms (channels, height, width)
        num_classes: Number of output classes
        
    Returns:
        ART PyTorchClassifier wrapper
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Define loss function and optimizer (required by ART)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Wrap model with ART classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=num_classes,
        clip_values=(0.0, 1.0)  # Assuming normalized spectrograms
    )
    
    return classifier


def generate_fgsm_attack(classifier, X, y, eps=0.1):
    """
    Generate adversarial examples using Fast Gradient Sign Method (FGSM).
    
    FGSM is a simple one-step attack that perturbs inputs in the direction
    of the gradient of the loss with respect to the input.
    
    Args:
        classifier: ART PyTorchClassifier
        X: Input spectrograms (batch, height, width, channels) - numpy array
        y: True labels (batch,) - numpy array
        eps: Maximum perturbation magnitude (L-infinity norm)
        
    Returns:
        X_adv: Adversarial examples with same shape as X
    """
    # Transpose to PyTorch format: (batch, channels, height, width)
    X_torch = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    
    # Create FGSM attack
    attack = FastGradientMethod(estimator=classifier, eps=eps)
    
    # Generate adversarial examples
    X_adv = attack.generate(x=X_torch, y=y)
    
    # Transpose back to original format: (batch, height, width, channels)
    X_adv = np.transpose(X_adv, (0, 2, 3, 1))
    
    return X_adv


def generate_pgd_attack(classifier, X, y, eps=0.1, eps_step=0.01, max_iter=40):
    """
    Generate adversarial examples using Projected Gradient Descent (PGD).
    
    PGD is an iterative attack that applies multiple small FGSM-like steps,
    projecting back into the epsilon ball after each step. More powerful than FGSM.
    
    Args:
        classifier: ART PyTorchClassifier
        X: Input spectrograms (batch, height, width, channels) - numpy array
        y: True labels (batch,) - numpy array
        eps: Maximum perturbation magnitude (L-infinity norm)
        eps_step: Step size for each iteration
        max_iter: Maximum number of iterations
        
    Returns:
        X_adv: Adversarial examples with same shape as X
    """
    # Transpose to PyTorch format: (batch, channels, height, width)
    X_torch = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
    
    # Create PGD attack
    attack = ProjectedGradientDescent(
        estimator=classifier,
        eps=eps,
        eps_step=eps_step,
        max_iter=max_iter,
        targeted=False,
        num_random_init=0  # Start from original sample
    )
    
    # Generate adversarial examples
    X_adv = attack.generate(x=X_torch, y=y)
    
    # Transpose back to original format: (batch, height, width, channels)
    X_adv = np.transpose(X_adv, (0, 2, 3, 1))
    
    return X_adv


def evaluate_robustness(classifier, X, y, attack_fn, attack_params):
    """
    Evaluate model robustness by generating adversarial examples and measuring accuracy drop.
    
    Args:
        classifier: ART PyTorchClassifier
        X: Clean input spectrograms (batch, height, width, channels)
        y: True labels (batch,)
        attack_fn: Attack function (generate_fgsm_attack or generate_pgd_attack)
        attack_params: Dictionary of parameters to pass to attack_fn
        
    Returns:
        results: Dictionary containing:
            - clean_accuracy: Accuracy on clean samples
            - adversarial_accuracy: Accuracy on adversarial samples
            - accuracy_drop: Difference between clean and adversarial accuracy
            - attack_success_rate: Percentage of successful attacks
            - mean_perturbation: Average L2 perturbation magnitude
            - max_perturbation: Maximum L2 perturbation magnitude
            - attack_params: Parameters used for the attack
    """
    try:
        # Transpose to PyTorch format for evaluation
        X_torch = np.transpose(X, (0, 3, 1, 2)).astype(np.float32)
        
        # Evaluate on clean data
        preds_clean = classifier.predict(X_torch)
        preds_clean_labels = np.argmax(preds_clean, axis=1)
        clean_acc = np.mean(preds_clean_labels == y)
        
        # Generate adversarial examples
        X_adv = attack_fn(classifier, X, y, **attack_params)
        
        # Transpose adversarial examples to PyTorch format
        X_adv_torch = np.transpose(X_adv, (0, 3, 1, 2)).astype(np.float32)
        
        # Evaluate on adversarial data
        preds_adv = classifier.predict(X_adv_torch)
        preds_adv_labels = np.argmax(preds_adv, axis=1)
        adv_acc = np.mean(preds_adv_labels == y)
        
        # Calculate perturbation statistics
        perturbation = np.abs(X_adv - X)
        l2_perturbation = np.sqrt(np.sum(perturbation ** 2, axis=(1, 2, 3)))
        
        # Compile results
        results = {
            'clean_accuracy': float(clean_acc),
            'adversarial_accuracy': float(adv_acc),
            'accuracy_drop': float(clean_acc - adv_acc),
            'attack_success_rate': float((clean_acc - adv_acc) / clean_acc) if clean_acc > 0 else 0.0,
            'mean_perturbation': float(np.mean(l2_perturbation)),
            'max_perturbation': float(np.max(l2_perturbation)),
            'mean_linf_perturbation': float(np.mean(perturbation)),
            'max_linf_perturbation': float(np.max(perturbation)),
            'num_samples': len(X),
            'attack_params': attack_params
        }
        
        return results, X_adv
        
    except Exception as e:
        raise RuntimeError(f"Error during robustness evaluation: {str(e)}") from e
