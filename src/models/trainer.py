"""
Training utilities for genre classification CNN.
"""

import numpy as np
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm


class GenreClassifierTrainer:
    """Handles CNN training for genre classification."""
    
    def __init__(self, model, device=None, results_dir='results'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Device to train on (cuda/cpu)
            results_dir: Directory to save results
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.results_dir = Path(results_dir)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Create directories
        (self.results_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        print(f"Using device: {self.device}")
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=30, batch_size=32, learning_rate=0.0005, 
              model_name='genre_cnn', patience=10,
              weight_decay=1e-4, scheduler_factor=0.3, scheduler_patience=5):
        """
        Train the model.
        
        Args:
            X_train: Training spectrograms (numpy array)
            y_train: Training labels (numpy array)
            X_val: Validation spectrograms
            y_val: Validation labels
            epochs: Maximum number of epochs (default: 30)
            batch_size: Batch size (default: 32)
            learning_rate: Initial learning rate (default: 0.0005)
            model_name: Name for saved models
            patience: Early stopping patience
            weight_decay: L2 regularization weight decay (default: 1e-4)
            scheduler_factor: Factor by which to reduce LR (default: 0.3)
            scheduler_patience: Epochs to wait before reducing LR (default: 5)
            
        Returns:
            Training history
        """
        print(f"\nTraining {model_name}...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Input shape: {X_train.shape[1:]}")
        print('-' * 40)
        
        # Prepare data loaders
        # PyTorch expects (batch, channels, height, width)
        # Our data is (batch, height, width, channels), so transpose
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_val = np.transpose(X_val, (0, 3, 1, 2))
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Train phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/len(train_loader):.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    self.results_dir / 'models' / f'{model_name}_best.pth'
                )
                print(f'  ✓ Best model saved (val_acc: {val_acc:.4f})')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break

            print('-' * 40)
        
        # Save final model
        torch.save(
            self.model.state_dict(),
            self.results_dir / 'models' / f'{model_name}_final.pth'
        )
        print(f"\nModel saved to {self.results_dir / 'models'}")
        
        return self.history
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test spectrograms
            y_test: Test labels
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with metrics
        """
        print("\nEvaluating on test set...")
        
        # Transpose for PyTorch format
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'num_samples': len(X_test)
        }
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        return results
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save figure (optional)
        """
        if not self.history['train_loss']:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot accuracy
        axes[0].plot(epochs, self.history['train_acc'], label='Train', marker='o')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(epochs, self.history['train_loss'], label='Train', marker='o')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_training_metrics(self, results, filename='training_metrics.json'):
        """
        Save training metrics to JSON.
        
        Args:
            results: Dictionary with metrics
            filename: Output filename
        """
        save_path = self.results_dir / 'metrics' / filename
        
        # Add history if available
        if self.history['train_loss']:
            results['history'] = {
                'train_loss': self.history['train_loss'],
                'train_acc': self.history['train_acc'],
                'val_loss': self.history['val_loss'],
                'val_acc': self.history['val_acc']
            }
            results['epochs_trained'] = len(self.history['train_loss'])
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Metrics saved to {save_path}")


def load_processed_data(data_path):
    """
    Load preprocessed spectrogram data.
    
    Args:
        data_path: Path to processed data directory
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, metadata)
    """
    data_path = Path(data_path)
    
    # Load data
    X_train = np.load(data_path / 'train' / 'X.npy')
    y_train = np.load(data_path / 'train' / 'y.npy')
    
    X_val = np.load(data_path / 'val' / 'X.npy')
    y_val = np.load(data_path / 'val' / 'y.npy')
    
    X_test = np.load(data_path / 'test' / 'X.npy')
    y_test = np.load(data_path / 'test' / 'y.npy')
    
    # Load metadata
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded data from {data_path}")
    print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val: {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata
