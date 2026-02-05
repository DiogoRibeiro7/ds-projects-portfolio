"""Deep Learning Utilities
=======================

Common utility functions for deep learning experiments, including:
- Data preprocessing and augmentation
- Model utilities and architecture helpers
- Training utilities and callbacks
- Visualization functions
- Metric computation
- Reproducibility helpers
"""

import json
import logging
import os
import pickle
import random
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Data Preprocessing and Augmentation
# ============================================================================


class DataNormalizer:
    """Normalize data for neural networks."""

    def __init__(self, method: str = "standardize"):
        self.method = method
        self.stats = {}

    def fit(self, data: np.ndarray | torch.Tensor) -> "DataNormalizer":
        """Fit normalizer to data."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if self.method == "standardize":
            self.stats["mean"] = np.mean(data, axis=0)
            self.stats["std"] = np.std(data, axis=0) + 1e-8
        elif self.method == "minmax":
            self.stats["min"] = np.min(data, axis=0)
            self.stats["max"] = np.max(data, axis=0)
        elif self.method == "robust":
            self.stats["median"] = np.median(data, axis=0)
            self.stats["mad"] = np.median(np.abs(data - self.stats["median"]), axis=0)

        return self

    def transform(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Transform data."""
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.numpy()

        if self.method == "standardize":
            normalized = (data - self.stats["mean"]) / self.stats["std"]
        elif self.method == "minmax":
            normalized = (data - self.stats["min"]) / (
                self.stats["max"] - self.stats["min"] + 1e-8
            )
        elif self.method == "robust":
            normalized = (data - self.stats["median"]) / (self.stats["mad"] + 1e-8)
        else:
            normalized = data

        return torch.from_numpy(normalized).float() if is_tensor else normalized

    def inverse_transform(
        self, data: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        """Inverse transform data."""
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            data = data.numpy()

        if self.method == "standardize":
            denormalized = data * self.stats["std"] + self.stats["mean"]
        elif self.method == "minmax":
            denormalized = (
                data * (self.stats["max"] - self.stats["min"]) + self.stats["min"]
            )
        elif self.method == "robust":
            denormalized = data * self.stats["mad"] + self.stats["median"]
        else:
            denormalized = data

        return torch.from_numpy(denormalized).float() if is_tensor else denormalized


class AdvancedAugmentation:
    """Advanced data augmentation techniques."""

    @staticmethod
    def mixup(
        images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        batch_size = images.size(0)

        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index]

        return mixed_images, labels, labels[index], lam

    @staticmethod
    def cutmix(
        images: torch.Tensor, labels: torch.Tensor, beta: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        batch_size = images.size(0)

        if beta > 0:
            lam = np.random.beta(beta, beta)
        else:
            lam = 1

        index = torch.randperm(batch_size).to(images.device)

        # Generate random box
        W = images.size(3)
        H = images.size(2)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[
            index, :, bby1:bby2, bbx1:bbx2
        ]

        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return mixed_images, labels, labels[index], lam

    @staticmethod
    def random_erasing(
        images: torch.Tensor,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: list[float] = [0.5, 0.5, 0.5],
    ) -> torch.Tensor:
        """Apply Random Erasing augmentation."""
        if random.random() > probability:
            return images

        for i in range(images.size(0)):
            for attempt in range(100):
                area = images[i].size(1) * images[i].size(2)

                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)

                h = int(round(np.sqrt(target_area * aspect_ratio)))
                w = int(round(np.sqrt(target_area / aspect_ratio)))

                if w < images[i].size(2) and h < images[i].size(1):
                    x1 = random.randint(0, images[i].size(1) - h)
                    y1 = random.randint(0, images[i].size(2) - w)

                    if images[i].size(0) == 3:
                        images[i][0, x1 : x1 + h, y1 : y1 + w] = mean[0]
                        images[i][1, x1 : x1 + h, y1 : y1 + w] = mean[1]
                        images[i][2, x1 : x1 + h, y1 : y1 + w] = mean[2]
                    else:
                        images[i][0, x1 : x1 + h, y1 : y1 + w] = mean[0]
                    break

        return images


# ============================================================================
# Model Utilities
# ============================================================================


class ModelBuilder:
    """Build neural network models dynamically."""

    @staticmethod
    def build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> nn.Module:
        """Build Multi-Layer Perceptron."""
        layers = []
        prev_dim = input_dim

        activation_fn = ModelBuilder._get_activation(activation)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(activation_fn())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    @staticmethod
    def build_cnn(
        input_channels: int,
        num_classes: int,
        conv_channels: list[int] = [32, 64, 128],
        kernel_sizes: list[int] = [3, 3, 3],
        fc_dims: list[int] = [256, 128],
    ) -> nn.Module:
        """Build Convolutional Neural Network."""

        class CNN(nn.Module):
            def __init__(self):
                super().__init__()

                # Convolutional layers
                conv_layers = []
                in_channels = input_channels

                for out_channels, kernel_size in zip(conv_channels, kernel_sizes, strict=False):
                    conv_layers.extend(
                        [
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size,
                                padding=kernel_size // 2,
                            ),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(),
                            nn.MaxPool2d(2),
                        ]
                    )
                    in_channels = out_channels

                self.features = nn.Sequential(*conv_layers)

                # Fully connected layers
                fc_layers = []
                prev_dim = conv_channels[-1]

                for fc_dim in fc_dims:
                    fc_layers.extend(
                        [nn.Linear(prev_dim, fc_dim), nn.ReLU(), nn.Dropout(0.5)]
                    )
                    prev_dim = fc_dim

                fc_layers.append(nn.Linear(prev_dim, num_classes))
                self.classifier = nn.Sequential(*fc_layers)

                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.features(x)
                x = self.adaptive_pool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        return CNN()

    @staticmethod
    def _get_activation(activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "selu": nn.SELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "softplus": nn.Softplus,
        }
        return activations.get(activation.lower(), nn.ReLU)


class ModelUtils:
    """Utility functions for models."""

    @staticmethod
    def count_parameters(model: nn.Module) -> dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable

        return {"total": total, "trainable": trainable, "frozen": frozen}

    @staticmethod
    def freeze_layers(
        model: nn.Module,
        freeze_until: int | None = None,
        freeze_pattern: str | None = None,
    ) -> None:
        """Freeze model layers."""
        if freeze_until is not None:
            for i, (name, param) in enumerate(model.named_parameters()):
                if i < freeze_until:
                    param.requires_grad = False

        if freeze_pattern:
            for name, param in model.named_parameters():
                if freeze_pattern in name:
                    param.requires_grad = False

    @staticmethod
    def unfreeze_layers(
        model: nn.Module,
        unfreeze_from: int | None = None,
        unfreeze_pattern: str | None = None,
    ) -> None:
        """Unfreeze model layers."""
        if unfreeze_from is not None:
            for i, (name, param) in enumerate(model.named_parameters()):
                if i >= unfreeze_from:
                    param.requires_grad = True

        if unfreeze_pattern:
            for name, param in model.named_parameters():
                if unfreeze_pattern in name:
                    param.requires_grad = True

    @staticmethod
    def get_layer_outputs(
        model: nn.Module, input_tensor: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Get outputs from all layers."""
        outputs = {}
        hooks = []

        def hook_fn(module, input, output, name):
            outputs[name] = output.detach()

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)

        model(input_tensor)

        for hook in hooks:
            hook.remove()

        return outputs

    @staticmethod
    def model_summary(model: nn.Module, input_shape: tuple[int, ...]) -> str:
        """Generate model summary."""
        import io
        from contextlib import redirect_stdout

        from torchsummary import summary

        f = io.StringIO()
        with redirect_stdout(f):
            summary(model, input_shape, device="cpu")

        return f.getvalue()


# ============================================================================
# Training Utilities
# ============================================================================


class EarlyStopping:
    """Early stopping callback."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min",
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = None
        self.best_weights = None
        self.counter = 0
        self.stopped = False

    def __call__(self, score: float, model: nn.Module | None = None) -> bool:
        """Check if should stop."""
        if self.best_score is None:
            self.best_score = score
            if model and self.restore_best:
                self.best_weights = model.state_dict().copy()
            return False

        improved = False
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            if model and self.restore_best:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
                if model and self.restore_best and self.best_weights:
                    model.load_state_dict(self.best_weights)
                return True

        return False


class LearningRateScheduler:
    """Custom learning rate schedulers."""

    @staticmethod
    def cosine_annealing_with_warmup(
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Cosine annealing with warmup."""

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr + 0.5 * (1 - min_lr) * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def exponential_warmup(
        optimizer: torch.optim.Optimizer, warmup_epochs: int, gamma: float = 0.95
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Exponential decay with warmup."""

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                return gamma ** (epoch - warmup_epochs)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class TrainingUtils:
    """Training utility functions."""

    @staticmethod
    def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_mixup: bool = False,
        mixup_alpha: float = 1.0,
    ) -> tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            if use_mixup:
                inputs, targets_a, targets_b, lam = AdvancedAugmentation.mixup(
                    inputs, targets, mixup_alpha
                )

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                    outputs, targets_b
                )
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if not use_mixup:
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total if total > 0 else 0

        return avg_loss, accuracy

    @staticmethod
    def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Validate model."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy


# ============================================================================
# Visualization
# ============================================================================


class Visualizer:
    """Visualization utilities for deep learning."""

    @staticmethod
    def plot_training_history(
        history: dict[str, list[float]], save_path: str | None = None
    ) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(history.get("train_loss", []), label="Train Loss")
        axes[0].plot(history.get("val_loss", []), label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(history.get("train_acc", []), label="Train Accuracy")
        axes[1].plot(history.get("val_acc", []), label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def visualize_activations(
        model: nn.Module,
        input_tensor: torch.Tensor,
        layer_name: str,
        save_path: str | None = None,
    ) -> None:
        """Visualize layer activations."""
        activations = {}

        def hook_fn(module, input, output):
            activations["output"] = output.detach()

        # Register hook
        target_layer = None
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found")

        hook = target_layer.register_forward_hook(hook_fn)

        # Forward pass
        model(input_tensor)

        # Remove hook
        hook.remove()

        # Get activations
        activation = activations["output"].squeeze(0).cpu().numpy()

        # Plot activations
        n_features = min(64, activation.shape[0])
        n_cols = 8
        n_rows = n_features // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 1.5))
        axes = axes.flatten()

        for i in range(n_features):
            if len(activation.shape) == 3:  # Conv layer
                axes[i].imshow(activation[i], cmap="viridis")
            else:  # FC layer
                axes[i].bar(range(len(activation[i])), activation[i])
            axes[i].axis("off")

        plt.suptitle(f"Activations from {layer_name}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_grad_flow(model: nn.Module) -> None:
        """Plot gradient flow through the network."""
        ave_grads = []
        max_grads = []
        layers = []

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())

        plt.figure(figsize=(12, 4))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(len(ave_grads)), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
        plt.xlabel("Layers")
        plt.ylabel("Gradient magnitude")
        plt.title("Gradient flow")
        plt.grid(True, alpha=0.3)
        plt.legend(["Zero gradient", "Max gradient", "Mean gradient"])
        plt.tight_layout()
        plt.show()


# ============================================================================
# Metrics
# ============================================================================


class Metrics:
    """Comprehensive metrics for model evaluation."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
        average: str = "weighted",
    ) -> dict[str, float]:
        """Compute classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true, y_pred, average=average, zero_division=0
            ),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }

        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute regression metrics."""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
        }

    @staticmethod
    def per_class_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute per-class accuracy."""
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(per_class_acc))]

        return pd.DataFrame(
            {"Class": class_names, "Accuracy": per_class_acc, "Support": cm.sum(axis=1)}
        )


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    additional_info: dict | None = None,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    return checkpoint


# ============================================================================
# Dataset Utilities
# ============================================================================


class CustomDataset(Dataset):
    """Custom dataset for general use."""

    def __init__(
        self,
        data: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        transform: Callable | None = None,
    ):
        self.data = torch.from_numpy(data) if isinstance(data, np.ndarray) else data
        self.labels = (
            torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


def create_data_loaders(
    data: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    dataset = CustomDataset(data, labels, transform)

    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# Model Export and Deployment
# ============================================================================


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple[int, ...],
    save_path: str,
    opset_version: int = 11,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, *input_shape).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    logger.info(f"Model exported to ONNX: {save_path}")


def export_to_torchscript(
    model: nn.Module,
    input_shape: tuple[int, ...],
    save_path: str,
    method: str = "trace",
) -> None:
    """Export model to TorchScript."""
    model.eval()

    if method == "trace":
        dummy_input = torch.randn(1, *input_shape).to(device)
        traced_model = torch.jit.trace(model, dummy_input)
    else:  # script
        traced_model = torch.jit.script(model)

    traced_model.save(save_path)
    logger.info(f"Model exported to TorchScript: {save_path}")


# ============================================================================
# Profiling and Optimization
# ============================================================================


class ModelProfiler:
    """Profile model performance."""

    @staticmethod
    def profile_model(
        model: nn.Module, input_shape: tuple[int, ...], num_iterations: int = 100
    ) -> dict[str, Any]:
        """Profile model performance."""
        model.eval()
        device = next(model.parameters()).device

        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(device)
        for _ in range(10):
            _ = model(dummy_input)

        # Profile
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)

        torch.cuda.synchronize() if device.type == "cuda" else None
        total_time = time.time() - start_time

        # Calculate metrics
        avg_time = total_time / num_iterations * 1000  # Convert to ms
        throughput = num_iterations / total_time

        # Memory usage
        if device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024  # MB
        else:
            memory_allocated = 0
            memory_reserved = 0

        # Model size
        param_count = ModelUtils.count_parameters(model)
        model_size = (
            param_count["total"] * 4 / 1024 / 1024
        )  # Assuming float32, convert to MB

        return {
            "avg_inference_time_ms": avg_time,
            "throughput_fps": throughput,
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "model_size_mb": model_size,
            "parameter_count": param_count,
        }

    @staticmethod
    def optimize_model(model: nn.Module, optimization_type: str = "prune") -> nn.Module:
        """Optimize model for deployment."""
        if optimization_type == "prune":
            import torch.nn.utils.prune as prune

            # Prune 20% of connections
            for module in model.modules():
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=0.2)
                    prune.remove(module, "weight")

        elif optimization_type == "quantize":
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )

        return model


# ============================================================================
# Utility Functions
# ============================================================================


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def move_to_device(
    data: torch.Tensor | list | dict, device: torch.device
) -> torch.Tensor | list | dict:
    """Recursively move data to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        return data


def save_results(results: dict, save_dir: str, prefix: str = "results") -> None:
    """Save experimental results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    json_path = save_dir / f"{prefix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save as pickle for complex objects
    pkl_path = save_dir / f"{prefix}_{timestamp}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to {save_dir}")


def load_results(file_path: str) -> dict:
    """Load experimental results."""
    file_path = Path(file_path)

    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    # Example: Build and profile a model
    model = ModelBuilder.build_cnn(
        input_channels=3,
        num_classes=10,
        conv_channels=[32, 64, 128],
        fc_dims=[256, 128],
    )

    print("Model Parameters:", ModelUtils.count_parameters(model))

    # Profile model
    profiler = ModelProfiler()
    profile = profiler.profile_model(model, input_shape=(3, 32, 32))
    print("\nModel Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")

    print("\nDeep Learning utilities loaded successfully!")
