"""
Evaluation metrics for language models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value
    """
    return np.exp(loss)


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    pad_token_id: Optional[int] = None,
) -> tuple[float, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: The language model
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run on
        pad_token_id: ID of padding token (to ignore in loss)

    Returns:
        avg_loss, perplexity
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Reshape for loss calculation
            # outputs: (batch_size, seq_len, vocab_size)
            # targets: (batch_size, seq_len)
            batch_size, seq_len, vocab_size = outputs.shape

            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            # Calculate loss
            if pad_token_id is not None:
                # Ignore padding tokens in loss
                mask = (targets_flat != pad_token_id)
                loss = criterion(outputs_flat[mask], targets_flat[mask])
                num_tokens = mask.sum().item()
            else:
                loss = criterion(outputs_flat, targets_flat)
                num_tokens = targets_flat.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = calculate_perplexity(avg_loss)

    return avg_loss, perplexity


def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_token_id: Optional[int] = None,
) -> float:
    """
    Calculate token-level accuracy.

    Args:
        logits: Model predictions (batch_size, seq_len, vocab_size)
        targets: True targets (batch_size, seq_len)
        pad_token_id: ID of padding token (to ignore)

    Returns:
        Accuracy as a percentage
    """
    predictions = torch.argmax(logits, dim=-1)

    if pad_token_id is not None:
        mask = (targets != pad_token_id)
        correct = ((predictions == targets) & mask).sum().item()
        total = mask.sum().item()
    else:
        correct = (predictions == targets).sum().item()
        total = targets.numel()

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy


class MetricsTracker:
    """
    Track and store metrics during training.
    """

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
        self.learning_rates = []
        self.epoch_times = []

    def update(
        self,
        train_loss: float,
        val_loss: float,
        lr: float,
        epoch_time: float,
    ):
        """Update metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_perplexities.append(calculate_perplexity(train_loss))
        self.val_perplexities.append(calculate_perplexity(val_loss))
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_perplexities": self.train_perplexities,
            "val_perplexities": self.val_perplexities,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "total_time": sum(self.epoch_times),
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
            "best_val_loss": min(self.val_losses) if self.val_losses else float('inf'),
            "best_val_perplexity": min(self.val_perplexities) if self.val_perplexities else float('inf'),
        }

    def save(self, path: str):
        """Save metrics to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        print(f"Metrics saved to {path}")

    def load(self, path: str):
        """Load metrics from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.train_losses = data.get("train_losses", [])
            self.val_losses = data.get("val_losses", [])
            self.train_perplexities = data.get("train_perplexities", [])
            self.val_perplexities = data.get("val_perplexities", [])
            self.learning_rates = data.get("learning_rates", [])
            self.epoch_times = data.get("epoch_times", [])
        print(f"Metrics loaded from {path}")


if __name__ == "__main__":
    # Test perplexity calculation
    losses = [2.5, 2.0, 1.5, 1.0, 0.5]
    print("Loss -> Perplexity:")
    for loss in losses:
        ppl = calculate_perplexity(loss)
        print(f"  {loss:.2f} -> {ppl:.2f}")
