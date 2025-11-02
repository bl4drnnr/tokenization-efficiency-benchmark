"""
Training script for Transformer language model.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import time
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config, TransformerConfig
from utils.dataset import create_dataloaders
from utils.metrics import MetricsTracker, evaluate_model, calculate_perplexity
from utils.plotting import TrainingPlotter
from models.transformer_model import TransformerLanguageModel


class WarmupLRScheduler:
    """
    Learning rate scheduler with linear warmup followed by ReduceLROnPlateau.
    """
    def __init__(self, optimizer, warmup_steps, base_lr, plateau_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.plateau_scheduler = plateau_scheduler
        self.current_step = 0
        self.finished_warmup = False

    def step(self, val_loss=None):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif not self.finished_warmup:
            # End of warmup, set to base LR
            self.finished_warmup = True
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr
        elif val_loss is not None:
            # After warmup, use plateau scheduler
            self.plateau_scheduler.step(val_loss)

    def get_last_lr(self):
        """Get current learning rate."""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    pad_token_id: int,
    gradient_clip: float = 1.0,
    scheduler=None,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss (ignore padding tokens)
        batch_size, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        mask = (targets_flat != pad_token_id)
        loss = criterion(outputs_flat[mask], targets_flat[mask])

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Update weights
        optimizer.step()

        # Update learning rate (for warmup)
        if scheduler is not None:
            scheduler.step()

        # Track statistics
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{current_lr:.2e}"
        })

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss


def train(
    dataset: str = None,
    resume_from: str = None,
):
    """
    Main training function.

    Args:
        dataset: Dataset name (e.g., 'shopping_1_general_corpus')
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 80)
    print("TRAINING TRANSFORMER LANGUAGE MODEL")
    print("=" * 80)

    # Load configuration
    config = get_config("transformer")

    # Auto-detect dataset if not specified
    if dataset is None:
        # Look for metadata files in processed dir
        metadata_files = list(config.data_processed_dir.glob("*_metadata.json"))
        if not metadata_files:
            raise ValueError("No preprocessed dataset found. Please run preprocess_data.py first.")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple datasets found: {[f.stem.replace('_metadata', '') for f in metadata_files]}. Please specify --dataset")
        dataset = metadata_files[0].stem.replace('_metadata', '')
        print(f"\nAuto-detected dataset: {dataset}")

    config.dataset_name = dataset
    print(f"Dataset: {dataset}")
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    train_ids = torch.load(config.data_processed_dir / f"{dataset}_train_ids.pt")
    val_ids = torch.load(config.data_processed_dir / f"{dataset}_val_ids.pt")
    test_ids = torch.load(config.data_processed_dir / f"{dataset}_test_ids.pt")

    with open(config.data_processed_dir / f"{dataset}_metadata.json", "r") as f:
        metadata = json.load(f)
        pad_token_id = metadata["pad_token_id"]

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ids,
        val_ids,
        test_ids,
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        pad_token_id=pad_token_id,
        num_workers=0,  # Mac-friendly
    )

    # Create model
    print("\nInitializing model...")
    model = TransformerLanguageModel(
        vocab_size=config.vocab_size,
        d_model=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.ff_dim,
        max_seq_len=config.max_seq_length,
        dropout=config.dropout,
        pad_token_id=pad_token_id,
    )

    model = model.to(config.device)
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Loss with label smoothing and optimizer
    criterion = nn.CrossEntropyLoss(
        reduction='mean',
        label_smoothing=config.label_smoothing
    )
    print(f"Using label smoothing: {config.label_smoothing}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler with warmup
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.scheduler_patience,
    )

    # Wrap with warmup scheduler
    scheduler = WarmupLRScheduler(
        optimizer=optimizer,
        warmup_steps=config.warmup_steps,
        base_lr=config.learning_rate,
        plateau_scheduler=plateau_scheduler,
    )

    print(f"Learning rate warmup: {config.warmup_steps} steps")

    # Metrics tracker
    metrics = MetricsTracker()

    # Initialize plotter with dataset-specific model name
    model_name_with_dataset = f"{dataset}_transformer"
    plotter = TrainingPlotter(save_dir=config.results_dir, model_name=model_name_with_dataset)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    total_training_time = 0.0

    for epoch in range(start_epoch, config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 80)

        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
            pad_token_id,
            config.gradient_clip,
            scheduler=scheduler,  # Pass scheduler for warmup
        )

        # Validate
        print("Validating...")
        val_loss, val_ppl = evaluate_model(
            model,
            val_loader,
            criterion,
            config.device,
            pad_token_id,
        )

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        # Update learning rate based on validation loss (for plateau scheduler)
        scheduler.step(val_loss=val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Track metrics
        metrics.update(train_loss, val_loss, current_lr, epoch_time)

        # Print epoch summary
        train_ppl = calculate_perplexity(train_loss)
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or val_loss < best_val_loss:
            checkpoint_path = config.checkpoints_dir / f"{dataset}_transformer_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = config.checkpoints_dir / f"{dataset}_transformer_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
            }, best_model_path)
            print(f"  Best model saved: {best_model_path}")

        # Generate and save plots
        plotter.plot_all_metrics(metrics, epoch, plot_frequency=config.plot_every_n_epochs)

    # Training complete
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total training time: {total_training_time:.1f}s ({total_training_time / 60:.1f}m)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {calculate_perplexity(best_val_loss):.2f}")

    # Save metrics
    metrics_path = config.results_dir / f"{dataset}_transformer_metrics.json"
    metrics.save(str(metrics_path))

    # Create final summary plot
    plotter.create_final_summary_plot(metrics_path)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Training plots saved to: {plotter.plots_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train transformer language model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., 'shopping_1_general_corpus'). Auto-detected if only one dataset exists.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    train(dataset=args.dataset, resume_from=args.resume)


if __name__ == "__main__":
    main()
