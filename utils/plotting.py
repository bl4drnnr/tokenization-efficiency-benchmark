"""
Plotting utilities for visualizing training metrics.
Creates and saves plots during training to track progress.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional
import json


class TrainingPlotter:
    """
    Creates and saves training visualization plots.
    Tracks metrics across epochs and generates plots periodically.
    """

    def __init__(self, save_dir: Path, model_name: str = "model"):
        """
        Initialize plotter.

        Args:
            save_dir: Directory to save plots
            model_name: Name of model (for plot titles and filenames)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # Create subdirectory for plots
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training plots will be saved to: {self.plots_dir}")

    def plot_losses(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Plot training and validation losses.

        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            epoch: Current epoch number
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.model_name} - Training and Validation Loss (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_loss_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_perplexity(
        self,
        train_perplexities: List[float],
        val_perplexities: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Plot training and validation perplexities.

        Args:
            train_perplexities: List of training perplexities per epoch
            val_perplexities: List of validation perplexities per epoch
            epoch: Current epoch number
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_perplexities) + 1)

        plt.plot(epochs, train_perplexities, 'b-o', label='Train PPL', linewidth=2, markersize=6)
        plt.plot(epochs, val_perplexities, 'r-s', label='Val PPL', linewidth=2, markersize=6)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Perplexity', fontsize=12)
        plt.title(f'{self.model_name} - Training and Validation Perplexity (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_perplexity_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_learning_rate(
        self,
        learning_rates: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Plot learning rate schedule.

        Args:
            learning_rates: List of learning rates per epoch
            epoch: Current epoch number
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(learning_rates) + 1)

        plt.plot(epochs, learning_rates, 'g-o', linewidth=2, markersize=6)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'{self.model_name} - Learning Rate Schedule (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_learning_rate_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_epoch_times(
        self,
        epoch_times: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Plot training time per epoch.

        Args:
            epoch_times: List of epoch training times
            epoch: Current epoch number
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(epoch_times) + 1)

        plt.bar(epochs, epoch_times, color='skyblue', edgecolor='navy', alpha=0.7)

        # Add average line
        avg_time = np.mean(epoch_times)
        plt.axhline(y=avg_time, color='r', linestyle='--', linewidth=2, label=f'Avg: {avg_time:.1f}s')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.title(f'{self.model_name} - Training Time per Epoch (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_epoch_times_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_combined_metrics(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_perplexities: List[float],
        val_perplexities: List[float],
        learning_rates: List[float],
        epoch_times: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Create a combined plot with all metrics in subplots.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            train_perplexities: List of training perplexities
            val_perplexities: List of validation perplexities
            learning_rates: List of learning rates
            epoch_times: List of epoch times
            epoch: Current epoch number
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.model_name} - Training Metrics (Epoch {epoch})', fontsize=16, fontweight='bold')

        epochs = range(1, len(train_losses) + 1)

        # Plot 1: Losses
        axes[0, 0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=5)
        axes[0, 0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=5)
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Perplexity
        axes[0, 1].plot(epochs, train_perplexities, 'b-o', label='Train PPL', linewidth=2, markersize=5)
        axes[0, 1].plot(epochs, val_perplexities, 'r-s', label='Val PPL', linewidth=2, markersize=5)
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Perplexity', fontsize=11)
        axes[0, 1].set_title('Training and Validation Perplexity', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Learning Rate
        axes[1, 0].plot(epochs, learning_rates, 'g-o', linewidth=2, markersize=5)
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Epoch Times
        axes[1, 1].bar(epochs, epoch_times, color='skyblue', edgecolor='navy', alpha=0.7)
        avg_time = np.mean(epoch_times)
        axes[1, 1].axhline(y=avg_time, color='r', linestyle='--', linewidth=2, label=f'Avg: {avg_time:.1f}s')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Time (seconds)', fontsize=11)
        axes[1, 1].set_title('Training Time per Epoch', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_combined_metrics_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_loss_comparison(
        self,
        train_losses: List[float],
        val_losses: List[float],
        epoch: int,
        save: bool = True,
    ):
        """
        Plot train vs validation loss to detect overfitting.

        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            epoch: Current epoch number
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)

        # Calculate gap between train and val
        gap = [val - train for train, val in zip(train_losses, val_losses)]

        plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        plt.fill_between(epochs, train_losses, val_losses, alpha=0.2, color='orange', label='Generalization Gap')

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'{self.model_name} - Overfitting Analysis (Epoch {epoch})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = self.plots_dir / f"{self.model_name}_overfitting_analysis_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_all_metrics(
        self,
        metrics_tracker,
        epoch: int,
        plot_frequency: int = 1,
    ):
        """
        Plot all metrics from MetricsTracker.
        This is the main method to call during training.

        Args:
            metrics_tracker: MetricsTracker instance with training data
            epoch: Current epoch number
            plot_frequency: How often to save plots (every N epochs)
        """
        # Only plot if it's time to plot
        if (epoch + 1) % plot_frequency != 0:
            return

        print(f"\nGenerating training plots (Epoch {epoch + 1})...")

        # Extract metrics
        train_losses = metrics_tracker.train_losses
        val_losses = metrics_tracker.val_losses
        train_ppls = metrics_tracker.train_perplexities
        val_ppls = metrics_tracker.val_perplexities
        lrs = metrics_tracker.learning_rates
        times = metrics_tracker.epoch_times

        # Create individual plots
        self.plot_losses(train_losses, val_losses, epoch + 1)
        self.plot_perplexity(train_ppls, val_ppls, epoch + 1)
        self.plot_learning_rate(lrs, epoch + 1)
        self.plot_epoch_times(times, epoch + 1)

        # Create combined plot
        self.plot_combined_metrics(
            train_losses, val_losses,
            train_ppls, val_ppls,
            lrs, times,
            epoch + 1
        )

        # Create overfitting analysis
        self.plot_loss_comparison(train_losses, val_losses, epoch + 1)

        print(f"Plots saved to: {self.plots_dir}")

    def create_final_summary_plot(self, metrics_path: Path):
        """
        Create a final comprehensive summary plot after training completes.

        Args:
            metrics_path: Path to saved metrics JSON file
        """
        print(f"\nCreating final summary plot from metrics: {metrics_path}")

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        train_losses = metrics['train_losses']
        val_losses = metrics['val_losses']
        train_ppls = metrics['train_perplexities']
        val_ppls = metrics['val_perplexities']
        lrs = metrics['learning_rates']
        times = metrics['epoch_times']

        # Create comprehensive final plot
        final_epoch = len(train_losses)
        self.plot_combined_metrics(
            train_losses, val_losses,
            train_ppls, val_ppls,
            lrs, times,
            final_epoch,
            save=True
        )

        # Also create a summary stats plot
        self._plot_summary_stats(metrics)

        print(f"Final summary plots saved to: {self.plots_dir}")

    def _plot_summary_stats(self, metrics: dict):
        """
        Create a text-based summary statistics plot.

        Args:
            metrics: Dictionary of metrics
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')

        summary_text = f"""
        TRAINING SUMMARY - {self.model_name.upper()}
        {'=' * 60}

        Best Performance:
        • Best Validation Loss:       {metrics['best_val_loss']:.4f}
        • Best Validation Perplexity:  {metrics['best_val_perplexity']:.2f}

        Training Statistics:
        • Total Epochs:                {len(metrics['train_losses'])}
        • Total Training Time:         {metrics['total_time']:.1f}s ({metrics['total_time']/60:.1f}m)
        • Average Epoch Time:          {metrics['avg_epoch_time']:.1f}s

        Final Metrics:
        • Final Train Loss:            {metrics['train_losses'][-1]:.4f}
        • Final Val Loss:              {metrics['val_losses'][-1]:.4f}
        • Final Train Perplexity:      {metrics['train_perplexities'][-1]:.2f}
        • Final Val Perplexity:        {metrics['val_perplexities'][-1]:.2f}
        • Final Learning Rate:         {metrics['learning_rates'][-1]:.6f}

        Improvement:
        • Loss Reduction:              {metrics['train_losses'][0] - metrics['train_losses'][-1]:.4f}
        • Perplexity Reduction:        {metrics['train_perplexities'][0] - metrics['train_perplexities'][-1]:.2f}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=ax.transAxes)

        plt.tight_layout()
        save_path = self.plots_dir / f"{self.model_name}_summary_stats.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Test the plotter with dummy data
    print("Testing TrainingPlotter...")

    plotter = TrainingPlotter(save_dir=Path("test_plots"), model_name="TestModel")

    # Create dummy data
    epochs = 10
    train_losses = [2.5 - 0.2 * i + np.random.random() * 0.1 for i in range(epochs)]
    val_losses = [2.6 - 0.18 * i + np.random.random() * 0.15 for i in range(epochs)]
    train_ppls = [np.exp(loss) for loss in train_losses]
    val_ppls = [np.exp(loss) for loss in val_losses]
    lrs = [0.001 * (0.5 ** (i // 3)) for i in range(epochs)]
    times = [120 + np.random.random() * 20 for _ in range(epochs)]

    # Create combined plot
    plotter.plot_combined_metrics(
        train_losses, val_losses,
        train_ppls, val_ppls,
        lrs, times,
        epoch=epochs,
        save=True
    )

    print(f"Test plots created in: {plotter.plots_dir}")
