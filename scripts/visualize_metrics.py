"""
Visualize training metrics from saved JSON files.
Useful for creating plots after training or comparing different runs.
"""

import sys
from pathlib import Path
import argparse
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.plotting import TrainingPlotter


def visualize_metrics(metrics_path: str, model_name: str = None):
    """
    Create plots from saved metrics JSON file.

    Args:
        metrics_path: Path to metrics JSON file
        model_name: Optional name override for plots
    """
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return

    # Determine model name from filename if not provided
    if model_name is None:
        model_name = metrics_path.stem.replace('_metrics', '')

    print(f"Visualizing metrics from: {metrics_path}")
    print(f"Model name: {model_name}")

    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create plotter
    save_dir = metrics_path.parent
    plotter = TrainingPlotter(save_dir=save_dir, model_name=model_name)

    # Extract data
    train_losses = metrics['train_losses']
    val_losses = metrics['val_losses']
    train_ppls = metrics['train_perplexities']
    val_ppls = metrics['val_perplexities']
    lrs = metrics['learning_rates']
    times = metrics['epoch_times']
    final_epoch = len(train_losses)

    print(f"\nGenerating plots for {final_epoch} epochs...")

    # Create all plots
    plotter.plot_losses(train_losses, val_losses, final_epoch)
    plotter.plot_perplexity(train_ppls, val_ppls, final_epoch)
    plotter.plot_learning_rate(lrs, final_epoch)
    plotter.plot_epoch_times(times, final_epoch)
    plotter.plot_combined_metrics(
        train_losses, val_losses,
        train_ppls, val_ppls,
        lrs, times,
        final_epoch
    )
    plotter.plot_loss_comparison(train_losses, val_losses, final_epoch)

    # Create summary plot
    plotter._plot_summary_stats(metrics)

    print(f"\nPlots saved to: {plotter.plots_dir}")
    print("\nGenerated plots:")
    print(f"  - {model_name}_loss_epoch_{final_epoch}.png")
    print(f"  - {model_name}_perplexity_epoch_{final_epoch}.png")
    print(f"  - {model_name}_learning_rate_epoch_{final_epoch}.png")
    print(f"  - {model_name}_epoch_times_epoch_{final_epoch}.png")
    print(f"  - {model_name}_combined_metrics_epoch_{final_epoch}.png")
    print(f"  - {model_name}_overfitting_analysis_epoch_{final_epoch}.png")
    print(f"  - {model_name}_summary_stats.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics for Transformer model")
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to metrics JSON file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for plot titles (optional)",
    )

    args = parser.parse_args()

    visualize_metrics(args.metrics, args.model_name)


if __name__ == "__main__":
    main()
