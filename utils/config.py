"""
Configuration file for Transformer language model.
Supports GPU (CUDA), Apple Silicon (MPS), and CPU training.
"""

import torch
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Global configuration for the project."""

    # Dataset name (will be set from preprocessing/training)
    dataset_name: str = "default"

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    data_wikipedia_dir: Path = project_root / "data" / "wikipedia"
    checkpoints_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    # Device configuration (GPU -> MPS -> CPU hierarchy)
    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Training environment (auto-detect or set via env variable)
    training_env: str = os.environ.get("TRAINING_ENV", "auto")  # "auto", "local", "cloud"

    # DataLoader optimization
    num_workers: int = 0  # Will be set based on environment
    pin_memory: bool = False  # Will be set based on device

    # Data processing
    vocab_size: int = 10000  # For whitespace and sentencepiece tokenizers (GPT-2 uses 50,257)
    max_seq_length: int = 128  # Shorter sequences for faster training
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05

    # Tokenizer
    min_frequency: int = 2  # Minimum token frequency to include in vocab

    # Training hyperparameters
    batch_size: int = 32  # Will be optimized based on GPU
    num_epochs: int = 50  # More epochs for better convergence on large batches
    learning_rate: float = 3e-4  # Will be adjusted for batch size
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Gradient accumulation (for effective larger batch sizes)
    gradient_accumulation_steps: int = 1  # 1 = no accumulation

    # Mixed precision training (for faster GPU training)
    use_amp: bool = False  # Will be set based on device

    # Learning rate warmup
    warmup_steps: int = 2000  # Longer warmup for large batches

    # Label smoothing for better generalization
    label_smoothing: float = 0.1

    # Model saving
    save_every_n_epochs: int = 2

    # Plotting
    plot_every_n_epochs: int = 1  # Generate plots after every N epochs

    # Evaluation
    eval_batch_size: int = 64

    # Generation
    max_gen_length: int = 100
    temperature: float = 1.0
    top_k: int = 50

    def __post_init__(self):
        """Create directories if they don't exist and optimize settings based on device."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect training environment
        if self.training_env == "auto":
            # Check if running in a cloud environment (common indicators)
            is_cloud = (
                os.path.exists("/.dockerenv") or  # Docker container
                os.environ.get("RUNPOD_POD_ID") is not None or  # RunPod
                os.environ.get("COLAB_GPU") is not None or  # Google Colab
                os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None  # Kaggle
            )
            self.training_env = "cloud" if is_cloud else "local"

        # Optimize settings based on device and environment
        if self.device == "cuda":
            self.pin_memory = True
            self.use_amp = True  # Enable mixed precision for NVIDIA GPUs

            if self.training_env == "cloud":
                # Cloud GPU optimization (RunPod, Colab, etc.)
                self.num_workers = 8  # More workers for cloud GPUs
                if self.batch_size <= 512:  # Only increase if using default/small batch
                    self.batch_size = 1024  # RTX 5090 has 32GB VRAM
                    # Scale learning rate with batch size (linear scaling rule)
                    # Base LR is for batch_size=32, so scale proportionally
                    if self.learning_rate == 3e-4:
                        self.learning_rate = 3e-4 * (self.batch_size / 32)
                        self.learning_rate = min(self.learning_rate, 1e-3)  # Cap at 1e-3
            else:
                # Local GPU optimization
                self.num_workers = 4
        elif self.device == "mps":
            # Apple Silicon optimization
            self.pin_memory = False
            self.num_workers = 0  # MPS works better with single process
            self.use_amp = False  # AMP not fully supported on MPS
        else:
            # CPU fallback
            self.pin_memory = False
            self.num_workers = 4
            self.use_amp = False


@dataclass
class TransformerConfig(Config):
    """Configuration specific to Transformer model."""

    # Model architecture
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ff_dim: int = 1024  # Feed-forward dimension
    dropout: float = 0.2  # Increased from 0.1 for better regularization

    # Learning rate scheduler
    scheduler_patience: int = 5  # Increased from 2 for more stability

    # Estimated parameters: ~8-12M

    def __repr__(self):
        return (f"Transformer(layers={self.num_layers}, heads={self.num_heads}, "
                f"emb={self.embedding_dim}, ff={self.ff_dim})")


def get_config(model_type: str = "transformer") -> Config:
    """
    Get configuration for transformer model.

    Args:
        model_type: Should be 'transformer' (kept for compatibility)

    Returns:
        TransformerConfig object
    """
    if model_type.lower() == "transformer":
        return TransformerConfig()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Only 'transformer' is supported.")


if __name__ == "__main__":
    # Test configuration
    transformer_config = get_config("transformer")

    print("Transformer Config:")
    print(f"  Device: {transformer_config.device}")
    print(f"  Vocab size: {transformer_config.vocab_size}")
    print(f"  Model: {transformer_config}")
