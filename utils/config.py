"""
Configuration file for Transformer language model.
Supports GPU (CUDA), Apple Silicon (MPS), and CPU training.
"""

import torch
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

    # Data processing
    vocab_size: int = 10000  # For whitespace and sentencepiece tokenizers (GPT-2 uses 50,257)
    max_seq_length: int = 128  # Shorter sequences for faster training
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05

    # Tokenizer
    min_frequency: int = 2  # Minimum token frequency to include in vocab

    # Training hyperparameters
    batch_size: int = 32  # Adjust based on available GPU memory
    num_epochs: int = 20  # Increased from 10 (transformers need more epochs)
    learning_rate: float = 3e-4  # Reduced from 1e-3 (transformers need lower LR)
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Learning rate warmup
    warmup_steps: int = 1000  # Warmup for first ~1000 steps

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
        """Create directories if they don't exist."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


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
