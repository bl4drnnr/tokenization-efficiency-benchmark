"""
Evaluation script for trained transformer language model.
Evaluates on test data and calculates perplexity.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.dataset import create_dataloaders
from utils.metrics import evaluate_model
from models.transformer_model import TransformerLanguageModel


def evaluate(
    checkpoint_path: str,
    data_type: str = "test",
    dataset: str = None,
    eval_dataset: str = None,
):
    """
    Evaluate a trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        data_type: Type of data to evaluate on ('test', 'val', or 'out-of-domain')
        dataset: Dataset name for model (auto-detected if not specified)
        eval_dataset: Dataset name for out-of-domain evaluation (e.g., 'plwiki')
    """
    print("=" * 80)
    print("EVALUATING TRANSFORMER LANGUAGE MODEL")
    print("=" * 80)

    # Load configuration
    config = get_config("transformer")

    # Auto-detect dataset if not specified
    if dataset is None:
        metadata_files = list(config.data_processed_dir.glob("*_metadata.json"))
        if not metadata_files:
            raise ValueError("No preprocessed dataset found.")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple datasets found. Please specify --dataset")
        dataset = metadata_files[0].stem.replace('_metadata', '')
        print(f"\nAuto-detected dataset: {dataset}")

    config.dataset_name = dataset
    print(f"Dataset: {dataset}")
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")

    # Load metadata
    with open(config.data_processed_dir / f"{dataset}_metadata.json", "r") as f:
        metadata = json.load(f)
        pad_token_id = metadata["pad_token_id"]

    # Load data based on type
    if data_type == "test":
        print(f"\nLoading test data from {dataset}...")
        eval_ids = torch.load(config.data_processed_dir / f"{dataset}_test_ids.pt")
        eval_dataset_name = dataset
    elif data_type == "val":
        print(f"\nLoading validation data from {dataset}...")
        eval_ids = torch.load(config.data_processed_dir / f"{dataset}_val_ids.pt")
        eval_dataset_name = dataset
    elif data_type == "out-of-domain":
        if eval_dataset is None:
            raise ValueError("--eval-dataset must be specified when using --data out-of-domain")
        print(f"\nLoading out-of-domain data from {eval_dataset}...")

        # Try to load test set from the out-of-domain dataset
        ood_test_path = config.data_processed_dir / f"{eval_dataset}_test_ids.pt"
        if not ood_test_path.exists():
            print(f"Error: Out-of-domain dataset not found at {ood_test_path}")
            print(f"Available datasets in {config.data_processed_dir}:")
            metadata_files = list(config.data_processed_dir.glob("*_metadata.json"))
            for mf in metadata_files:
                print(f"  - {mf.stem.replace('_metadata', '')}")
            return
        eval_ids = torch.load(ood_test_path)
        eval_dataset_name = eval_dataset
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'test', 'val', or 'out-of-domain'")

    print(f"Loaded {len(eval_ids)} sequences")

    # Create dataloader
    from torch.utils.data import DataLoader
    from utils.dataset import LanguageModelingDataset

    eval_dataset = LanguageModelingDataset(
        eval_ids,
        max_length=config.max_seq_length,
        pad_token_id=pad_token_id,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # Create model
    print("Initializing model...")
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Evaluate
    print(f"\nEvaluating on {data_type} set...")
    criterion = nn.CrossEntropyLoss(reduction='mean')

    start_time = time.time()
    loss, perplexity = evaluate_model(
        model,
        eval_loader,
        criterion,
        config.device,
        pad_token_id,
    )
    eval_time = time.time() - start_time

    # Calculate tokens per second
    total_tokens = sum(len(seq) for seq in eval_ids)
    tokens_per_second = total_tokens / eval_time

    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: TRANSFORMER")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_type}")
    print(f"Sequences: {len(eval_ids)}")
    print(f"Total tokens: {total_tokens:,}")
    print("-" * 80)
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Evaluation time: {eval_time:.2f}s")
    print(f"Throughput: {tokens_per_second:.0f} tokens/s")
    print("=" * 80)

    # Save results
    results = {
        "model_type": "transformer",
        "model_dataset": dataset,
        "eval_dataset": eval_dataset_name,
        "checkpoint": str(checkpoint_path),
        "data_type": data_type,
        "num_sequences": len(eval_ids),
        "total_tokens": total_tokens,
        "loss": loss,
        "perplexity": perplexity,
        "eval_time": eval_time,
        "tokens_per_second": tokens_per_second,
    }

    # Create results filename
    if data_type == "out-of-domain":
        results_filename = f"{dataset}_transformer_eval_ood_{eval_dataset_name}.json"
    else:
        results_filename = f"{dataset}_transformer_eval_{data_type}.json"

    results_path = config.results_dir / results_filename
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer language model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="test",
        choices=["test", "val", "out-of-domain"],
        help="Data to evaluate on: 'test' (in-domain), 'val' (validation), 'out-of-domain' (requires --eval-dataset)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Model's training dataset name. Auto-detected if only one dataset exists.",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="Dataset name for out-of-domain evaluation (e.g., 'plwiki'). Required when --data is 'out-of-domain'.",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    evaluate(str(checkpoint_path), args.data, args.dataset, args.eval_dataset)


if __name__ == "__main__":
    main()
