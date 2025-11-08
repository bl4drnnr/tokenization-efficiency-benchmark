"""
Data preprocessing script: Load data, train tokenizer, and save processed data.
"""

import sys
from pathlib import Path
import torch
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.tokenizer_factory import create_tokenizer, get_tokenizer_name
from utils.dataset import load_text_file, load_jsonl_file, split_data


def preprocess_data(
    input_file: Path,
    tokenizer_type: str = "gpt2",
    file_type: str = "txt",
    text_field: str = "text",
):
    """
    Preprocess data: load, tokenize, and save.

    Args:
        input_file: Path to input data file
        tokenizer_type: Type of tokenizer ('gpt2', 'whitespace', or 'sentencepiece')
        file_type: Type of input file ('txt' or 'jsonl')
        text_field: Field name for JSONL files
    """
    print("=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)

    # Extract dataset name from input file
    dataset_name = Path(input_file).stem  # e.g., "shopping_1_general_corpus"

    # Load configuration (transformer only)
    config = get_config("transformer")
    config.dataset_name = dataset_name
    print(f"\nDataset: {dataset_name}")
    print(f"Tokenizer: {get_tokenizer_name(tokenizer_type)}")
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")

    # Load raw text data
    print(f"\nLoading data from {input_file}...")
    if file_type == "txt":
        texts = load_text_file(input_file)
    elif file_type == "jsonl":
        texts = load_jsonl_file(input_file, text_field=text_field)
    else:
        raise ValueError(f"Unknown file type: {file_type}")

    print(f"Loaded {len(texts)} documents")
    if len(texts) > 0:
        print(f"Sample text: {texts[0][:200]}...")

    # Split data
    print(f"\nSplitting data (train={config.train_split}, val={config.val_split}, test={config.test_split})...")
    train_texts, val_texts, test_texts = split_data(
        texts,
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        test_ratio=config.test_split,
    )
    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")

    # Train tokenizer on training data
    print(f"\nTraining tokenizer (vocab_size={config.vocab_size})...")
    tokenizer = create_tokenizer(tokenizer_type, vocab_size=config.vocab_size)
    tokenizer.train(train_texts)

    # Save tokenizer (include tokenizer type in filename)
    tokenizer_path = config.data_processed_dir / f"{dataset_name}_{tokenizer_type}_tokenizer.json"
    tokenizer.save(tokenizer_path)

    # Tokenize all splits
    print("\nTokenizing data...")

    print("  Tokenizing training set...")
    train_ids = []
    for text in tqdm(train_texts, desc="Train"):
        ids = tokenizer.encode(text, add_special_tokens=True)
        train_ids.append(ids)

    print("  Tokenizing validation set...")
    val_ids = []
    for text in tqdm(val_texts, desc="Val"):
        ids = tokenizer.encode(text, add_special_tokens=True)
        val_ids.append(ids)

    print("  Tokenizing test set...")
    test_ids = []
    for text in tqdm(test_texts, desc="Test"):
        ids = tokenizer.encode(text, add_special_tokens=True)
        test_ids.append(ids)

    # Statistics
    avg_train_len = sum(len(ids) for ids in train_ids) / len(train_ids) if train_ids else 0
    avg_val_len = sum(len(ids) for ids in val_ids) / len(val_ids) if val_ids else 0
    avg_test_len = sum(len(ids) for ids in test_ids) / len(test_ids) if test_ids else 0

    print(f"\nTokenization statistics:")
    print(f"  Train avg length: {avg_train_len:.1f} tokens")
    print(f"  Val avg length: {avg_val_len:.1f} tokens")
    print(f"  Test avg length: {avg_test_len:.1f} tokens")

    # Save tokenized data (include tokenizer type in filename)
    print("\nSaving preprocessed data...")
    torch.save(train_ids, config.data_processed_dir / f"{dataset_name}_{tokenizer_type}_train_ids.pt")
    torch.save(val_ids, config.data_processed_dir / f"{dataset_name}_{tokenizer_type}_val_ids.pt")
    torch.save(test_ids, config.data_processed_dir / f"{dataset_name}_{tokenizer_type}_test_ids.pt")

    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "tokenizer_type": tokenizer_type,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "vocab_size": tokenizer.vocab_size,  # Use tokenizer's actual vocab size
        "avg_train_len": avg_train_len,
        "avg_val_len": avg_val_len,
        "avg_test_len": avg_test_len,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    import json
    with open(config.data_processed_dir / f"{dataset_name}_{tokenizer_type}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nPreprocessing complete!")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Data: {config.data_processed_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for language modeling")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        choices=["gpt2", "whitespace", "sentencepiece"],
        help="Type of tokenizer to use (gpt2, whitespace, or sentencepiece)",
    )
    parser.add_argument(
        "--file-type",
        type=str,
        default="txt",
        choices=["txt", "jsonl"],
        help="Type of input file",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name for JSONL files",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    preprocess_data(
        input_file=input_path,
        tokenizer_type=args.tokenizer,
        file_type=args.file_type,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()
