"""
Dataset utilities for language modeling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path
import json


class LanguageModelingDataset(Dataset):
    """
    Dataset for causal language modeling.
    Returns input sequences and targets (shifted by 1).
    """

    def __init__(
        self,
        token_ids: List[List[int]],
        max_length: int = 128,
        pad_token_id: int = 0,
    ):
        """
        Initialize dataset.

        Args:
            token_ids: List of tokenized sequences
            max_length: Maximum sequence length
            pad_token_id: ID of padding token
        """
        self.token_ids = token_ids
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single item.

        Returns:
            input_ids: Input token sequence (padded)
            target_ids: Target token sequence (input shifted by 1)
        """
        ids = self.token_ids[idx]

        # Truncate if too long
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]

        # Pad if too short
        if len(ids) < self.max_length:
            ids = ids + [self.pad_token_id] * (self.max_length - len(ids))

        # Convert to tensors
        ids_tensor = torch.tensor(ids, dtype=torch.long)

        # For causal LM: input is ids[:-1], target is ids[1:]
        # But we'll return full sequence and handle shifting in the model
        input_ids = ids_tensor[:-1]  # All tokens except last
        target_ids = ids_tensor[1:]  # All tokens except first

        return input_ids, target_ids


def load_text_file(file_path: Path) -> List[str]:
    """
    Load text from a file.
    Assumes one document per line.

    Args:
        file_path: Path to text file

    Returns:
        List of text strings
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                texts.append(line)
    return texts


def load_jsonl_file(file_path: Path, text_field: str = "text") -> List[str]:
    """
    Load text from a JSONL file.

    Args:
        file_path: Path to JSONL file
        text_field: Name of the field containing text

    Returns:
        List of text strings
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if text_field in data:
                text = data[text_field].strip()
                if text:
                    texts.append(text)
    return texts


def create_dataloaders(
    train_ids: List[List[int]],
    val_ids: List[List[int]],
    test_ids: List[List[int]],
    batch_size: int,
    max_length: int,
    pad_token_id: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.

    Args:
        train_ids: Training token IDs
        val_ids: Validation token IDs
        test_ids: Test token IDs
        batch_size: Batch size
        max_length: Maximum sequence length
        pad_token_id: Padding token ID
        num_workers: Number of worker processes (use 4-8 for GPU training)
        pin_memory: Pin memory for faster GPU transfer (set True for CUDA)

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = LanguageModelingDataset(train_ids, max_length, pad_token_id)
    val_dataset = LanguageModelingDataset(val_ids, max_length, pad_token_id)
    test_dataset = LanguageModelingDataset(test_ids, max_length, pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader


def split_data(
    data: List,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
) -> Tuple[List, List, List]:
    """
    Split data into train/val/test sets.

    Args:
        data: List of data items
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing

    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test dataset
    sample_ids = [
        [1, 2, 3, 4, 5, 6],
        [7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ]

    dataset = LanguageModelingDataset(sample_ids, max_length=10, pad_token_id=0)

    print(f"Dataset size: {len(dataset)}")
    for i in range(len(dataset)):
        inputs, targets = dataset[i]
        print(f"Sample {i}:")
        print(f"  Inputs:  {inputs.tolist()}")
        print(f"  Targets: {targets.tolist()}")
