"""
Tokenizer for Polish text using BPE (Byte-Pair Encoding).
"""

import json
from pathlib import Path
from typing import List, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


class PolishTokenizer:
    """
    Custom tokenizer for Polish text using BPE.
    """

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Tokenizer] = None

        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]

    def train(self, texts: List[str]) -> None:
        """
        Train tokenizer on provided texts.

        Args:
            texts: List of text strings to train on
        """
        # Create a BPE tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))

        # Set up normalizer (lowercase and strip accents for Polish)
        self.tokenizer.normalizer = Sequence([NFD(), Lowercase()])

        # Set up pre-tokenizer (split on whitespace)
        self.tokenizer.pre_tokenizer = Whitespace()

        # Create trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True,
        )

        # Train the tokenizer
        print(f"Training tokenizer on {len(texts)} texts...")
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

        # Enable padding
        self.tokenizer.enable_padding(pad_token=self.pad_token)

        # Enable truncation
        self.tokenizer.enable_truncation(max_length=512)

        print(f"Tokenizer trained! Vocabulary size: {self.tokenizer.get_vocab_size()}")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        if add_special_tokens:
            bos_id = self.tokenizer.token_to_id(self.bos_token)
            eos_id = self.tokenizer.token_to_id(self.eos_token)
            ids = [bos_id] + ids + [eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Path) -> None:
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")

        path = Path(path)
        self.tokenizer.save(str(path))
        print(f"Tokenizer saved to {path}")

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path) -> None:
        """
        Load tokenizer from file.

        Args:
            path: Path to load tokenizer from
        """
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path))
        print(f"Tokenizer loaded from {path}")

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.vocab_size = metadata["vocab_size"]
                self.special_tokens = metadata["special_tokens"]

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet!")
        return self.tokenizer.token_to_id(self.eos_token)


if __name__ == "__main__":
    # Test tokenizer
    sample_texts = [
        "To jest przykładowy tekst po polsku.",
        "Witaj świecie! Jak się masz?",
        "Uczenie maszynowe jest fascynujące.",
    ]

    tokenizer = PolishTokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)

    # Test encoding/decoding
    text = "Witaj świecie!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(f"Original: {text}")
    print(f"Encoded: {ids}")
    print(f"Decoded: {decoded}")
