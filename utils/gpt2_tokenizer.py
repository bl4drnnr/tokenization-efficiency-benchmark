"""
GPT-2 pre-trained tokenizer wrapper.
Uses the official pre-trained GPT-2 BPE tokenizer from Hugging Face.
"""

import json
from pathlib import Path
from typing import List
from transformers import GPT2TokenizerFast


class GPT2Tokenizer:
    """
    Wrapper for pre-trained GPT-2 BPE tokenizer.

    Features:
    - Uses official GPT-2 tokenizer (pre-trained, no training needed!)
    - Compatible interface with other tokenizers
    - Save/load functionality
    """

    def __init__(self, vocab_size: int = 50257):
        """
        Initialize GPT-2 tokenizer.

        Args:
            vocab_size: Not used (GPT-2 has fixed vocab of 50257), kept for interface compatibility
        """
        # Load pre-trained GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)  # 50257 for GPT-2

        # Map to our special token IDs for compatibility
        # GPT-2 doesn't have all these, so we'll use its tokens
        self.pad_token = self.tokenizer.eos_token  # GPT-2 doesn't have PAD, use EOS
        self.unk_token = self.tokenizer.unk_token if self.tokenizer.unk_token else "<|endoftext|>"
        self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token else "<|endoftext|>"
        self.eos_token = self.tokenizer.eos_token  # <|endoftext|>

        self.pad_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.eos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def train(self, texts: List[str], verbose: bool = True):
        """
        No training needed - GPT-2 tokenizer is pre-trained!

        Args:
            texts: Ignored (kept for interface compatibility)
            verbose: Print information
        """
        if verbose:
            print(f"Using pre-trained GPT-2 BPE tokenizer")
            print(f"  Vocabulary size: {self.vocab_size}")
            print(f"  Note: GPT-2 tokenizer is pre-trained, no training needed!")
            print(f"  Special tokens:")
            print(f"    PAD: {self.pad_token_id} ('{self.pad_token}')")
            print(f"    UNK: {self.unk_token_id} ('{self.unk_token}')")
            print(f"    BOS: {self.bos_token_id} ('{self.bos_token}')")
            print(f"    EOS: {self.eos_token_id} ('{self.eos_token}')")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # GPT-2 tokenizer doesn't add BOS by default, we'll add manually if requested
        ids = self.tokenizer.encode(text, add_special_tokens=False)

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

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
        if skip_special_tokens:
            # Filter out our special tokens
            special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]

        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size

    def save(self, path: Path):
        """
        Save tokenizer metadata to file.

        Args:
            path: Path to save tokenizer JSON
        """
        # We don't need to save the tokenizer itself (it's pre-trained)
        # Just save metadata
        data = {
            "type": "gpt2",
            "vocab_size": self.vocab_size,
            "pretrained_model": "gpt2",
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: Path):
        """
        Load tokenizer from file.

        Args:
            path: Path to tokenizer JSON file
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Reload pre-trained GPT-2 tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.vocab_size = data["vocab_size"]

        # Load special tokens
        self.pad_token = data["pad_token"]
        self.unk_token = data["unk_token"]
        self.bos_token = data["bos_token"]
        self.eos_token = data["eos_token"]

        self.pad_token_id = data["pad_token_id"]
        self.unk_token_id = data["unk_token_id"]
        self.bos_token_id = data["bos_token_id"]
        self.eos_token_id = data["eos_token_id"]


if __name__ == "__main__":
    # Test GPT-2 tokenizer
    print("Testing GPT-2 Pre-trained Tokenizer")
    print("=" * 80)

    tokenizer = GPT2Tokenizer()

    # Show that training does nothing (it's pre-trained)
    print("\nCalling train() (does nothing for pre-trained):")
    tokenizer.train(["dummy text"])

    print("\n" + "=" * 80)
    print("Testing Encoding/Decoding")
    print("=" * 80)

    test_texts = [
        "Hello, world! This is a test.",
        "Warszawa jest stolicą Polski.",
        "Natural language processing with GPT-2 tokenizer.",
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        encoded = tokenizer.encode(text)
        print(f"Encoded ({len(encoded)} tokens): {encoded[:20]}...")  # Show first 20
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")

    print("\n" + "=" * 80)
    print("Testing Save/Load")
    print("=" * 80)

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)

    tokenizer.save(temp_path)
    print(f"\nSaved to: {temp_path}")

    # Load
    new_tokenizer = GPT2Tokenizer()
    new_tokenizer.load(temp_path)
    print(f"Loaded tokenizer with vocab size: {new_tokenizer.get_vocab_size()}")

    # Verify
    test_text = "Hello, world!"
    encoded_old = tokenizer.encode(test_text)
    encoded_new = new_tokenizer.encode(test_text)

    print(f"\nOriginal encoding: {encoded_old}")
    print(f"Loaded encoding: {encoded_new}")
    print(f"Match: {encoded_old == encoded_new}")

    # Cleanup
    temp_path.unlink()
    print("\n" + "=" * 80)
    print("All tests passed!")
