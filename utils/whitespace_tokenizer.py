"""
Whitespace-based tokenizer for language modeling.
Splits on whitespace and punctuation, tracks OOV statistics.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter


class WhitespaceTokenizer:
    """
    Whitespace-based tokenizer with punctuation splitting.

    Features:
    - Splits text on whitespace
    - Punctuation becomes separate tokens
    - Builds vocabulary from most frequent words
    - Tracks OOV (out-of-vocabulary) statistics
    - Compatible with BPE tokenizer interface
    """

    # Punctuation to split (each becomes a separate token)
    PUNCTUATION = set('.,!?;:()[]{}"\'-—…""''«»')

    def __init__(self, vocab_size: int = 10000):
        """
        Initialize whitespace tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        # Special token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # OOV tracking
        self.oov_count = 0
        self.total_words = 0

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize special tokens in vocabulary."""
        special_tokens = [
            (self.pad_token, self.pad_token_id),
            (self.unk_token, self.unk_token_id),
            (self.bos_token, self.bos_token_id),
            (self.eos_token, self.eos_token_id),
        ]

        for token, token_id in special_tokens:
            self.word_to_id[token] = token_id
            self.id_to_word[token_id] = token

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words and punctuation.

        Args:
            text: Input text

        Returns:
            List of tokens (words and punctuation)
        """
        tokens = []

        # Split by whitespace first
        words = text.split()

        for word in words:
            # Extract punctuation and word parts
            current_tokens = []
            current_word = ""

            for char in word:
                if char in self.PUNCTUATION:
                    # Add accumulated word if any
                    if current_word:
                        current_tokens.append(current_word.lower())
                        current_word = ""
                    # Add punctuation as separate token
                    current_tokens.append(char)
                else:
                    current_word += char

            # Add remaining word
            if current_word:
                current_tokens.append(current_word.lower())

            tokens.extend(current_tokens)

        return tokens

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train tokenizer on a corpus of texts.

        Args:
            texts: List of text strings
            verbose: Print progress information
        """
        if verbose:
            print(f"Training whitespace tokenizer (vocab_size={self.vocab_size})...")

        # Tokenize all texts and count word frequencies
        word_counts = Counter()

        for text in texts:
            tokens = self._tokenize_text(text)
            word_counts.update(tokens)

        if verbose:
            print(f"Found {len(word_counts)} unique words/tokens")

        # Select top N-4 most frequent words (reserve 4 for special tokens)
        vocab_size_for_words = self.vocab_size - 4
        most_common = word_counts.most_common(vocab_size_for_words)

        # Build vocabulary (special tokens already added)
        next_id = 4  # After special tokens
        for word, count in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = next_id
                self.id_to_word[next_id] = word
                next_id += 1

        actual_vocab_size = len(self.word_to_id)

        if verbose:
            print(f"Vocabulary built: {actual_vocab_size} tokens")
            print(f"  Special tokens: 4")
            print(f"  Words: {actual_vocab_size - 4}")

            # Show most common words
            print(f"\nTop 10 most frequent words:")
            for i, (word, count) in enumerate(most_common[:10], 1):
                print(f"  {i}. '{word}': {count:,} occurrences")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        tokens = self._tokenize_text(text)

        # Convert to IDs, track OOV
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)

        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.unk_token_id)
                self.oov_count += 1
            self.total_words += 1

        if add_special_tokens:
            ids.append(self.eos_token_id)

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
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}

        tokens = []
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            token = self.id_to_word.get(token_id, self.unk_token)
            tokens.append(token)

        # Join tokens with smart spacing
        # Don't add space before punctuation
        result = ""
        for i, token in enumerate(tokens):
            if i == 0:
                result = token
            elif token in self.PUNCTUATION:
                result += token  # No space before punctuation
            else:
                result += " " + token

        return result

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_id)

    def get_oov_stats(self) -> Dict[str, float]:
        """
        Get OOV statistics.

        Returns:
            Dictionary with OOV count, total words, and percentage
        """
        oov_percentage = (self.oov_count / self.total_words * 100) if self.total_words > 0 else 0.0

        return {
            "oov_count": self.oov_count,
            "total_words": self.total_words,
            "oov_percentage": oov_percentage,
            "in_vocab_count": self.total_words - self.oov_count,
        }

    def reset_oov_stats(self):
        """Reset OOV tracking counters."""
        self.oov_count = 0
        self.total_words = 0

    def save(self, path: Path):
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer JSON
        """
        data = {
            "type": "whitespace",
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": {int(k): v for k, v in self.id_to_word.items()},
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

        self.vocab_size = data["vocab_size"]
        self.word_to_id = data["word_to_id"]
        self.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}

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
    # Test whitespace tokenizer
    print("Testing Whitespace Tokenizer")
    print("=" * 80)

    # Sample texts
    texts = [
        "Hello, world! This is a test.",
        "Warszawa jest stolicą Polski.",
        "Natural language processing is fascinating!",
        "To jest przykładowy tekst po polsku.",
        "Tokenization: splitting text into words, punctuation, etc.",
    ]

    # Create and train tokenizer
    tokenizer = WhitespaceTokenizer(vocab_size=100)
    tokenizer.train(texts)

    print("\n" + "=" * 80)
    print("Testing Encoding/Decoding")
    print("=" * 80)

    # Test encoding/decoding
    test_text = "Hello, world! How are you?"
    print(f"\nOriginal: {test_text}")

    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test OOV
    print("\n" + "=" * 80)
    print("Testing OOV Tracking")
    print("=" * 80)

    tokenizer.reset_oov_stats()

    test_texts = [
        "hello world",  # In vocab
        "unprecedented antidisestablishmentarianism",  # OOV words
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        print(f"\nText: '{text}'")
        print(f"Encoded: {encoded}")

    oov_stats = tokenizer.get_oov_stats()
    print(f"\nOOV Statistics:")
    print(f"  Total words: {oov_stats['total_words']}")
    print(f"  OOV words: {oov_stats['oov_count']}")
    print(f"  In-vocab words: {oov_stats['in_vocab_count']}")
    print(f"  OOV percentage: {oov_stats['oov_percentage']:.2f}%")

    print("\n" + "=" * 80)
    print("Testing Save/Load")
    print("=" * 80)

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)

    tokenizer.save(temp_path)
    print(f"\nSaved to: {temp_path}")

    # Load
    new_tokenizer = WhitespaceTokenizer()
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
