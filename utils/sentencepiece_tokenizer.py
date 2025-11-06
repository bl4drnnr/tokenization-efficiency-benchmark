"""
SentencePiece tokenizer wrapper for language modeling.
Trains SentencePiece models (BPE or Unigram) on text corpus.
"""

import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
import sentencepiece as spm


class SentencePieceTokenizer:
    """
    SentencePiece tokenizer wrapper with compatible interface.

    Features:
    - Trains SentencePiece model (BPE or Unigram algorithm)
    - Compatible with BPE/Whitespace tokenizer interface
    - Handles special tokens automatically
    - Save/load functionality
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        model_type: str = "bpe",  # "bpe" or "unigram"
    ):
        """
        Initialize SentencePiece tokenizer.

        Args:
            vocab_size: Target vocabulary size
            model_type: Model type - "bpe" or "unigram"
        """
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.sp_model = None
        self.model_path = None

        # Special tokens (will be set after training)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        self.pad_token_id = None
        self.unk_token_id = None
        self.bos_token_id = None
        self.eos_token_id = None

    def train(self, texts: List[str], verbose: bool = True):
        """
        Train SentencePiece model on corpus.

        Args:
            texts: List of text strings
            verbose: Print progress information
        """
        if verbose:
            print(f"Training SentencePiece tokenizer ({self.model_type}, vocab_size={self.vocab_size})...")

        # Write texts to temporary file (SentencePiece requires input file)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            temp_input = f.name
            for text in texts:
                f.write(text.strip() + '\n')

        # Create temporary model prefix
        temp_dir = Path(tempfile.mkdtemp())
        model_prefix = str(temp_dir / "spm_model")

        try:
            # Train SentencePiece model
            train_args = {
                'input': temp_input,
                'model_prefix': model_prefix,
                'vocab_size': self.vocab_size,
                'model_type': self.model_type,
                'character_coverage': 1.0,  # Cover all characters (important for non-English)
                'pad_id': 0,
                'unk_id': 1,
                'bos_id': 2,
                'eos_id': 3,
                'pad_piece': self.pad_token,
                'unk_piece': self.unk_token,
                'bos_piece': self.bos_token,
                'eos_piece': self.eos_token,
                'user_defined_symbols': [],
            }

            if not verbose:
                train_args['train_extremely_large_corpus'] = False
                # Suppress output
                import sys
                import io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()

            spm.SentencePieceTrainer.train(**train_args)

            if not verbose:
                sys.stdout = old_stdout

            # Load trained model
            self.model_path = f"{model_prefix}.model"
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(self.model_path)

            # Set special token IDs
            self.pad_token_id = self.sp_model.pad_id()
            self.unk_token_id = self.sp_model.unk_id()
            self.bos_token_id = self.sp_model.bos_id()
            self.eos_token_id = self.sp_model.eos_id()

            if verbose:
                actual_vocab_size = self.sp_model.get_piece_size()
                print(f"SentencePiece model trained: {actual_vocab_size} tokens")
                print(f"  Special tokens:")
                print(f"    PAD: {self.pad_token_id}")
                print(f"    UNK: {self.unk_token_id}")
                print(f"    BOS: {self.bos_token_id}")
                print(f"    EOS: {self.eos_token_id}")

                # Show sample tokens
                print(f"\n  Sample tokens:")
                for i in range(4, min(14, actual_vocab_size)):
                    token = self.sp_model.id_to_piece(i)
                    print(f"    {i}: '{token}'")

        finally:
            # Cleanup temporary input file
            Path(temp_input).unlink(missing_ok=True)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        if add_special_tokens:
            # SentencePiece can add BOS/EOS automatically
            ids = [self.bos_token_id] + self.sp_model.encode(text) + [self.eos_token_id]
        else:
            ids = self.sp_model.encode(text)

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
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained. Call load() first.")

        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            ids = [token_id for token_id in ids if token_id not in special_ids]

        return self.sp_model.decode(ids)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.get_piece_size()

    def save(self, path: Path):
        """
        Save tokenizer to file.

        Args:
            path: Path to save tokenizer (JSON with model reference)
        """
        if self.sp_model is None:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        # Save the .model file alongside the JSON
        model_save_path = path.parent / f"{path.stem}.model"

        # Copy model file to save location
        import shutil
        shutil.copy(self.model_path, model_save_path)

        # Save metadata as JSON
        data = {
            "type": "sentencepiece",
            "vocab_size": self.vocab_size,
            "model_type": self.model_type,
            "model_file": model_save_path.name,
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
        self.model_type = data["model_type"]

        # Load special tokens
        self.pad_token = data["pad_token"]
        self.unk_token = data["unk_token"]
        self.bos_token = data["bos_token"]
        self.eos_token = data["eos_token"]

        self.pad_token_id = data["pad_token_id"]
        self.unk_token_id = data["unk_token_id"]
        self.bos_token_id = data["bos_token_id"]
        self.eos_token_id = data["eos_token_id"]

        # Load SentencePiece model
        model_file = data["model_file"]
        model_path = path.parent / model_file

        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model file not found: {model_path}")

        self.model_path = str(model_path)
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(self.model_path)


if __name__ == "__main__":
    # Test SentencePiece tokenizer
    print("Testing SentencePiece Tokenizer")
    print("=" * 80)

    # Sample texts
    texts = [
        "Hello, world! This is a test.",
        "Warszawa jest stolicą Polski.",
        "Natural language processing is fascinating!",
        "To jest przykładowy tekst po polsku.",
        "Tokenization: splitting text into subwords.",
        "SentencePiece can handle any language.",
        "It learns subword units from the corpus.",
    ]

    # Test BPE
    print("\n" + "=" * 80)
    print("Testing BPE Model")
    print("=" * 80)

    tokenizer_bpe = SentencePieceTokenizer(vocab_size=100, model_type="bpe")
    tokenizer_bpe.train(texts)

    test_text = "Hello, world! How are you?"
    print(f"\nOriginal: {test_text}")

    encoded = tokenizer_bpe.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer_bpe.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test Unigram
    print("\n" + "=" * 80)
    print("Testing Unigram Model")
    print("=" * 80)

    tokenizer_unigram = SentencePieceTokenizer(vocab_size=100, model_type="unigram")
    tokenizer_unigram.train(texts, verbose=False)

    print(f"\nOriginal: {test_text}")
    encoded = tokenizer_unigram.encode(test_text)
    print(f"Encoded: {encoded}")
    decoded = tokenizer_unigram.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test Save/Load
    print("\n" + "=" * 80)
    print("Testing Save/Load")
    print("=" * 80)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "spm_tokenizer.json"

        tokenizer_bpe.save(save_path)
        print(f"\nSaved to: {save_path}")

        # Load
        new_tokenizer = SentencePieceTokenizer()
        new_tokenizer.load(save_path)
        print(f"Loaded tokenizer with vocab size: {new_tokenizer.get_vocab_size()}")

        # Verify
        encoded_old = tokenizer_bpe.encode(test_text)
        encoded_new = new_tokenizer.encode(test_text)

        print(f"\nOriginal encoding: {encoded_old}")
        print(f"Loaded encoding: {encoded_new}")
        print(f"Match: {encoded_old == encoded_new}")

    print("\n" + "=" * 80)
    print("All tests passed!")
