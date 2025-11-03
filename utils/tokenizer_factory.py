"""
Tokenizer factory for creating different types of tokenizers.
Provides a unified interface for BPE, Whitespace, and SentencePiece tokenizers.
"""

from pathlib import Path
from typing import Union
from utils.tokenizer import PolishTokenizer  # BPE tokenizer
from utils.whitespace_tokenizer import WhitespaceTokenizer


def create_tokenizer(tokenizer_type: str, vocab_size: int = 10000):
    """
    Create a tokenizer of the specified type.

    Args:
        tokenizer_type: Type of tokenizer ('bpe', 'whitespace', or 'sentencepiece')
        vocab_size: Vocabulary size

    Returns:
        Tokenizer instance

    Raises:
        ValueError: If tokenizer type is unknown
    """
    tokenizer_type = tokenizer_type.lower()

    if tokenizer_type == "bpe":
        return PolishTokenizer(vocab_size=vocab_size)
    elif tokenizer_type == "whitespace":
        return WhitespaceTokenizer(vocab_size=vocab_size)
    elif tokenizer_type == "sentencepiece":
        # TODO: Implement SentencePiece tokenizer
        raise NotImplementedError("SentencePiece tokenizer not yet implemented")
    else:
        raise ValueError(
            f"Unknown tokenizer type: {tokenizer_type}. "
            f"Choose from: 'bpe', 'whitespace', 'sentencepiece'"
        )


def load_tokenizer(tokenizer_path: Path, tokenizer_type: str = None):
    """
    Load a tokenizer from file.

    Args:
        tokenizer_path: Path to tokenizer file
        tokenizer_type: Type of tokenizer (optional, auto-detected from file if not provided)

    Returns:
        Loaded tokenizer instance

    Raises:
        ValueError: If tokenizer type cannot be determined
    """
    import json

    # Try to auto-detect tokenizer type from file
    if tokenizer_type is None:
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tokenizer_type = data.get('type', None)

            if tokenizer_type is None:
                # BPE tokenizer doesn't have 'type' field
                # Check for BPE-specific fields
                if 'model' in data and 'vocab' in data:
                    tokenizer_type = 'bpe'
                elif 'word_to_id' in data:
                    tokenizer_type = 'whitespace'
        except Exception:
            # Default to BPE if can't determine
            tokenizer_type = 'bpe'

    # Create appropriate tokenizer and load
    if tokenizer_type.lower() == "bpe":
        tokenizer = PolishTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer
    elif tokenizer_type.lower() == "whitespace":
        tokenizer = WhitespaceTokenizer()
        tokenizer.load(tokenizer_path)
        return tokenizer
    elif tokenizer_type.lower() == "sentencepiece":
        # TODO: Implement SentencePiece loading
        raise NotImplementedError("SentencePiece tokenizer not yet implemented")
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def get_tokenizer_name(tokenizer_type: str) -> str:
    """
    Get a human-readable name for the tokenizer type.

    Args:
        tokenizer_type: Type of tokenizer

    Returns:
        Human-readable name
    """
    names = {
        "bpe": "BPE (Byte-Pair Encoding)",
        "whitespace": "Whitespace-based",
        "sentencepiece": "SentencePiece",
    }
    return names.get(tokenizer_type.lower(), tokenizer_type)


def get_available_tokenizers() -> list:
    """
    Get list of available tokenizer types.

    Returns:
        List of tokenizer type strings
    """
    return ["bpe", "whitespace", "sentencepiece"]


if __name__ == "__main__":
    # Test tokenizer factory
    print("Testing Tokenizer Factory")
    print("=" * 80)

    # Show available tokenizers
    print("\nAvailable tokenizers:")
    for tok_type in get_available_tokenizers():
        print(f"  - {tok_type}: {get_tokenizer_name(tok_type)}")

    # Test BPE creation
    print("\n" + "=" * 80)
    print("Testing BPE Tokenizer Creation")
    print("=" * 80)

    bpe = create_tokenizer("bpe", vocab_size=1000)
    print(f"Created: {type(bpe).__name__}")

    # Test Whitespace creation
    print("\n" + "=" * 80)
    print("Testing Whitespace Tokenizer Creation")
    print("=" * 80)

    whitespace = create_tokenizer("whitespace", vocab_size=1000)
    print(f"Created: {type(whitespace).__name__}")

    # Test invalid tokenizer
    print("\n" + "=" * 80)
    print("Testing Invalid Tokenizer")
    print("=" * 80)

    try:
        invalid = create_tokenizer("invalid_type")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\n" + "=" * 80)
    print("All tests passed!")
