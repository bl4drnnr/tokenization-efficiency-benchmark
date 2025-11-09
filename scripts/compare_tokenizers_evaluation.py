#!/usr/bin/env python3
"""
Complete evaluation script for tokenizer comparison lab.
Calculates word-level and character-level perplexity, OOV stats, and efficiency metrics.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import json
import time
import numpy as np
from tqdm import tqdm
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.tokenizer_factory import create_tokenizer
from utils.dataset import LanguageModelingDataset
from torch.utils.data import DataLoader
from models.transformer_model import TransformerLanguageModel


def calculate_word_level_perplexity(model, tokenizer, test_texts, device, pad_token_id):
    """
    Calculate word-level perplexity (not token-level!).
    This measures perplexity per word, not per token.
    """
    model.eval()
    total_log_likelihood = 0.0
    total_words = 0

    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for text in tqdm(test_texts, desc="Calculating word-level perplexity"):
            # Count words (whitespace-separated)
            words = text.split()
            num_words = len(words)

            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            if len(token_ids) < 2:
                continue

            # Create input/target pairs
            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
            target_ids = torch.tensor(token_ids[1:], dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass
            try:
                outputs = model(input_ids)

                # Calculate loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = target_ids.reshape(-1)

                # Filter padding
                mask = (targets_flat != pad_token_id)
                if mask.sum() == 0:
                    continue

                losses = criterion(outputs_flat, targets_flat)
                masked_loss = losses[mask].sum()

                # Accumulate per word, not per token
                total_log_likelihood += masked_loss.item()
                total_words += num_words

            except Exception as e:
                continue

    if total_words == 0:
        return float('inf')

    # Perplexity per word
    avg_nll_per_word = total_log_likelihood / total_words
    word_perplexity = np.exp(avg_nll_per_word)

    return word_perplexity


def calculate_character_level_perplexity(model, tokenizer, test_texts, device, pad_token_id):
    """
    Calculate character-level perplexity.
    This measures perplexity per character.
    """
    model.eval()
    total_log_likelihood = 0.0
    total_chars = 0

    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for text in tqdm(test_texts, desc="Calculating character-level perplexity"):
            # Count characters (excluding whitespace for fairness)
            num_chars = len(text.replace(" ", ""))

            # Tokenize
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            if len(token_ids) < 2:
                continue

            # Create input/target pairs
            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
            target_ids = torch.tensor(token_ids[1:], dtype=torch.long).unsqueeze(0).to(device)

            # Forward pass
            try:
                outputs = model(input_ids)

                # Calculate loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = target_ids.reshape(-1)

                # Filter padding
                mask = (targets_flat != pad_token_id)
                if mask.sum() == 0:
                    continue

                losses = criterion(outputs_flat, targets_flat)
                masked_loss = losses[mask].sum()

                # Accumulate per character
                total_log_likelihood += masked_loss.item()
                total_chars += num_chars

            except Exception as e:
                continue

    if total_chars == 0:
        return float('inf')

    # Perplexity per character
    avg_nll_per_char = total_log_likelihood / total_chars
    char_perplexity = np.exp(avg_nll_per_char)

    return char_perplexity


def calculate_oov_statistics(tokenizer, test_texts, tokenizer_type):
    """
    Calculate OOV statistics (mainly for whitespace tokenizer).
    """
    if tokenizer_type != "whitespace":
        return {"oov_words": 0, "total_words": 0, "oov_percentage": 0.0}

    total_words = 0
    oov_words = 0

    for text in test_texts:
        words = text.split()
        total_words += len(words)

        # Check each word
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            # If encoded as UNK token (id=1 typically)
            if len(token_ids) == 1 and token_ids[0] == tokenizer.unk_token_id:
                oov_words += 1

    oov_percentage = (oov_words / total_words * 100) if total_words > 0 else 0.0

    return {
        "oov_words": oov_words,
        "total_words": total_words,
        "oov_percentage": oov_percentage
    }


def calculate_tokens_per_word(tokenizer, test_texts):
    """
    Calculate average tokens per word.
    """
    total_tokens = 0
    total_words = 0

    for text in test_texts:
        words = text.split()
        total_words += len(words)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)

    avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0.0

    return avg_tokens_per_word


def calculate_words_encoded_directly(tokenizer, test_texts, tokenizer_type):
    """
    Calculate percentage of words encoded as single tokens (not split).
    """
    if tokenizer_type == "whitespace":
        # For whitespace, count non-UNK words
        total_words = 0
        direct_words = 0

        for text in test_texts:
            words = text.split()
            total_words += len(words)

            for word in words:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1 and token_ids[0] != tokenizer.unk_token_id:
                    direct_words += 1
    else:
        # For subword tokenizers (GPT-2, SentencePiece), count words that map to single token
        total_words = 0
        direct_words = 0

        for text in test_texts:
            words = text.split()
            total_words += len(words)

            for word in words:
                token_ids = tokenizer.encode(word, add_special_tokens=False)
                if len(token_ids) == 1:
                    direct_words += 1

    percentage = (direct_words / total_words * 100) if total_words > 0 else 0.0

    return {
        "direct_words": direct_words,
        "total_words": total_words,
        "percentage": percentage
    }


def measure_inference_time(model, tokenizer, test_texts, device):
    """
    Measure inference time (tokens per second).
    """
    model.eval()
    total_tokens = 0
    total_time = 0.0

    with torch.no_grad():
        for text in test_texts[:100]:  # Use first 100 texts for speed
            token_ids = tokenizer.encode(text, add_special_tokens=True)

            if len(token_ids) < 2:
                continue

            input_ids = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(0).to(device)

            start_time = time.time()
            try:
                _ = model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
            except:
                continue
            elapsed = time.time() - start_time

            total_time += elapsed
            total_tokens += len(token_ids) - 1

    tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0

    return tokens_per_second


def load_test_data(dataset_name, config):
    """Load test texts for evaluation."""
    # Load test IDs
    test_ids = torch.load(config.data_processed_dir / f"{dataset_name}_test_ids.pt")

    # We need raw text, not just IDs
    # Try to load from corpus if available
    corpus_path = config.data_raw_dir / "forum_forum_poradnikogrodniczy_pl_corpus.txt"

    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            all_texts = [line.strip() for line in f if line.strip()]

        # Take a subset for testing (last 1000 texts as they weren't used in training)
        test_texts = all_texts[-1000:]
    else:
        # Fallback: create dummy texts
        test_texts = [
            "Uprawa pomidorów wymaga odpowiedniego podłoża i regularnego podlewania.",
            "Ogórki rosną najlepiej w ciepłych i słonecznych miejscach w ogrodzie.",
            "Papryka potrzebuje dużo światła i ciepła aby wydać obfite plony.",
        ] * 100

    return test_texts


def main():
    """Main evaluation function."""
    config = get_config("transformer")

    # Tokenizer configurations
    tokenizers_config = [
        ("gpt2", "forum_forum_poradnikogrodniczy_pl_corpus_gpt2"),
        ("sentencepiece", "forum_forum_poradnikogrodniczy_pl_corpus_sentencepiece"),
        ("whitespace", "forum_forum_poradnikogrodniczy_pl_corpus_whitespace"),
    ]

    results = {}

    print("=" * 80)
    print("TOKENIZER COMPARISON EVALUATION")
    print("=" * 80)

    for tokenizer_type, dataset_name in tokenizers_config:
        print(f"\n{'='*80}")
        print(f"Evaluating: {tokenizer_type.upper()} tokenizer")
        print(f"{'='*80}")

        # Load tokenizer
        tokenizer = create_tokenizer(tokenizer_type, vocab_size=config.vocab_size)
        tokenizer_path = config.data_processed_dir / f"{dataset_name}_tokenizer.json"
        tokenizer.load(tokenizer_path)

        # Load metadata
        metadata_path = config.data_processed_dir / f"{dataset_name}_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            pad_token_id = metadata["pad_token_id"]

        # Load model checkpoint
        checkpoint_path = config.checkpoints_dir / f"{dataset_name}_{tokenizer_type}_transformer_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

        # Create model
        model = TransformerLanguageModel(
            vocab_size=metadata["vocab_size"],
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

        # Load test data
        print("\nLoading test data...")
        test_texts = load_test_data(dataset_name, config)
        print(f"Loaded {len(test_texts)} test texts")

        # Load training metrics
        metrics_path = config.results_dir / f"{dataset_name}_{tokenizer_type}_transformer_metrics.json"
        with open(metrics_path, "r") as f:
            training_metrics = json.load(f)

        # Calculate evaluations
        print("\n1. Calculating word-level perplexity...")
        word_ppl = calculate_word_level_perplexity(model, tokenizer, test_texts[:500], config.device, pad_token_id)

        print("2. Calculating character-level perplexity...")
        char_ppl = calculate_character_level_perplexity(model, tokenizer, test_texts[:500], config.device, pad_token_id)

        print("3. Calculating OOV statistics...")
        oov_stats = calculate_oov_statistics(tokenizer, test_texts, tokenizer_type)

        print("4. Calculating tokens per word...")
        tokens_per_word = calculate_tokens_per_word(tokenizer, test_texts)

        print("5. Calculating words encoded directly...")
        direct_encoding = calculate_words_encoded_directly(tokenizer, test_texts, tokenizer_type)

        print("6. Measuring inference speed...")
        inference_speed = measure_inference_time(model, tokenizer, test_texts, config.device)

        # Store results
        results[tokenizer_type] = {
            "tokenizer_type": tokenizer_type,
            "vocab_size": metadata["vocab_size"],
            "word_level_perplexity": word_ppl,
            "character_level_perplexity": char_ppl,
            "oov_statistics": oov_stats,
            "tokens_per_word": tokens_per_word,
            "words_encoded_directly": direct_encoding,
            "inference_speed_tokens_per_sec": inference_speed,
            "training_time_total_seconds": training_metrics.get("total_time", 0),
            "training_time_avg_epoch_seconds": training_metrics.get("avg_epoch_time", 0),
            "best_val_perplexity_token_level": training_metrics.get("best_val_perplexity", 0),
        }

        # Print summary
        print(f"\n{tokenizer_type.upper()} Results:")
        print(f"  Word-level perplexity: {word_ppl:.2f}")
        print(f"  Character-level perplexity: {char_ppl:.2f}")
        print(f"  OOV: {oov_stats['oov_words']}/{oov_stats['total_words']} ({oov_stats['oov_percentage']:.2f}%)")
        print(f"  Tokens per word: {tokens_per_word:.2f}")
        print(f"  Words encoded directly: {direct_encoding['percentage']:.2f}%")
        print(f"  Inference speed: {inference_speed:.0f} tokens/sec")

    # Save results
    output_path = config.results_dir / "tokenizer_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    return results


if __name__ == "__main__":
    main()
