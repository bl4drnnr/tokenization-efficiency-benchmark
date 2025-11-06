"""
Comprehensive tokenizer comparison script.
Implements word-level and character-level perplexity, tokenization statistics,
and qualitative analysis as required by the lab instructions.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import json
import time
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.tokenizer_factory import load_tokenizer, get_tokenizer_name
from utils.dataset import LanguageModelingDataset
from torch.utils.data import DataLoader
from models.transformer_model import TransformerLanguageModel
import math


def calculate_word_level_perplexity(
    model, dataloader, criterion, device, pad_token_id, tokenizer, original_texts
):
    """
    Calculate word-level perplexity (comparable across tokenizers).

    Returns total loss normalized by number of words, not tokens.
    """
    model.eval()
    total_loss = 0.0
    total_words = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            batch_size, seq_len, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            mask = (targets_flat != pad_token_id)
            loss = criterion(outputs_flat[mask], targets_flat[mask])

            # Count words in original text for this batch
            batch_word_count = sum(len(text.split()) for text in original_texts[idx * batch_size:(idx + 1) * batch_size])

            total_loss += loss.item() * mask.sum().item()
            total_words += batch_word_count

    avg_loss_per_word = total_loss / total_words if total_words > 0 else float('inf')
    perplexity = math.exp(avg_loss_per_word) if avg_loss_per_word < 100 else float('inf')

    return perplexity, avg_loss_per_word


def calculate_char_level_perplexity(
    model, dataloader, criterion, device, pad_token_id, tokenizer, original_texts
):
    """
    Calculate character-level perplexity (comparable across tokenizers).

    Returns total loss normalized by number of characters, not tokens.
    """
    model.eval()
    total_loss = 0.0
    total_chars = 0

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            batch_size, seq_len, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)

            mask = (targets_flat != pad_token_id)
            loss = criterion(outputs_flat[mask], targets_flat[mask])

            # Count characters in original text for this batch
            batch_char_count = sum(len(text) for text in original_texts[idx * batch_size:(idx + 1) * batch_size])

            total_loss += loss.item() * mask.sum().item()
            total_chars += batch_char_count

    avg_loss_per_char = total_loss / total_chars if total_chars > 0 else float('inf')
    perplexity = math.exp(avg_loss_per_char) if avg_loss_per_char < 100 else float('inf')

    return perplexity, avg_loss_per_char


def calculate_tokenization_stats(tokenizer, texts: List[str]) -> Dict:
    """
    Calculate tokenization statistics: tokens per word, direct word coverage.
    """
    total_tokens = 0
    total_words = 0
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()

        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += len(text)

    avg_tokens_per_word = total_tokens / total_words if total_words > 0 else 0
    avg_tokens_per_char = total_tokens / total_chars if total_chars > 0 else 0

    # Calculate vocabulary coverage (for whitespace tokenizer)
    if hasattr(tokenizer, 'get_oov_stats'):
        tokenizer.reset_oov_stats()
        for text in texts:
            _ = tokenizer.encode(text, add_special_tokens=False)
        oov_stats = tokenizer.get_oov_stats()
    else:
        oov_stats = None

    return {
        'total_tokens': total_tokens,
        'total_words': total_words,
        'total_chars': total_chars,
        'avg_tokens_per_word': avg_tokens_per_word,
        'avg_tokens_per_char': avg_tokens_per_char,
        'oov_stats': oov_stats,
    }


def qualitative_analysis(tokenizer, tokenizer_type: str, sample_texts: List[str]):
    """
    Perform qualitative analysis on sample texts.
    """
    print(f"\n{'='*80}")
    print(f"QUALITATIVE ANALYSIS - {get_tokenizer_name(tokenizer_type)}")
    print(f"{'='*80}\n")

    for idx, text in enumerate(sample_texts, 1):
        print(f"Sample {idx}:")
        print(f"  Original: {text[:100]}...")

        tokens_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens_ids, skip_special_tokens=True)

        words = text.split()
        num_words = len(words)
        num_tokens = len(tokens_ids)
        tokens_per_word = num_tokens / num_words if num_words > 0 else 0

        print(f"  Words: {num_words}")
        print(f"  Tokens: {num_tokens}")
        print(f"  Tokens/Word: {tokens_per_word:.2f}")
        print(f"  Decoded: {decoded[:100]}...")
        print()


def compare_tokenizers(
    dataset: str,
    tokenizer_types: List[str] = ["bpe", "whitespace", "sentencepiece"],
    data_type: str = "test"
):
    """
    Compare all tokenizers comprehensively.
    """
    print("="*80)
    print("COMPREHENSIVE TOKENIZER COMPARISON")
    print("="*80)

    config = get_config("transformer")
    results = {}

    # Sample texts for qualitative analysis
    sample_texts = [
        "Warszawa jest stolicą Polski i największym miastem w kraju.",
        "Sztuczna inteligencja i uczenie maszynowe rewolucjonizują wiele dziedzin nauki.",
        "Tokenizacja to proces dzielenia tekstu na mniejsze jednostki zwane tokenami.",
    ]

    for tok_type in tokenizer_types:
        print(f"\n{'='*80}")
        print(f"Analyzing: {get_tokenizer_name(tok_type)}")
        print(f"{'='*80}")

        # Load tokenizer and data
        tokenizer_path = config.data_processed_dir / f"{dataset}_{tok_type}_tokenizer.json"
        if not tokenizer_path.exists():
            print(f"⚠️  Tokenizer not found: {tokenizer_path}")
            print(f"   Run: python scripts/preprocess_data.py --input data/raw/your_data.txt --tokenizer {tok_type}")
            continue

        tokenizer = load_tokenizer(tokenizer_path, tok_type)
        print(f"✓ Loaded tokenizer: {tokenizer_path.name}")
        print(f"  Vocabulary size: {tokenizer.get_vocab_size()}")

        # Load model checkpoint
        checkpoint_path = config.checkpoints_dir / f"{dataset}_{tok_type}_transformer_best.pt"
        if not checkpoint_path.exists():
            print(f"⚠️  Model checkpoint not found: {checkpoint_path}")
            print(f"   Run: python scripts/train.py --dataset {dataset}_{tok_type}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

        # Create model
        model = TransformerLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.embedding_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.ff_dim,
            max_seq_len=config.max_seq_length,
            dropout=config.dropout,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.device)
        model.eval()

        # Load test data and original texts
        test_ids = torch.load(config.data_processed_dir / f"{dataset}_{tok_type}_test_ids.pt")

        # Load original texts for word/char perplexity
        from utils.dataset import load_text_file
        raw_data_path = config.data_raw_dir / f"{dataset}.txt"
        if raw_data_path.exists():
            all_texts = load_text_file(raw_data_path)
            # Use last 5% for test (matching split ratio)
            test_size = int(len(all_texts) * 0.05)
            original_test_texts = all_texts[-test_size:]
        else:
            print("⚠️  Original text file not found, skipping word/char perplexity")
            original_test_texts = []

        # Create dataloader
        test_dataset = LanguageModelingDataset(
            test_ids,
            max_length=config.max_seq_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Calculate metrics
        print("\nCalculating metrics...")

        # Word-level perplexity
        if original_test_texts:
            word_ppl, word_loss = calculate_word_level_perplexity(
                model, test_loader, criterion, config.device,
                tokenizer.pad_token_id, tokenizer, original_test_texts
            )
            print(f"  Word-level perplexity: {word_ppl:.2f}")
        else:
            word_ppl = None

        # Character-level perplexity
        if original_test_texts:
            char_ppl, char_loss = calculate_char_level_perplexity(
                model, test_loader, criterion, config.device,
                tokenizer.pad_token_id, tokenizer, original_test_texts
            )
            print(f"  Char-level perplexity: {char_ppl:.2f}")
        else:
            char_ppl = None

        # Tokenization statistics
        if original_test_texts:
            tok_stats = calculate_tokenization_stats(tokenizer, original_test_texts[:1000])  # Sample
            print(f"  Avg tokens per word: {tok_stats['avg_tokens_per_word']:.3f}")
            if tok_stats['oov_stats']:
                print(f"  OOV percentage: {tok_stats['oov_stats']['oov_percentage']:.2f}%")

        # Inference speed
        start_time = time.time()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(config.device)
                _ = model(inputs)
        inference_time = time.time() - start_time
        total_tokens_processed = sum(len(ids) for ids in test_ids)
        tokens_per_second = total_tokens_processed / inference_time
        print(f"  Inference speed: {tokens_per_second:.0f} tokens/sec")

        # Qualitative analysis
        qualitative_analysis(tokenizer, tok_type, sample_texts)

        # Store results
        results[tok_type] = {
            'tokenizer_type': tok_type,
            'vocab_size': tokenizer.get_vocab_size(),
            'word_level_perplexity': word_ppl,
            'char_level_perplexity': char_ppl,
            'tokenization_stats': tok_stats if original_test_texts else None,
            'inference_tokens_per_sec': tokens_per_second,
            'inference_time': inference_time,
        }

    # Save comparison results
    results_path = config.results_dir / f"{dataset}_tokenizer_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    print(f"Results saved to: {results_path}")

    # Print comparison table
    print(f"\n{'Metric':<30} {'BPE':<15} {'Whitespace':<15} {'SentencePiece':<15}")
    print("-" * 75)

    if all(results.get(t, {}).get('word_level_perplexity') for t in tokenizer_types):
        print(f"{'Word-level PPL':<30}", end="")
        for t in tokenizer_types:
            if t in results and results[t]['word_level_perplexity']:
                print(f"{results[t]['word_level_perplexity']:<15.2f}", end="")
        print()

    if all(results.get(t, {}).get('char_level_perplexity') for t in tokenizer_types):
        print(f"{'Char-level PPL':<30}", end="")
        for t in tokenizer_types:
            if t in results and results[t]['char_level_perplexity']:
                print(f"{results[t]['char_level_perplexity']:<15.2f}", end="")
        print()

    print(f"{'Inference (tokens/sec)':<30}", end="")
    for t in tokenizer_types:
        if t in results:
            print(f"{results[t]['inference_tokens_per_sec']:<15.0f}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers comprehensively")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (without tokenizer suffix)",
    )
    parser.add_argument(
        "--tokenizers",
        type=str,
        nargs="+",
        default=["bpe", "whitespace", "sentencepiece"],
        help="Tokenizers to compare",
    )

    args = parser.parse_args()

    compare_tokenizers(args.dataset, args.tokenizers)


if __name__ == "__main__":
    main()
