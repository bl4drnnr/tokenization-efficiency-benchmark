#!/usr/bin/env python3
"""
Qualitative analysis script for tokenizer comparison.
Shows tokenization examples for sample texts.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.tokenizer_factory import create_tokenizer


def analyze_tokenization(text, tokenizer, tokenizer_type):
    """Analyze how a text is tokenized."""
    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.decode_tokens(token_ids) if hasattr(tokenizer, 'decode_tokens') else ["?" for _ in token_ids]

    # Count words
    words = text.split()
    num_words = len(words)
    num_tokens = len(token_ids)

    # Calculate tokens per word
    tokens_per_word = num_tokens / num_words if num_words > 0 else 0

    # Count words encoded directly (as single token)
    direct_words = 0
    for word in words:
        word_token_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(word_token_ids) == 1:
            # Check if not UNK for whitespace tokenizer
            if tokenizer_type == "whitespace":
                if word_token_ids[0] != tokenizer.unk_token_id:
                    direct_words += 1
            else:
                direct_words += 1

    direct_percentage = (direct_words / num_words * 100) if num_words > 0 else 0

    return {
        "text": text,
        "words": words,
        "num_words": num_words,
        "token_ids": token_ids,
        "tokens": tokens,
        "num_tokens": num_tokens,
        "tokens_per_word": tokens_per_word,
        "direct_words": direct_words,
        "direct_percentage": direct_percentage,
    }


def format_tokens_visualization(analysis):
    """Create a nice visualization of tokenization."""
    output = []
    output.append(f"Original text ({analysis['num_words']} words):")
    output.append(f"  \"{analysis['text']}\"")
    output.append(f"")
    output.append(f"Tokenized ({analysis['num_tokens']} tokens):")

    # Show tokens with delimiters
    token_str = " | ".join([str(t) for t in analysis['tokens'][:50]])  # Limit to first 50 tokens
    if len(analysis['tokens']) > 50:
        token_str += " | ..."
    output.append(f"  [{token_str}]")

    output.append(f"")
    output.append(f"Statistics:")
    output.append(f"  - Tokens per word: {analysis['tokens_per_word']:.2f}")
    output.append(f"  - Words encoded directly: {analysis['direct_words']}/{analysis['num_words']} ({analysis['direct_percentage']:.1f}%)")

    return "\n".join(output)


def main():
    """Main qualitative analysis function."""
    config = get_config("transformer")

    # Sample texts (at least 30 words each, as per instructions)
    sample_texts = [
        # Sample 1: General gardening advice (31 words)
        "Uprawa pomidorów w ogrodzie wymaga odpowiedniego przygotowania gleby, regularnego podlewania oraz "
        "stosowania nawozów organicznych. Najlepsze wyniki uzyskuje się sadząc rośliny w miejscach "
        "nasłonecznionych, gdzie panuje stały dostęp do wody.",

        # Sample 2: Specific technical advice (35 words)
        "Ogórki gruntowe powinny być uprawiane w glebie zasobnej w próchnicę, o odczynie lekko kwaśnym. "
        "Wysiew nasion przeprowadza się pod koniec maja, gdy temperatura gleby osiągnie co najmniej 15 "
        "stopni Celsjusza. Regularnie podlewaj rośliny rano lub wieczorem.",

        # Sample 3: Forum-style question (32 words)
        "Witam wszystkich na forum ogrodniczym! Mam pytanie dotyczące uprawy papryki w tunelu foliowym. "
        "Czy ktoś z Was ma doświadczenie z odmianami papryki słodkiej? Proszę o porady i wskazówki "
        "dla początkujących ogrodników. Pozdrawiam serdecznie!",
    ]

    # Tokenizer configurations
    tokenizers_config = [
        ("gpt2", "forum_forum_poradnikogrodniczy_pl_corpus_gpt2"),
        ("sentencepiece", "forum_forum_poradnikogrodniczy_pl_corpus_sentencepiece"),
        ("whitespace", "forum_forum_poradnikogrodniczy_pl_corpus_whitespace"),
    ]

    print("=" * 80)
    print("QUALITATIVE TOKENIZATION ANALYSIS")
    print("=" * 80)

    results = {}

    for tokenizer_type, dataset_name in tokenizers_config:
        print(f"\n{'='*80}")
        print(f"{tokenizer_type.upper()} TOKENIZER")
        print(f"{'='*80}")

        # Load tokenizer
        tokenizer = create_tokenizer(tokenizer_type, vocab_size=config.vocab_size)
        tokenizer_path = config.data_processed_dir / f"{dataset_name}_tokenizer.json"
        tokenizer.load(tokenizer_path)

        results[tokenizer_type] = []

        for i, text in enumerate(sample_texts, 1):
            print(f"\n--- Sample {i} ---")
            analysis = analyze_tokenization(text, tokenizer, tokenizer_type)
            results[tokenizer_type].append(analysis)

            # Print visualization
            print(format_tokens_visualization(analysis))

    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")

    print(f"{'Sample':<10} {'Tokenizer':<15} {'Words':<8} {'Tokens':<8} {'Tok/Word':<10} {'Direct %':<10}")
    print("-" * 80)

    for i in range(len(sample_texts)):
        for tokenizer_type in ["gpt2", "sentencepiece", "whitespace"]:
            analysis = results[tokenizer_type][i]
            print(f"#{i+1:<9} {tokenizer_type:<15} {analysis['num_words']:<8} "
                  f"{analysis['num_tokens']:<8} {analysis['tokens_per_word']:<10.2f} "
                  f"{analysis['direct_percentage']:<10.1f}%")

    # Save detailed results to file
    output_path = config.results_dir / "qualitative_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("QUALITATIVE TOKENIZATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        for tokenizer_type, dataset_name in tokenizers_config:
            f.write(f"\n{'='*80}\n")
            f.write(f"{tokenizer_type.upper()} TOKENIZER\n")
            f.write(f"{'='*80}\n")

            for i, analysis in enumerate(results[tokenizer_type], 1):
                f.write(f"\n--- Sample {i} ---\n")
                f.write(format_tokens_visualization(analysis) + "\n")

        # Add comparison table
        f.write(f"\n{'='*80}\n")
        f.write("COMPARISON TABLE\n")
        f.write(f"{'='*80}\n\n")

        f.write(f"{'Sample':<10} {'Tokenizer':<15} {'Words':<8} {'Tokens':<8} {'Tok/Word':<10} {'Direct %':<10}\n")
        f.write("-" * 80 + "\n")

        for i in range(len(sample_texts)):
            for tokenizer_type in ["gpt2", "sentencepiece", "whitespace"]:
                analysis = results[tokenizer_type][i]
                f.write(f"#{i+1:<9} {tokenizer_type:<15} {analysis['num_words']:<8} "
                        f"{analysis['num_tokens']:<8} {analysis['tokens_per_word']:<10.2f} "
                        f"{analysis['direct_percentage']:<10.1f}%\n")

    print(f"\n{'='*80}")
    print(f"Detailed results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
