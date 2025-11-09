#!/usr/bin/env python3
"""
Generate final lab report with all metrics and analysis.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config


def generate_markdown_report():
    """Generate a comprehensive markdown report."""
    config = get_config("transformer")

    # Load comparison results
    comparison_path = config.results_dir / "tokenizer_comparison_results.json"
    if not comparison_path.exists():
        print(f"Error: {comparison_path} not found. Run compare_tokenizers_evaluation.py first.")
        return

    with open(comparison_path, "r") as f:
        results = json.load(f)

    # Start building the report
    report = []
    report.append("# Tokenization Efficiency Benchmark - Lab Report")
    report.append("")
    report.append("## 1. Introduction")
    report.append("")
    report.append("This report compares three tokenization strategies for Polish language modeling:")
    report.append("1. **GPT-2 BPE tokenizer** (pre-trained, 50,257 tokens)")
    report.append("2. **SentencePiece tokenizer** (custom-trained, 10,000 tokens)")
    report.append("3. **Whitespace tokenizer** (custom implementation, 10,000 tokens)")
    report.append("")
    report.append("All three models use identical Transformer architecture and training procedures,")
    report.append("differing only in the tokenization strategy.")
    report.append("")

    # Model Architecture
    report.append("## 2. Model Architecture")
    report.append("")
    report.append("**Transformer Language Model:**")
    report.append(f"- Layers: {config.num_layers}")
    report.append(f"- Embedding dimension: {config.embedding_dim}")
    report.append(f"- Attention heads: {config.num_heads}")
    report.append(f"- Feed-forward dimension: {config.ff_dim}")
    report.append(f"- Dropout: {config.dropout}")
    report.append(f"- Max sequence length: {config.max_seq_length}")
    report.append(f"- Estimated parameters: ~50M")
    report.append("")

    # Training Configuration
    report.append("## 3. Training Configuration")
    report.append("")
    report.append(f"- Batch size: {config.batch_size}")
    report.append(f"- Learning rate: {config.learning_rate}")
    report.append(f"- Epochs trained: 25")
    report.append(f"- Optimizer: AdamW")
    report.append(f"- Mixed precision (AMP): {config.use_amp}")
    report.append(f"- Gradient accumulation: {config.gradient_accumulation_steps}")
    report.append("")

    # Hardware Specification
    report.append("## 4. Hardware Specification")
    report.append("")
    report.append(f"- **Training:** NVIDIA GeForce RTX 5090 (32GB VRAM) on RunPod")
    report.append(f"- **Dataset:** Polish gardening forum corpus")
    report.append(f"- **Optimization:** Multi-worker data loading, pinned memory, TF32")
    report.append("")

    # Quantitative Results
    report.append("## 5. Quantitative Results")
    report.append("")

    # Perplexity Table
    report.append("### 5.1 Perplexity Comparison")
    report.append("")
    report.append("| Tokenizer | Word-Level PPL | Character-Level PPL | Token-Level PPL* |")
    report.append("|-----------|----------------|---------------------|------------------|")

    for tok_type in ["gpt2", "sentencepiece", "whitespace"]:
        tok_data = results[tok_type]
        report.append(f"| {tok_type.upper()} | "
                     f"{tok_data['word_level_perplexity']:.2f} | "
                     f"{tok_data['character_level_perplexity']:.2f} | "
                     f"{tok_data['best_val_perplexity_token_level']:.2f} |")

    report.append("")
    report.append("*Token-level perplexity is shown for reference but is NOT comparable across tokenizers.")
    report.append("")

    # OOV Statistics
    report.append("### 5.2 Out-of-Vocabulary (OOV) Statistics")
    report.append("")
    report.append("| Tokenizer | OOV Words | Total Words | OOV % |")
    report.append("|-----------|-----------|-------------|-------|")

    for tok_type in ["gpt2", "sentencepiece", "whitespace"]:
        tok_data = results[tok_type]
        oov = tok_data['oov_statistics']
        report.append(f"| {tok_type.upper()} | "
                     f"{oov['oov_words']} | "
                     f"{oov['total_words']} | "
                     f"{oov['oov_percentage']:.2f}% |")

    report.append("")
    report.append("*Note: GPT-2 and SentencePiece use subword tokenization, so they don't have true OOV words.*")
    report.append("")

    # Tokenization Efficiency
    report.append("### 5.3 Tokenization Efficiency")
    report.append("")
    report.append("| Tokenizer | Tokens/Word | Direct Encoding % | Vocab Size |")
    report.append("|-----------|-------------|-------------------|------------|")

    for tok_type in ["gpt2", "sentencepiece", "whitespace"]:
        tok_data = results[tok_type]
        report.append(f"| {tok_type.upper()} | "
                     f"{tok_data['tokens_per_word']:.2f} | "
                     f"{tok_data['words_encoded_directly']['percentage']:.1f}% | "
                     f"{tok_data['vocab_size']:,} |")

    report.append("")

    # Training and Inference Speed
    report.append("### 5.4 Training and Inference Performance")
    report.append("")
    report.append("| Tokenizer | Avg Training Time/Epoch | Total Training Time | Inference Speed |")
    report.append("|-----------|-------------------------|---------------------|-----------------|")

    for tok_type in ["gpt2", "sentencepiece", "whitespace"]:
        tok_data = results[tok_type]
        avg_epoch_time = tok_data['training_time_avg_epoch_seconds']
        total_time = tok_data['training_time_total_seconds']
        inference_speed = tok_data['inference_speed_tokens_per_sec']

        report.append(f"| {tok_type.upper()} | "
                     f"{avg_epoch_time:.1f}s | "
                     f"{total_time/60:.1f}min | "
                     f"{inference_speed:.0f} tok/s |")

    report.append("")

    # Qualitative Analysis
    report.append("## 6. Qualitative Analysis")
    report.append("")
    report.append("See `results/qualitative_analysis.txt` for detailed tokenization examples.")
    report.append("")
    report.append("**Key Observations:**")
    report.append("")
    report.append("1. **GPT-2 tokenizer** splits Polish words into many subword tokens (3.22-3.90 tokens/word)")
    report.append("   - Trained on English text, so Polish morphology is not well represented")
    report.append("   - Only 10-27% of words encoded as single tokens")
    report.append("   - Despite this, achieves best perplexity due to large vocabulary (50K)")
    report.append("")
    report.append("2. **SentencePiece tokenizer** provides balanced performance (1.30-1.72 tokens/word)")
    report.append("   - Custom-trained on Polish text")
    report.append("   - 58-79% of words encoded as single tokens")
    report.append("   - Good balance between vocabulary size and coverage")
    report.append("")
    report.append("3. **Whitespace tokenizer** is most efficient for encoding (1.14-1.15 tokens/word)")
    report.append("   - 78-83% of words encoded as single tokens")
    report.append("   - However, suffers from limited vocabulary (10K) and OOV issues")
    report.append("   - Worst perplexity despite efficient tokenization")
    report.append("")

    # Discussion
    report.append("## 7. Discussion")
    report.append("")
    report.append("### Trade-offs")
    report.append("")
    report.append("**GPT-2 (Winner):**")
    report.append("- ✅ Best perplexity (30.95 word-level)")
    report.append("- ✅ No OOV issues (subword tokenization)")
    report.append("- ✅ Large vocabulary captures more patterns")
    report.append("- ❌ Inefficient for Polish (3.9 tokens/word)")
    report.append("- ❌ Longer sequences → more memory, slower inference")
    report.append("")
    report.append("**SentencePiece (2nd):**")
    report.append("- ✅ Balanced efficiency (1.5 tokens/word)")
    report.append("- ✅ Custom-trained on Polish")
    report.append("- ✅ Reasonable perplexity (154.58)")
    report.append("- ❌ Smaller vocabulary limits expressiveness")
    report.append("")
    report.append("**Whitespace (3rd):**")
    report.append("- ✅ Most efficient encoding (1.14 tokens/word)")
    report.append("- ✅ Fastest tokenization")
    report.append("- ❌ Worst perplexity (109.57)")
    report.append("- ❌ OOV problems with rare words")
    report.append("- ❌ Limited vocabulary (10K)")
    report.append("")

    # Conclusions
    report.append("## 8. Conclusions")
    report.append("")
    report.append("1. **Vocabulary size matters more than tokenization efficiency**")
    report.append("   - GPT-2's 50K vocabulary outperforms despite being inefficient for Polish")
    report.append("   - Word-level perplexity: GPT-2 (30.95) vs SentencePiece (154.58) vs Whitespace (109.57)")
    report.append("")
    report.append("2. **Subword tokenization eliminates OOV issues**")
    report.append("   - Both GPT-2 and SentencePiece handle unknown words gracefully")
    report.append("   - Whitespace tokenizer struggles with vocabulary coverage")
    report.append("")
    report.append("3. **Training on target language improves efficiency but not necessarily accuracy**")
    report.append("   - SentencePiece was trained on Polish but has lower perplexity than GPT-2")
    report.append("   - Suggests that vocabulary size is more important than language-specific training")
    report.append("")
    report.append("4. **For production use, GPT-2 tokenizer is recommended**")
    report.append("   - Best perplexity and no OOV issues")
    report.append("   - Trade-off: higher memory/compute due to longer sequences")
    report.append("")

    # References
    report.append("## 9. References")
    report.append("")
    report.append("- Kudo, Taku. \"SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.\" EMNLP 2018")
    report.append("- Sennrich, Haddow, and Birch. \"Neural Machine Translation of Rare Words with Subword Units.\" ACL 2016")
    report.append("- Mielke, Sabrina. \"Comparing perplexities\" https://sjmielke.com/comparing-perplexities.htm")
    report.append("")

    # Save report
    report_text = "\n".join(report)

    output_path = config.results_dir / "LAB_REPORT.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report generated: {output_path}")

    # Also print to console
    print("\n" + "="*80)
    print(report_text)
    print("="*80)

    return output_path


if __name__ == "__main__":
    generate_markdown_report()
