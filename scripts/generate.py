"""
Text generation script for trained transformer language model.
"""

import sys
from pathlib import Path
import torch
import argparse
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_config
from utils.tokenizer import PolishTokenizer
from utils.tokenizer_factory import create_tokenizer
from models.transformer_model import TransformerLanguageModel
import json


def generate_text(
    checkpoint_path: str,
    prompts: list[str],
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    dataset: str = None,
):
    """
    Generate text completions for given prompts.

    Args:
        checkpoint_path: Path to model checkpoint
        prompts: List of prompt strings
        max_length: Maximum length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        dataset: Dataset name (auto-detected if not specified)
    """
    print("=" * 80)
    print("TEXT GENERATION - TRANSFORMER MODEL")
    print("=" * 80)

    # Load configuration
    config = get_config("transformer")

    # Auto-detect dataset if not specified
    if dataset is None:
        metadata_files = list(config.data_processed_dir.glob("*_metadata.json"))
        if not metadata_files:
            raise ValueError("No preprocessed dataset found.")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple datasets found. Please specify --dataset")
        dataset = metadata_files[0].stem.replace('_metadata', '')
        print(f"\nAuto-detected dataset: {dataset}")

    config.dataset_name = dataset
    print(f"Dataset: {dataset}")
    print(f"Device: {config.device}")

    # Load metadata to get tokenizer type
    metadata_path = config.data_processed_dir / f"{dataset}_metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        tokenizer_type = metadata.get("tokenizer_type", "bpe")

    # Load tokenizer
    print(f"\nLoading tokenizer (type: {tokenizer_type})...")
    tokenizer = create_tokenizer(tokenizer_type, vocab_size=config.vocab_size)

    # SentencePiece uses .json, others may use different extensions
    if tokenizer_type == "sentencepiece":
        tokenizer_path = config.data_processed_dir / f"{dataset}_tokenizer.json"
    else:
        tokenizer_path = config.data_processed_dir / f"{dataset}_tokenizer.json"

    tokenizer.load(tokenizer_path)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    # Create model
    print("Initializing model...")
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

    print(f"Model parameters: {model.get_num_parameters():,}")

    # Generate for each prompt
    print("\n" + "=" * 80)
    print("GENERATING TEXT")
    print("=" * 80)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i + 1}/{len(prompts)}:")
        print(f"  Input: \"{prompt}\"")

        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(config.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                input_tensor,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_time = time.time() - start_time

        # Decode
        generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

        # Extract only the generated part (after the prompt)
        # This is approximate since tokenization might differ
        completion = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text

        print(f"  Output: \"{generated_text}\"")
        print(f"  Time: {gen_time:.3f}s")

        results.append({
            "prompt": prompt,
            "completion": completion,
            "full_text": generated_text,
            "time": gen_time,
        })

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    results_path = config.results_dir / f"{dataset}_transformer_generations.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Text Generation Results - TRANSFORMER Model\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Max length: {max_length}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Top-k: {top_k}\n")
        f.write("=" * 80 + "\n\n")

        for i, result in enumerate(results):
            f.write(f"PROMPT {i + 1}:\n")
            f.write(f"{result['prompt']}\n\n")
            f.write(f"COMPLETION:\n")
            f.write(f"{result['full_text']}\n\n")
            f.write(f"Time: {result['time']:.3f}s\n")
            f.write("-" * 80 + "\n\n")

    print(f"Results saved to: {results_path}")

    # Also save as JSON for programmatic access
    json_path = config.results_dir / f"{dataset}_transformer_generations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"JSON results saved to: {json_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained transformer model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        help="Text prompts for generation",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="File containing prompts (one per line)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum length to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Auto-detected if only one dataset exists.",
    )

    args = parser.parse_args()

    # Get prompts from either command line or file
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts:
        prompts = args.prompts
    else:
        # Default Polish prompts
        prompts = [
            "Warszawa jest",
            "W Polsce",
            "Dzisiaj pogoda",
            "Nauka języków obcych",
            "Sztuczna inteligencja",
        ]
        print("No prompts specified, using defaults...")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    generate_text(
        str(checkpoint_path),
        prompts,
        args.max_length,
        args.temperature,
        args.top_k,
        args.dataset,
    )


if __name__ == "__main__":
    main()
