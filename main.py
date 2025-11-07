"""
Download and prepare Polish datasets from Speakleash for language model training.

Usage:
    python main.py <dataset_name>
    python main.py shopping_1_general_corpus
    python main.py --list  # List all available datasets
"""

from speakleash import Speakleash
import sys
import argparse
from pathlib import Path
from tqdm import tqdm


def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent
    data_raw_dir = base_dir / "data" / "raw"
    metadata_dir = base_dir / "speakleash_datasets_metadata"

    data_raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return base_dir, data_raw_dir, metadata_dir


def list_datasets(sl: Speakleash, category: str = None):
    """
    List available datasets with their information.

    Args:
        sl: Speakleash instance
        category: Optional category filter (e.g., 'Internet', 'Literature')
    """
    print("=" * 80)
    print("AVAILABLE SPEAKLEASH DATASETS")
    print("=" * 80)

    datasets = sl.populate_datasets()

    if not datasets:
        print("No datasets found!")
        return

    for dataset in datasets:
        manifest = dataset.manifest

        # Filter by category if specified
        if category and manifest.get('category') != category:
            continue

        print(f"\nName: {manifest['name']}")
        print(f"Category: {manifest.get('category', 'N/A')}")
        print(f"Size: {manifest.get('file_size', 0) / (1024**3):.2f} GB")
        print(f"Documents: {manifest.get('stats', {}).get('documents', 'N/A'):,}")
        print(f"Words: {manifest.get('stats', {}).get('words', 'N/A'):,}")
        print(f"License: {manifest.get('license', 'N/A')}")
        print(f"Description: {manifest.get('description', 'N/A')}")
        print("-" * 80)


def download_dataset(dataset_name: str, output_dir: Path, metadata_dir: Path):
    """
    Download a dataset from Speakleash and save it to data/raw/ directory.

    Args:
        dataset_name: Name of the dataset to download
        output_dir: Directory to save the dataset
        metadata_dir: Directory for Speakleash metadata
    """
    print("=" * 80)
    print(f"DOWNLOADING DATASET: {dataset_name}")
    print("=" * 80)

    # Initialize Speakleash
    print(f"\nInitializing Speakleash (metadata dir: {metadata_dir})...")
    sl = Speakleash(str(metadata_dir))

    # Get dataset
    print(f"Fetching dataset information...")
    dataset = sl.get(dataset_name)

    if dataset is None:
        print(f"\nError: Dataset '{dataset_name}' not found!")
        print("\nUse 'python main.py --list' to see available datasets.")
        return False

    # Display dataset information
    manifest = dataset.manifest
    print(f"\nDataset Information:")
    print(f"  Name: {manifest['name']}")
    print(f"  Category: {manifest.get('category', 'N/A')}")
    print(f"  Size: {manifest.get('file_size', 0) / (1024**3):.2f} GB")
    print(f"  Documents: {manifest.get('stats', {}).get('documents', 'N/A'):,}")
    print(f"  Sentences: {manifest.get('stats', {}).get('sentences', 'N/A'):,}")
    print(f"  Words: {manifest.get('stats', {}).get('words', 'N/A'):,}")
    print(f"  License: {manifest.get('license', 'N/A')}")

    if manifest.get('disclaimer'):
        print(f"\nDISCLAIMER:")
        print(f"  {manifest['disclaimer'][:200]}...")

    # Confirm download
    response = input(f"\nProceed with download? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Download cancelled.")
        return False

    # Output file path
    output_file = output_dir / f"{dataset_name}.txt"

    print(f"\nDownloading and processing dataset...")
    print(f"Output file: {output_file}")
    print("\nThis may take a while depending on dataset size...")
    print("-" * 80)

    try:
        # Download and iterate through data
        # The .data() method automatically downloads the dataset if not cached
        document_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            # Use .data() which returns an iterator over documents
            for document in tqdm(dataset.data, desc="Processing documents", unit="docs"):
                # Each document is a dictionary with 'text' field
                if isinstance(document, dict) and 'text' in document:
                    text = document['text'].strip()
                    if text:  # Only write non-empty documents
                        f.write(text + '\n')
                        document_count += 1
                elif isinstance(document, str):
                    text = document.strip()
                    if text:
                        f.write(text + '\n')
                        document_count += 1

        print("\n" + "=" * 80)
        print("DOWNLOAD COMPLETE")
        print("=" * 80)
        print(f"Documents saved: {document_count:,}")
        print(f"Output file: {output_file}")
        print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")
        print("\nNext steps:")
        print(f"  python scripts/preprocess_data.py --input {output_file} --file-type txt")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nError during download: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Polish datasets from Speakleash for language model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python main.py --list

  # Download a specific dataset
  python main.py shopping_1_general_corpus

  # List datasets by category
  python main.py --list --category Internet
        """
    )

    parser.add_argument(
        'dataset_name',
        nargs='?',
        help='Name of the dataset to download (e.g., shopping_1_general_corpus)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available datasets'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Filter datasets by category (use with --list)'
    )

    args = parser.parse_args()

    # Setup directories
    base_dir, data_raw_dir, metadata_dir = setup_directories()

    # List datasets
    if args.list:
        sl = Speakleash(str(metadata_dir))
        list_datasets(sl, args.category)
        return

    # Download dataset
    if args.dataset_name:
        success = download_dataset(args.dataset_name, data_raw_dir, metadata_dir)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        print("\nError: Please provide a dataset name or use --list to see available datasets.")
        sys.exit(1)


if __name__ == "__main__":
    main()
