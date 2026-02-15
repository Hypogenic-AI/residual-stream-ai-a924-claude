"""
Download recommended datasets for the "sounds like AI" direction research.

Usage:
    pip install datasets
    python download_datasets.py [--dataset NAME] [--all]

Recommended datasets are ordered by relevance to our research question.
"""

import argparse
from pathlib import Path

DATASETS = {
    "hc3": {
        "hf_id": "Hello-SimpleAI/HC3",
        "description": "Human ChatGPT Comparison Corpus - paired human/ChatGPT answers to same questions",
        "size": "~37k QA pairs",
        "license": "CC-BY-SA-4.0",
        "why": "Best for contrastive pairs: same question answered by both human and ChatGPT",
    },
    "dmitva_paired": {
        "hf_id": "dmitva/human_ai_generated_text",
        "description": "1M paired human/AI texts with same instructions",
        "size": "~1M rows",
        "license": "CC-BY-4.0",
        "why": "Large-scale paired dataset ideal for computing difference-of-means vectors",
    },
    "ai_text_pile": {
        "hf_id": "artem9k/ai-text-detection-pile",
        "description": "1.39M texts from human + GPT-2/3/J/ChatGPT",
        "size": "~1.39M rows",
        "license": "MIT",
        "why": "Large, multi-model AI text corpus for training/validation",
    },
    "ai_text_pile_cleaned": {
        "hf_id": "srikanthgali/ai-text-detection-pile-cleaned",
        "description": "Cleaned version of ai-text-detection-pile with balanced classes",
        "size": "~722k rows",
        "license": "MIT",
        "why": "Preprocessed version with deduplication and balanced distribution",
    },
    "raid": {
        "hf_id": "liamdugan/raid",
        "description": "10M+ docs from 12 LLMs across 4 domains with adversarial attacks",
        "size": "~10M rows",
        "license": "MIT",
        "why": "Most comprehensive benchmark; tests generalization across models/domains",
    },
    "ai_human_text": {
        "hf_id": "andythetechnerd03/AI-human-text",
        "description": "58k+ NYT articles paired with synthetic versions from 6 LLMs",
        "size": "~58k rows",
        "license": "Unknown",
        "why": "Multi-model generation from same source articles (arxiv 2510.22874)",
    },
}


def download_dataset(name: str, output_dir: Path) -> None:
    from datasets import load_dataset

    info = DATASETS[name]
    print(f"\nDownloading {name}: {info['hf_id']}")
    print(f"  Description: {info['description']}")
    print(f"  Size: {info['size']}")

    ds = load_dataset(info["hf_id"])
    save_path = output_dir / name
    save_path.mkdir(exist_ok=True)
    ds.save_to_disk(str(save_path))
    print(f"  Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for AI text direction research")
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), help="Download a specific dataset")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name}")
            print(f"    HF: {info['hf_id']}")
            print(f"    Size: {info['size']} | License: {info['license']}")
            print(f"    Why: {info['why']}")
            print()
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.all:
        for name in DATASETS:
            download_dataset(name, output_dir)
    elif args.dataset:
        download_dataset(args.dataset, output_dir)
    else:
        # Default: download the two most relevant paired datasets
        print("No dataset specified. Downloading recommended paired datasets (hc3, dmitva_paired).")
        print("Use --all for all datasets or --list to see options.\n")
        for name in ["hc3", "dmitva_paired"]:
            download_dataset(name, output_dir)


if __name__ == "__main__":
    main()
