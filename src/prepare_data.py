"""
Data preparation for the 'sounds like AI' direction experiment.
Loads HC3 dataset and creates contrastive pairs of human vs AI text.
"""

import json
import random
import numpy as np
from datasets import load_from_disk
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "datasets" / "data" / "hc3"
OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_filter_hc3(min_len=50, max_len=500):
    """Load HC3 and create paired (human_text, ai_text) samples.

    Each sample is a pair of answers to the same question — one human, one ChatGPT.
    We filter by character length to ensure manageable token counts.
    """
    ds = load_from_disk(str(DATA_DIR))["train"]

    pairs = []
    for row in ds:
        human_answers = row["human_answers"]
        chatgpt_answers = row["chatgpt_answers"]
        question = row["question"]
        source = row.get("source", "unknown")

        if not human_answers or not chatgpt_answers:
            continue

        # Take first human and first ChatGPT answer
        human_text = human_answers[0].strip()
        ai_text = chatgpt_answers[0].strip()

        # Filter by length (character count)
        if len(human_text) < min_len or len(human_text) > max_len * 3:
            continue
        if len(ai_text) < min_len or len(ai_text) > max_len * 3:
            continue

        pairs.append({
            "question": question,
            "human_text": human_text,
            "ai_text": ai_text,
            "source": source,
        })

    return pairs


def create_splits(pairs, n_train=200, n_val=50, n_test=100):
    """Create train/val/test splits with topic diversity."""
    random.shuffle(pairs)

    total_needed = n_train + n_val + n_test
    if len(pairs) < total_needed:
        print(f"Warning: Only {len(pairs)} pairs available, need {total_needed}")
        # Adjust proportions
        ratio = len(pairs) / total_needed
        n_train = int(n_train * ratio)
        n_val = int(n_val * ratio)
        n_test = len(pairs) - n_train - n_val

    train = pairs[:n_train]
    val = pairs[n_train:n_train + n_val]
    test = pairs[n_train + n_val:n_train + n_val + n_test]

    return train, val, test


def compute_stats(pairs, label=""):
    """Print basic statistics about the pairs."""
    human_lens = [len(p["human_text"]) for p in pairs]
    ai_lens = [len(p["ai_text"]) for p in pairs]

    print(f"\n{label} Statistics ({len(pairs)} pairs):")
    print(f"  Human text length: mean={np.mean(human_lens):.0f}, "
          f"std={np.std(human_lens):.0f}, "
          f"min={np.min(human_lens)}, max={np.max(human_lens)}")
    print(f"  AI text length:    mean={np.mean(ai_lens):.0f}, "
          f"std={np.std(ai_lens):.0f}, "
          f"min={np.min(ai_lens)}, max={np.max(ai_lens)}")

    # Source distribution
    sources = {}
    for p in pairs:
        s = p["source"]
        sources[s] = sources.get(s, 0) + 1
    print(f"  Sources: {sources}")


def main():
    print("Loading HC3 dataset...")
    pairs = load_and_filter_hc3()
    print(f"Found {len(pairs)} valid pairs after filtering")

    train, val, test = create_splits(pairs)

    compute_stats(train, "Train")
    compute_stats(val, "Validation")
    compute_stats(test, "Test")

    # Save splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"{name}_pairs.json"
        with open(path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Saved {name} split to {path}")

    # Show example pairs
    print("\n--- Example Pairs ---")
    for i in range(min(3, len(train))):
        pair = train[i]
        print(f"\nPair {i+1}:")
        print(f"  Q: {pair['question'][:80]}...")
        print(f"  Human: {pair['human_text'][:120]}...")
        print(f"  AI:    {pair['ai_text'][:120]}...")


if __name__ == "__main__":
    main()
