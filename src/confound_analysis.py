"""
Confound analysis: check whether the 'AI direction' is simply capturing
text length, formality, or other surface-level features rather than
genuine AI style.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = RESULTS_DIR / "plots"


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def compute_length_direction(acts_by_layer, lengths, layer):
    """Compute a direction that predicts text length using median split."""
    acts = acts_by_layer[layer]
    median_len = np.median(lengths)
    short_mask = np.array(lengths) <= median_len
    long_mask = ~short_mask

    short_mean = acts[short_mask].mean(dim=0)
    long_mean = acts[long_mask].mean(dim=0)
    direction = long_mean - short_mean
    return direction / direction.norm()


def compute_perplexity_proxy(texts, tokenizer, model):
    """Not used directly — we use text length as a simpler proxy since
    AI text tends to be longer and more verbose."""
    pass


def analyze_confounds(best_layer):
    """Analyze whether the AI direction correlates with length confounds."""
    # Load test data
    test_data = torch.load(RESULTS_DIR / "test_activations.pt", weights_only=True)
    test_human = test_data["human"]
    test_ai = test_data["ai"]

    # Load test pairs for text metadata
    with open(RESULTS_DIR / "test_pairs.json") as f:
        test_pairs = json.load(f)

    # Load the AI direction
    ai_direction = torch.load(RESULTS_DIR / "best_direction.pt", weights_only=True)

    # Compute text lengths
    human_lens = [len(p["human_text"]) for p in test_pairs]
    ai_lens = [len(p["ai_text"]) for p in test_pairs]

    print("=== Confound Analysis ===")
    print(f"\nText Length Statistics:")
    print(f"  Human: mean={np.mean(human_lens):.0f}, std={np.std(human_lens):.0f}")
    print(f"  AI:    mean={np.mean(ai_lens):.0f}, std={np.std(ai_lens):.0f}")

    # Compute length direction using combined data
    all_acts = {layer: torch.cat([test_human[layer], test_ai[layer]], dim=0)
                for layer in test_human}
    all_lengths = human_lens + ai_lens

    length_dir = compute_length_direction(all_acts, all_lengths, best_layer)

    # Cosine similarity between AI direction and length direction
    cos_sim_length = cosine_similarity(ai_direction, length_dir)
    print(f"\nCosine similarity between AI direction and length direction: {cos_sim_length:.4f}")

    # Compute AI direction accuracy after projecting out length direction
    # Ablated direction = AI direction - (AI_dir . length_dir) * length_dir
    ablated_direction = ai_direction - cos_sim_length * length_dir
    ablated_direction = ablated_direction / ablated_direction.norm()

    # Test ablated direction accuracy
    midpoint = (test_ai[best_layer].mean(dim=0) + test_human[best_layer].mean(dim=0)) / 2
    human_proj = ((test_human[best_layer] - midpoint) @ ablated_direction).numpy()
    ai_proj = ((test_ai[best_layer] - midpoint) @ ablated_direction).numpy()
    all_proj = np.concatenate([human_proj, ai_proj])
    all_labels = np.array([0] * len(human_proj) + [1] * len(ai_proj))
    preds = (all_proj > 0).astype(int)
    ablated_acc = (preds == all_labels).mean()

    # Original direction accuracy for comparison
    human_proj_orig = ((test_human[best_layer] - midpoint) @ ai_direction).numpy()
    ai_proj_orig = ((test_ai[best_layer] - midpoint) @ ai_direction).numpy()
    all_proj_orig = np.concatenate([human_proj_orig, ai_proj_orig])
    preds_orig = (all_proj_orig > 0).astype(int)
    orig_acc = (preds_orig == all_labels).mean()

    print(f"\nOriginal AI direction accuracy: {orig_acc:.3f}")
    print(f"After removing length component: {ablated_acc:.3f}")
    print(f"Accuracy drop from ablating length: {orig_acc - ablated_acc:.3f}")

    # Within-class analysis: does the AI direction separate short vs long texts
    # within the same class?
    print("\n--- Within-class Length Analysis ---")
    human_ai_proj = ((test_human[best_layer] - midpoint) @ ai_direction).numpy()
    ai_ai_proj = ((test_ai[best_layer] - midpoint) @ ai_direction).numpy()

    # Correlation between text length and projection onto AI direction
    human_corr = np.corrcoef(human_lens, human_ai_proj)[0, 1]
    ai_corr = np.corrcoef(ai_lens, ai_ai_proj)[0, 1]
    print(f"Correlation (human text length vs AI projection): {human_corr:.3f}")
    print(f"Correlation (AI text length vs AI projection):    {ai_corr:.3f}")

    # Word count analysis
    human_word_counts = [len(p["human_text"].split()) for p in test_pairs]
    ai_word_counts = [len(p["ai_text"].split()) for p in test_pairs]
    print(f"\nWord Count Statistics:")
    print(f"  Human: mean={np.mean(human_word_counts):.0f}, std={np.std(human_word_counts):.0f}")
    print(f"  AI:    mean={np.mean(ai_word_counts):.0f}, std={np.std(ai_word_counts):.0f}")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Projection distributions
    ax = axes[0]
    ax.hist(human_ai_proj, bins=30, alpha=0.5, label="Human", color="blue", density=True)
    ax.hist(ai_ai_proj, bins=30, alpha=0.5, label="AI", color="red", density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Projection onto AI Direction")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Projections")
    ax.legend()

    # 2. Length vs projection scatter
    ax = axes[1]
    ax.scatter(human_lens, human_ai_proj, alpha=0.3, s=15, label="Human", c="blue")
    ax.scatter(ai_lens, ai_ai_proj, alpha=0.3, s=15, label="AI", c="red")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Projection onto AI Direction")
    ax.set_title("Length vs AI Projection")
    ax.legend()

    # 3. Cosine similarity with various directions
    ax = axes[2]
    # Random directions for comparison
    n_random = 100
    dim = ai_direction.shape[0]
    random_sims = []
    for _ in range(n_random):
        rand_dir = torch.randn(dim)
        rand_dir = rand_dir / rand_dir.norm()
        random_sims.append(abs(cosine_similarity(ai_direction, rand_dir)))

    categories = ["Length\nDirection", f"Random\n(n={n_random})"]
    values = [abs(cos_sim_length), np.mean(random_sims)]
    errors = [0, np.std(random_sims)]
    colors = ["orange", "gray"]

    bars = ax.bar(categories, values, color=colors, yerr=errors, capsize=5)
    ax.set_ylabel("|Cosine Similarity|")
    ax.set_title("AI Direction Overlap with Confounds")
    ax.set_ylim(0, max(values) * 1.3 + 0.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confound_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confound analysis plot to {PLOT_DIR / 'confound_analysis.png'}")

    # Save results
    confound_results = {
        "cosine_sim_ai_vs_length": cos_sim_length,
        "original_accuracy": float(orig_acc),
        "accuracy_after_removing_length": float(ablated_acc),
        "accuracy_drop": float(orig_acc - ablated_acc),
        "human_length_ai_corr": float(human_corr),
        "ai_length_ai_corr": float(ai_corr),
        "random_cosine_sim_mean": float(np.mean(random_sims)),
        "random_cosine_sim_std": float(np.std(random_sims)),
        "human_text_len_mean": float(np.mean(human_lens)),
        "ai_text_len_mean": float(np.mean(ai_lens)),
    }
    with open(RESULTS_DIR / "confound_results.json", "w") as f:
        json.dump(confound_results, f, indent=2)
    print(f"Saved confound results to {RESULTS_DIR / 'confound_results.json'}")


def main():
    with open(RESULTS_DIR / "direction_results.json") as f:
        dir_results = json.load(f)
    best_layer = dir_results["best_layer"]
    print(f"Using best layer: {best_layer}")

    analyze_confounds(best_layer)


if __name__ == "__main__":
    main()
