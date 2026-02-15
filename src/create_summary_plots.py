"""
Create summary plots for the steering experiment results.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = RESULTS_DIR / "plots"


def plot_steering_scores():
    """Plot LLM judge scores by multiplier."""
    with open(RESULTS_DIR / "scored_steering_results.json") as f:
        results = json.load(f)

    # Group by multiplier
    by_mult = {}
    for r in results:
        m = r["multiplier"]
        if r.get("ai_score") is not None:
            by_mult.setdefault(m, []).append(r["ai_score"])

    mults = sorted(by_mult.keys())
    means = [np.mean(by_mult[m]) for m in mults]
    stds = [np.std(by_mult[m]) for m in mults]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(mults)), means, yerr=stds, capsize=5,
           color=["#2196F3", "#64B5F6", "#9E9E9E", "#EF9A9A", "#F44336"])
    ax.set_xticks(range(len(mults)))
    ax.set_xticklabels([f"{m:+.0f}\n({'human' if m<0 else 'AI' if m>0 else 'base'})"
                        for m in mults], fontsize=9)
    ax.set_ylabel("AI-Likeness Score (1-7)")
    ax.set_xlabel("Steering Multiplier")
    ax.set_title("LLM Judge: AI-Likeness Score vs Steering Direction")
    ax.set_ylim(1, 7.5)
    ax.axhline(4, color="gray", linestyle="--", alpha=0.3, label="Ambiguous (4)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    for i, (m, v) in enumerate(zip(mults, means)):
        ax.text(i, v + 0.2, f"{v:.1f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "steering_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved steering scores plot to {PLOT_DIR / 'steering_scores.png'}")


def create_steering_examples_table():
    """Create a text summary of the best steering examples."""
    with open(RESULTS_DIR / "steering_results.json") as f:
        results = json.load(f)

    # Select climate change examples (clearest contrast)
    print("\n=== Steering Examples (Climate Change Prompt) ===")
    for r in results:
        if r["prompt_label"] == "Climate change":
            m = r["multiplier"]
            text = r["generated_text"][:200]
            label = "HUMAN-LIKE" if m < 0 else ("AI-LIKE" if m > 0 else "BASELINE")
            print(f"\n[{label}, mult={m:+.1f}]:")
            print(f"  {text}")


def plot_combined_summary():
    """Create a combined summary figure."""
    # Load all results
    with open(RESULTS_DIR / "direction_results.json") as f:
        dir_results = json.load(f)
    with open(RESULTS_DIR / "confound_results.json") as f:
        confound_results = json.load(f)
    with open(RESULTS_DIR / "length_controlled_results.json") as f:
        length_results = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Accuracy by layer
    ax = axes[0, 0]
    layers = sorted([int(k) for k in dir_results["layer_results"].keys()])
    test_accs = [dir_results["layer_results"][str(l)]["test_acc"] for l in layers]
    test_aucs = [dir_results["layer_results"][str(l)]["test_auc"] for l in layers]
    ax.plot(layers, test_accs, "b-o", markersize=3, label="Test Accuracy")
    ax.plot(layers, test_aucs, "g-^", markersize=3, label="Test AUC")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title("A. Mass-Mean Probe: AI vs Human Classification")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    # Panel B: Length disentangled
    ax = axes[0, 1]
    layers_lc = sorted([int(k) for k in length_results["layer_results_controlled"].keys()])
    orig = [length_results["layer_results_controlled"][str(l)]["original_acc"] for l in layers_lc]
    orth = [length_results["layer_results_controlled"][str(l)]["length_orthogonal_acc"] for l in layers_lc]
    lenonly = [length_results["layer_results_controlled"][str(l)]["length_only_acc"] for l in layers_lc]
    ax.plot(layers_lc, orig, "b-o", markersize=3, label="Original Direction")
    ax.plot(layers_lc, orth, "r-s", markersize=3, label="Length-Orthogonal")
    ax.plot(layers_lc, lenonly, "g-^", markersize=3, label="Length Only")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("B. Disentangling Length from AI Style")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    # Panel C: Cross-layer consistency
    ax = axes[1, 0]
    cos_sims = dir_results["cross_layer_cos_sims"]
    ax.plot(range(1, len(cos_sims)+1), cos_sims, "k-o", markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity with Previous Layer")
    ax.set_title("C. Direction Consistency Across Layers")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # Panel D: Steering scores
    ax = axes[1, 1]
    try:
        with open(RESULTS_DIR / "scored_steering_results.json") as f:
            scored_results = json.load(f)
        by_mult = {}
        for r in scored_results:
            m = r["multiplier"]
            if r.get("ai_score") is not None:
                by_mult.setdefault(m, []).append(r["ai_score"])
        mults = sorted(by_mult.keys())
        means = [np.mean(by_mult[m]) for m in mults]
        stds = [np.std(by_mult[m]) for m in mults]
        colors = ["#2196F3", "#64B5F6", "#9E9E9E", "#EF9A9A", "#F44336"]
        bars = ax.bar(range(len(mults)), means, yerr=stds, capsize=5, color=colors)
        ax.set_xticks(range(len(mults)))
        ax.set_xticklabels([f"{m:+.0f}" for m in mults], fontsize=9)
        ax.set_ylabel("AI-Likeness Score (1-7)")
        ax.set_xlabel("Steering Multiplier")
        ax.set_title("D. LLM Judge: Steering Effects")
        ax.set_ylim(1, 7.5)
        ax.axhline(4, color="gray", linestyle="--", alpha=0.3)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                    f"{val:.1f}", ha='center', fontsize=9, fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, f"Scoring data unavailable\n{e}", ha='center', va='center',
                transform=ax.transAxes)

    plt.suptitle("Is There a 'Sounds Like AI' Direction in the Residual Stream?",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "summary_figure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary figure to {PLOT_DIR / 'summary_figure.png'}")


if __name__ == "__main__":
    plot_steering_scores()
    create_steering_examples_table()
    plot_combined_summary()
