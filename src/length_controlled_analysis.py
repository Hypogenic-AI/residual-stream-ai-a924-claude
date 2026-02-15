"""
Length-controlled analysis: re-extract the AI direction after matching
human and AI texts by length to isolate the 'style' component from the
'length' component.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = RESULTS_DIR / "plots"


def length_matched_direction(train_pairs, train_human_acts, train_ai_acts, layer,
                             length_tolerance=0.3):
    """Compute AI direction using only length-matched pairs.

    For each pair, keep it only if the human and AI text are within
    length_tolerance of each other (ratio).
    """
    matched_indices = []
    for i, pair in enumerate(train_pairs):
        h_len = len(pair["human_text"])
        a_len = len(pair["ai_text"])
        ratio = max(h_len, a_len) / max(min(h_len, a_len), 1)
        if ratio <= 1 + length_tolerance:
            matched_indices.append(i)

    if len(matched_indices) < 10:
        print(f"  Warning: Only {len(matched_indices)} length-matched pairs (tol={length_tolerance})")
        return None, matched_indices

    matched_human = train_human_acts[layer][matched_indices]
    matched_ai = train_ai_acts[layer][matched_indices]

    direction = matched_ai.mean(dim=0) - matched_human.mean(dim=0)
    direction = direction / direction.norm()

    return direction, matched_indices


def truncation_matched_direction(train_pairs, train_human_acts, train_ai_acts, layer):
    """Alternative approach: use text truncated to the same length.

    Since we extract activations at the last token, this would require
    re-extracting. Instead, we approximate by using the orthogonal
    complement of the length direction.
    """
    # Compute length direction
    all_acts = torch.cat([train_human_acts[layer], train_ai_acts[layer]], dim=0)
    all_lens = ([len(p["human_text"]) for p in train_pairs] +
                [len(p["ai_text"]) for p in train_pairs])
    median_len = np.median(all_lens)
    short_mask = np.array(all_lens) <= median_len
    long_mask = ~short_mask

    length_dir = all_acts[long_mask].mean(dim=0) - all_acts[short_mask].mean(dim=0)
    length_dir = length_dir / length_dir.norm()

    # Compute raw AI direction
    raw_dir = train_ai_acts[layer].mean(dim=0) - train_human_acts[layer].mean(dim=0)

    # Project out length
    ai_dir_no_length = raw_dir - (torch.dot(raw_dir, length_dir)) * length_dir
    ai_dir_no_length = ai_dir_no_length / ai_dir_no_length.norm()

    return ai_dir_no_length, length_dir


def evaluate_direction(direction, test_human_acts, test_ai_acts, layer):
    """Evaluate a direction's classification performance."""
    midpoint = (test_ai_acts[layer].mean(dim=0) + test_human_acts[layer].mean(dim=0)) / 2
    human_proj = ((test_human_acts[layer] - midpoint) @ direction).numpy()
    ai_proj = ((test_ai_acts[layer] - midpoint) @ direction).numpy()

    all_proj = np.concatenate([human_proj, ai_proj])
    all_labels = np.array([0] * len(human_proj) + [1] * len(ai_proj))

    preds = (all_proj > 0).astype(int)
    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_proj)
    return acc, auc, human_proj, ai_proj


def main():
    # Load data
    with open(RESULTS_DIR / "train_pairs.json") as f:
        train_pairs = json.load(f)
    with open(RESULTS_DIR / "test_pairs.json") as f:
        test_pairs = json.load(f)

    train_data = torch.load(RESULTS_DIR / "train_activations.pt", weights_only=True)
    test_data = torch.load(RESULTS_DIR / "test_activations.pt", weights_only=True)

    with open(RESULTS_DIR / "direction_results.json") as f:
        dir_results = json.load(f)
    best_layer = dir_results["best_layer"]

    print(f"Best layer: {best_layer}")

    # Method 1: Length-matched pairs
    print("\n--- Method 1: Length-Matched Pairs ---")
    for tol in [0.3, 0.5, 1.0, 2.0]:
        direction, indices = length_matched_direction(
            train_pairs, train_data["human"], train_data["ai"], best_layer, tol)
        if direction is not None:
            acc, auc, _, _ = evaluate_direction(
                direction, test_data["human"], test_data["ai"], best_layer)
            print(f"  Tolerance {tol:.1f}: {len(indices)} pairs, "
                  f"test acc={acc:.3f}, test AUC={auc:.3f}")

    # Method 2: Orthogonal to length
    print("\n--- Method 2: Direction Orthogonal to Length ---")
    ai_dir_no_length, length_dir = truncation_matched_direction(
        train_pairs, train_data["human"], train_data["ai"], best_layer)

    acc_no_length, auc_no_length, h_proj, a_proj = evaluate_direction(
        ai_dir_no_length, test_data["human"], test_data["ai"], best_layer)
    print(f"  Test acc (orthog. to length): {acc_no_length:.3f}")
    print(f"  Test AUC (orthog. to length): {auc_no_length:.3f}")

    # Also test length direction alone
    acc_length, auc_length, _, _ = evaluate_direction(
        length_dir, test_data["human"], test_data["ai"], best_layer)
    print(f"  Test acc (length dir only):   {acc_length:.3f}")
    print(f"  Test AUC (length dir only):   {auc_length:.3f}")

    # Method 3: Sweep across all layers for the length-orthogonal direction
    print("\n--- Layer-wise Analysis: Length-Orthogonal Direction ---")
    print(f"{'Layer':>5} | {'Original Acc':>12} | {'Orth-Length Acc':>14} | {'Length-Only Acc':>15}")
    print("-" * 55)

    layer_results_controlled = {}
    for layer in range(len(train_data["human"])):
        ai_dir_orth, len_dir = truncation_matched_direction(
            train_pairs, train_data["human"], train_data["ai"], layer)
        acc_orth, auc_orth, _, _ = evaluate_direction(
            ai_dir_orth, test_data["human"], test_data["ai"], layer)
        acc_len, _, _, _ = evaluate_direction(
            len_dir, test_data["human"], test_data["ai"], layer)
        orig_acc = dir_results["layer_results"][str(layer)]["test_acc"]

        layer_results_controlled[layer] = {
            "original_acc": orig_acc,
            "length_orthogonal_acc": acc_orth,
            "length_orthogonal_auc": auc_orth,
            "length_only_acc": acc_len,
        }
        print(f"{layer:>5} | {orig_acc:>12.3f} | {acc_orth:>14.3f} | {acc_len:>15.3f}")

    # Find best layer for length-orthogonal direction
    best_orth_layer = max(layer_results_controlled,
                          key=lambda l: layer_results_controlled[l]["length_orthogonal_acc"])
    print(f"\nBest layer for length-orthogonal direction: {best_orth_layer}")
    print(f"  Accuracy: {layer_results_controlled[best_orth_layer]['length_orthogonal_acc']:.3f}")

    # Visualization
    layers = sorted(layer_results_controlled.keys())
    orig_accs = [layer_results_controlled[l]["original_acc"] for l in layers]
    orth_accs = [layer_results_controlled[l]["length_orthogonal_acc"] for l in layers]
    len_accs = [layer_results_controlled[l]["length_only_acc"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, orig_accs, "b-o", markersize=3, label="Original AI Direction")
    ax.plot(layers, orth_accs, "r-s", markersize=3, label="Length-Orthogonal AI Direction")
    ax.plot(layers, len_accs, "g-^", markersize=3, label="Length Direction Only")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Disentangling Length from AI Style in the Residual Stream")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "length_controlled_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {PLOT_DIR / 'length_controlled_analysis.png'}")

    # Projection distribution for best orthogonal layer
    ai_dir_best, _ = truncation_matched_direction(
        train_pairs, train_data["human"], train_data["ai"], best_orth_layer)
    _, _, h_proj_best, a_proj_best = evaluate_direction(
        ai_dir_best, test_data["human"], test_data["ai"], best_orth_layer)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(h_proj_best, bins=30, alpha=0.5, label="Human", color="blue", density=True)
    ax.hist(a_proj_best, bins=30, alpha=0.5, label="AI", color="red", density=True)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_xlabel("Projection onto Length-Orthogonal AI Direction")
    ax.set_ylabel("Density")
    ax.set_title(f"Length-Orthogonal AI Direction Projection (Layer {best_orth_layer})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "length_orthogonal_projection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved projection distribution to {PLOT_DIR / 'length_orthogonal_projection.png'}")

    # Save results
    save_data = {
        "best_layer_original": best_layer,
        "best_layer_length_orthogonal": best_orth_layer,
        "layer_results_controlled": {str(k): v for k, v in layer_results_controlled.items()},
    }
    with open(RESULTS_DIR / "length_controlled_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved results to {RESULTS_DIR / 'length_controlled_results.json'}")


if __name__ == "__main__":
    main()
