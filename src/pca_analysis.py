"""
PCA visualization of residual stream activations.
Shows how AI vs human text separates across layers.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = RESULTS_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def pca_visualization_grid(human_acts_by_layer, ai_acts_by_layer, layers_to_plot=None):
    """Create a grid of PCA plots for selected layers."""
    n_total_layers = len(human_acts_by_layer)

    if layers_to_plot is None:
        # Select ~9 evenly spaced layers
        layers_to_plot = np.linspace(0, n_total_layers - 1, 9, dtype=int).tolist()

    n_plots = len(layers_to_plot)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    silhouette_scores = {}

    for idx, layer in enumerate(layers_to_plot):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        human = human_acts_by_layer[layer].numpy()
        ai = ai_acts_by_layer[layer].numpy()

        combined = np.vstack([human, ai])
        labels = np.array([0] * len(human) + [1] * len(ai))

        pca = PCA(n_components=2)
        projected = pca.fit_transform(combined)

        # Silhouette score
        sil = silhouette_score(projected, labels)
        silhouette_scores[layer] = sil

        ax.scatter(projected[:len(human), 0], projected[:len(human), 1],
                   alpha=0.5, s=20, label="Human", c="blue")
        ax.scatter(projected[len(human):, 0], projected[len(human):, 1],
                   alpha=0.5, s=20, label="AI", c="red")
        ax.set_title(f"Layer {layer} (sil={sil:.2f})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(n_plots, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("PCA of Residual Stream: Human vs AI Text", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "pca_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA grid to {PLOT_DIR / 'pca_grid.png'}")

    return silhouette_scores


def silhouette_by_layer(human_acts_by_layer, ai_acts_by_layer):
    """Compute and plot silhouette score at every layer."""
    n_layers = len(human_acts_by_layer)
    sil_scores = []

    for layer in range(n_layers):
        human = human_acts_by_layer[layer].numpy()
        ai = ai_acts_by_layer[layer].numpy()
        combined = np.vstack([human, ai])
        labels = np.array([0] * len(human) + [1] * len(ai))

        pca = PCA(n_components=2)
        projected = pca.fit_transform(combined)
        sil = silhouette_score(projected, labels)
        sil_scores.append(sil)

    return sil_scores


def plot_accuracy_and_silhouette(direction_results, sil_scores):
    """Plot classification accuracy and silhouette score across layers."""
    layers = sorted([int(k) for k in direction_results["layer_results"].keys()])
    val_accs = [direction_results["layer_results"][str(l)]["val_acc"] for l in layers]
    test_accs = [direction_results["layer_results"][str(l)]["test_acc"] for l in layers]
    test_aucs = [direction_results["layer_results"][str(l)]["test_auc"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Accuracy plot
    ax1.plot(layers, val_accs, "b-o", markersize=4, label="Val Accuracy (mass-mean)")
    ax1.plot(layers, test_accs, "r-s", markersize=4, label="Test Accuracy (mass-mean)")
    ax1.plot(layers, test_aucs, "g-^", markersize=4, label="Test AUC")
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Performance by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.05)

    # Silhouette plot
    ax2.plot(layers, sil_scores, "m-d", markersize=4, label="Silhouette Score (PCA 2D)")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("PCA Cluster Separation by Layer")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark best layer
    best_layer = direction_results["best_layer"]
    ax1.axvline(best_layer, color="orange", linestyle=":", alpha=0.7, label=f"Best layer={best_layer}")
    ax2.axvline(best_layer, color="orange", linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_silhouette_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy/silhouette plot to {PLOT_DIR / 'accuracy_silhouette_by_layer.png'}")


def plot_cross_layer_similarity(direction_results):
    """Plot cosine similarity between the direction at adjacent layers."""
    cos_sims = direction_results["cross_layer_cos_sims"]
    layers = list(range(1, len(cos_sims) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, cos_sims, "k-o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity with Previous Layer")
    ax.set_title("Direction Consistency Across Adjacent Layers")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.05)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cross_layer_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cross-layer similarity plot to {PLOT_DIR / 'cross_layer_similarity.png'}")


def main():
    # Load test activations for PCA
    print("Loading test activations...")
    test_data = torch.load(RESULTS_DIR / "test_activations.pt", weights_only=True)
    test_human = test_data["human"]
    test_ai = test_data["ai"]

    # Load direction results
    with open(RESULTS_DIR / "direction_results.json") as f:
        direction_results = json.load(f)

    # PCA grid
    print("Creating PCA grid...")
    sil_grid = pca_visualization_grid(test_human, test_ai)

    # Silhouette by layer
    print("Computing silhouette scores by layer...")
    sil_scores = silhouette_by_layer(test_human, test_ai)

    # Save silhouette scores
    with open(RESULTS_DIR / "silhouette_scores.json", "w") as f:
        json.dump(sil_scores, f)

    # Plot accuracy and silhouette
    print("Creating accuracy/silhouette plot...")
    plot_accuracy_and_silhouette(direction_results, sil_scores)

    # Plot cross-layer similarity
    print("Creating cross-layer similarity plot...")
    plot_cross_layer_similarity(direction_results)

    print("\nAll PCA analysis complete!")


if __name__ == "__main__":
    main()
