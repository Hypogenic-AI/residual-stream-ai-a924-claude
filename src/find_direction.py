"""
Find the 'sounds like AI' direction in the residual stream.
Uses difference-in-means (mass-mean probing) to extract a direction at each layer,
then evaluates classification accuracy and selects the best layer.
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def compute_direction(human_acts, ai_acts):
    """Compute difference-in-means direction: AI mean - Human mean.

    The resulting direction points from 'human' toward 'AI'.
    """
    ai_mean = ai_acts.mean(dim=0)
    human_mean = human_acts.mean(dim=0)
    direction = ai_mean - human_mean
    return direction


def mass_mean_probe_accuracy(direction, human_acts, ai_acts):
    """Classify using mass-mean probing: project onto direction, threshold at 0.

    Positive projection = AI, negative = human.
    """
    midpoint = (ai_acts.mean(dim=0) + human_acts.mean(dim=0)) / 2

    # Project human and AI activations onto direction
    human_proj = ((human_acts - midpoint) @ direction).numpy()
    ai_proj = ((ai_acts - midpoint) @ direction).numpy()

    # Labels: 0 = human, 1 = AI
    all_proj = np.concatenate([human_proj, ai_proj])
    all_labels = np.array([0] * len(human_proj) + [1] * len(ai_proj))

    # Predict: positive projection = AI
    predictions = (all_proj > 0).astype(int)
    acc = accuracy_score(all_labels, predictions)

    # AUC
    auc = roc_auc_score(all_labels, all_proj)

    return acc, auc


def logistic_regression_accuracy(human_acts, ai_acts):
    """Train a logistic regression classifier as an upper bound on linear separability."""
    X = torch.cat([human_acts, ai_acts], dim=0).numpy()
    y = np.array([0] * len(human_acts) + [1] * len(ai_acts))

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X, y)
    preds = clf.predict(X)
    proba = clf.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, proba)
    return acc, auc, clf


def random_direction_accuracy(human_acts, ai_acts, n_trials=100):
    """Baseline: accuracy of random directions."""
    dim = human_acts.shape[1]
    accs = []
    for _ in range(n_trials):
        rand_dir = torch.randn(dim)
        rand_dir = rand_dir / rand_dir.norm()
        acc, _ = mass_mean_probe_accuracy(rand_dir, human_acts, ai_acts)
        accs.append(acc)
    return np.mean(accs), np.std(accs)


def bootstrap_ci(scores, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence interval."""
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


def main():
    # Load activations
    print("Loading activations...")
    train_data = torch.load(RESULTS_DIR / "train_activations.pt", weights_only=True)
    val_data = torch.load(RESULTS_DIR / "val_activations.pt", weights_only=True)
    test_data = torch.load(RESULTS_DIR / "test_activations.pt", weights_only=True)

    train_human = train_data["human"]
    train_ai = train_data["ai"]
    val_human = val_data["human"]
    val_ai = val_data["ai"]
    test_human = test_data["human"]
    test_ai = test_data["ai"]

    n_layers = len(train_human)
    print(f"Number of layers: {n_layers}")
    print(f"Hidden dim: {train_human[0].shape[1]}")

    # Compute direction at each layer and evaluate
    results = {}
    directions = {}

    print("\n--- Layer-wise Direction Analysis ---")
    print(f"{'Layer':>5} | {'Train Acc':>9} | {'Val Acc':>7} | {'Val AUC':>7} | {'Test Acc':>8} | {'Test AUC':>8} | {'LR Acc':>6} | {'Rand Acc':>8}")
    print("-" * 90)

    for layer in range(n_layers):
        # Compute direction on training data
        direction = compute_direction(train_human[layer], train_ai[layer])
        direction_norm = direction / direction.norm()
        directions[layer] = direction_norm

        # Evaluate on all splits
        train_acc, train_auc = mass_mean_probe_accuracy(direction_norm, train_human[layer], train_ai[layer])
        val_acc, val_auc = mass_mean_probe_accuracy(direction_norm, val_human[layer], val_ai[layer])
        test_acc, test_auc = mass_mean_probe_accuracy(direction_norm, test_human[layer], test_ai[layer])

        # Logistic regression upper bound (on train, eval on val)
        lr_acc, lr_auc, _ = logistic_regression_accuracy(train_human[layer], train_ai[layer])

        # Random baseline (quick)
        rand_acc, rand_std = random_direction_accuracy(train_human[layer], train_ai[layer], n_trials=50)

        results[layer] = {
            "train_acc": train_acc,
            "train_auc": train_auc,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "lr_train_acc": lr_acc,
            "lr_train_auc": lr_auc,
            "rand_acc_mean": rand_acc,
            "rand_acc_std": rand_std,
        }

        print(f"{layer:>5} | {train_acc:>9.3f} | {val_acc:>7.3f} | {val_auc:>7.3f} | "
              f"{test_acc:>8.3f} | {test_auc:>8.3f} | {lr_acc:>6.3f} | {rand_acc:>5.3f}±{rand_std:.3f}")

    # Find best layer by validation accuracy
    best_layer = max(results, key=lambda l: results[l]["val_acc"])
    print(f"\nBest layer (by val acc): {best_layer}")
    print(f"  Val Acc: {results[best_layer]['val_acc']:.3f}")
    print(f"  Val AUC: {results[best_layer]['val_auc']:.3f}")
    print(f"  Test Acc: {results[best_layer]['test_acc']:.3f}")
    print(f"  Test AUC: {results[best_layer]['test_auc']:.3f}")

    # Bootstrap CI on test accuracy for best layer
    direction = directions[best_layer]
    midpoint = (test_ai[best_layer].mean(dim=0) + test_human[best_layer].mean(dim=0)) / 2
    all_acts = torch.cat([test_human[best_layer], test_ai[best_layer]])
    all_labels = np.array([0] * len(test_human[best_layer]) + [1] * len(test_ai[best_layer]))
    all_proj = ((all_acts - midpoint) @ direction).numpy()
    preds = (all_proj > 0).astype(int)
    correct = (preds == all_labels).astype(float)
    ci_lower, ci_upper = bootstrap_ci(correct)
    print(f"  Test Acc 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    # Cross-layer consistency: cosine similarity between adjacent layers
    print("\n--- Cross-layer Direction Consistency ---")
    cos_sims = []
    for l in range(1, n_layers):
        sim = torch.dot(directions[l], directions[l - 1]).item()
        cos_sims.append(sim)
    print(f"Adjacent layer cosine sim: mean={np.mean(cos_sims):.3f}, min={np.min(cos_sims):.3f}, max={np.max(cos_sims):.3f}")

    # Save results
    save_data = {
        "layer_results": {str(k): v for k, v in results.items()},
        "best_layer": best_layer,
        "best_test_acc": results[best_layer]["test_acc"],
        "best_test_auc": results[best_layer]["test_auc"],
        "test_acc_ci": [ci_lower, ci_upper],
        "cross_layer_cos_sims": cos_sims,
    }
    with open(RESULTS_DIR / "direction_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    # Save best direction
    torch.save(directions[best_layer], RESULTS_DIR / "best_direction.pt")
    torch.save(directions, RESULTS_DIR / "all_directions.pt")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
