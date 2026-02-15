"""
Extract residual stream activations from a language model for human vs AI text.
Computes activations at every layer for the last token position.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
DEVICE = "cuda:0"


def load_model(model_name="Qwen/Qwen2.5-3B"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        output_hidden_states=True,
    )
    model.eval()
    print(f"Model loaded. {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def get_activations(model, tokenizer, text, max_length=512):
    """Get residual stream activations at the last real token for all layers.

    Returns: dict mapping layer_idx -> activation tensor (hidden_dim,)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # hidden_states[0] = embeddings, hidden_states[1..N] = layer outputs
    hidden_states = outputs.hidden_states

    # Get last real token position
    seq_len = inputs["input_ids"].shape[1]
    last_pos = seq_len - 1

    activations = {}
    for layer_idx in range(len(hidden_states)):
        # Shape: (1, seq_len, hidden_dim) -> (hidden_dim,)
        act = hidden_states[layer_idx][0, last_pos, :].float().cpu()
        activations[layer_idx] = act

    return activations


def extract_all_activations(model, tokenizer, pairs, label=""):
    """Extract activations for all human and AI texts in the pairs."""
    human_acts_by_layer = {}  # layer -> list of tensors
    ai_acts_by_layer = {}

    for pair in tqdm(pairs, desc=f"Extracting {label}"):
        # Human text activations
        h_acts = get_activations(model, tokenizer, pair["human_text"])
        for layer, act in h_acts.items():
            if layer not in human_acts_by_layer:
                human_acts_by_layer[layer] = []
            human_acts_by_layer[layer].append(act)

        # AI text activations
        a_acts = get_activations(model, tokenizer, pair["ai_text"])
        for layer, act in a_acts.items():
            if layer not in ai_acts_by_layer:
                ai_acts_by_layer[layer] = []
            ai_acts_by_layer[layer].append(act)

    # Stack into tensors
    for layer in human_acts_by_layer:
        human_acts_by_layer[layer] = torch.stack(human_acts_by_layer[layer])
        ai_acts_by_layer[layer] = torch.stack(ai_acts_by_layer[layer])

    return human_acts_by_layer, ai_acts_by_layer


def main():
    # Load data splits
    train_pairs = json.loads((RESULTS_DIR / "train_pairs.json").read_text())
    val_pairs = json.loads((RESULTS_DIR / "val_pairs.json").read_text())
    test_pairs = json.loads((RESULTS_DIR / "test_pairs.json").read_text())

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

    # Load model
    model, tokenizer = load_model()

    # Extract activations for each split
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        print(f"\n--- Extracting {name} activations ---")
        human_acts, ai_acts = extract_all_activations(model, tokenizer, pairs, label=name)

        # Save
        save_path = RESULTS_DIR / f"{name}_activations.pt"
        torch.save({"human": human_acts, "ai": ai_acts}, save_path)
        print(f"Saved {name} activations to {save_path}")

        # Print shapes
        example_layer = list(human_acts.keys())[0]
        print(f"  Layers: {len(human_acts)}")
        print(f"  Human shape per layer: {human_acts[example_layer].shape}")
        print(f"  AI shape per layer: {ai_acts[example_layer].shape}")


if __name__ == "__main__":
    main()
