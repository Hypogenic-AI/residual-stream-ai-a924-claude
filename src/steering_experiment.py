"""
Causal intervention experiment: use the AI direction to steer model generation.
Tests both activation addition (make human text sound more AI) and
subtraction (make AI text sound more human).
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOT_DIR = RESULTS_DIR / "plots"
DEVICE = "cuda:0"


class SteeringHook:
    """Hook to add/subtract a direction from the residual stream at a specific layer."""

    def __init__(self, direction, multiplier, layer_idx):
        self.direction = direction.to(torch.float16)
        self.multiplier = multiplier
        self.layer_idx = layer_idx
        self.handle = None

    def hook_fn(self, module, input, output):
        # output can be a tuple or a single tensor depending on the model
        if isinstance(output, tuple):
            hidden = output[0]
            steering = self.multiplier * self.direction.to(hidden.device)
            hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        else:
            # Single tensor output
            hidden = output
            steering = self.multiplier * self.direction.to(hidden.device)
            hidden = hidden + steering.unsqueeze(0).unsqueeze(0)
            return hidden

    def register(self, model):
        # Find the right layer module
        layers = get_layer_modules(model)
        if self.layer_idx < len(layers):
            self.handle = layers[self.layer_idx].register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


def get_layer_modules(model):
    """Get the list of transformer layer modules."""
    # Try common architectures
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'layers'):
            return list(inner.layers)
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            return list(model.transformer.h)
    raise ValueError("Could not find layer modules")


def generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_with_steering(model, tokenizer, prompt, direction, layer_idx,
                           multiplier=1.0, max_new_tokens=150, temperature=0.7):
    """Generate text with the AI direction added/subtracted."""
    hook = SteeringHook(direction, multiplier, layer_idx)
    hook.register(model)
    try:
        text = generate_text(model, tokenizer, prompt, max_new_tokens, temperature)
    finally:
        hook.remove()
    return text


def run_steering_experiment(model, tokenizer, direction, best_layer, prompts, multipliers):
    """Run steering at different multipliers and collect outputs."""
    results = []

    for prompt_info in prompts:
        prompt = prompt_info["prompt"]
        prompt_label = prompt_info["label"]
        print(f"\n--- Prompt: {prompt_label} ---")
        print(f"  '{prompt[:80]}...'")

        for mult in multipliers:
            print(f"  Multiplier: {mult:+.1f}", end=" -> ")
            if mult == 0:
                text = generate_text(model, tokenizer, prompt, max_new_tokens=150, temperature=0.7)
            else:
                text = generate_with_steering(
                    model, tokenizer, prompt, direction, best_layer,
                    multiplier=mult, max_new_tokens=150, temperature=0.7
                )
            print(f"'{text[:80]}...'")

            results.append({
                "prompt": prompt,
                "prompt_label": prompt_label,
                "multiplier": mult,
                "generated_text": text,
            })

    return results


def main():
    # Load direction results
    with open(RESULTS_DIR / "direction_results.json") as f:
        dir_results = json.load(f)

    best_layer = dir_results["best_layer"]
    print(f"Best layer: {best_layer}")

    # Load the direction
    direction = torch.load(RESULTS_DIR / "best_direction.pt", weights_only=True)
    print(f"Direction shape: {direction.shape}, norm: {direction.norm():.3f}")

    # Scale the direction by the typical magnitude observed in activations
    # Load train activations to compute typical scale
    train_data = torch.load(RESULTS_DIR / "train_activations.pt", weights_only=True)
    ai_acts = train_data["ai"][best_layer]
    human_acts = train_data["human"][best_layer]
    raw_direction = ai_acts.mean(dim=0) - human_acts.mean(dim=0)
    direction_magnitude = raw_direction.norm().item()
    print(f"Raw direction magnitude: {direction_magnitude:.2f}")

    # Use the raw (unnormalized) direction for steering — its magnitude is already
    # calibrated to the activation space scale
    steering_direction = raw_direction / raw_direction.norm()

    # Load model
    print("\nLoading model...")
    model_name = "Qwen/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=DEVICE,
    )
    model.eval()

    # Define test prompts
    prompts = [
        {"prompt": "Explain what machine learning is to a beginner:\n", "label": "ML explanation"},
        {"prompt": "What are the benefits of exercise?\n", "label": "Exercise benefits"},
        {"prompt": "Write a short paragraph about climate change:\n", "label": "Climate change"},
        {"prompt": "How does the internet work?\n", "label": "Internet explanation"},
        {"prompt": "Tell me about the history of pizza:\n", "label": "Pizza history"},
    ]

    # Test at different multiplier strengths
    # Positive = more AI-like, Negative = more human-like
    # Use the raw direction magnitude as scale reference
    scale = direction_magnitude
    multipliers = [-1.0 * scale, -0.5 * scale, 0.0, 0.5 * scale, 1.0 * scale]

    print(f"\nMultipliers: {[f'{m:.1f}' for m in multipliers]}")
    print("Positive = push toward AI style, Negative = push toward human style")

    # Run experiment
    results = run_steering_experiment(model, tokenizer, steering_direction, best_layer,
                                      prompts, multipliers)

    # Save results
    with open(RESULTS_DIR / "steering_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} steering results to {RESULTS_DIR / 'steering_results.json'}")

    # Print summary
    print("\n" + "=" * 80)
    print("STEERING RESULTS SUMMARY")
    print("=" * 80)
    for prompt_info in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt_info['label']}")
        print(f"{'='*60}")
        for r in results:
            if r["prompt_label"] == prompt_info["label"]:
                direction_label = "BASELINE" if r["multiplier"] == 0 else (
                    f"→ MORE AI (+{r['multiplier']:.1f})" if r["multiplier"] > 0
                    else f"→ MORE HUMAN ({r['multiplier']:.1f})"
                )
                print(f"\n  [{direction_label}]:")
                print(f"  {r['generated_text'][:200]}")


if __name__ == "__main__":
    main()
