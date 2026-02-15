# Literature Notes: "Refusal in Language Models Is Mediated by a Single Direction"

**Paper:** Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda (NeurIPS 2024)
**arXiv:** 2406.11717
**Code:** https://github.com/andyrdt/refusal_direction

---

## Core Claim

Refusal behavior in safety-fine-tuned chat LLMs is mediated by a **single direction** (a one-dimensional subspace) in the residual stream. This holds across 13 open-source chat models up to 72B parameters. Erasing this direction prevents refusal on harmful instructions; adding it induces refusal on harmless instructions.

---

## 1. How the "Refusal Direction" Is Identified (Methodology)

### 1.1 Difference-in-Means

The method uses **contrastive pairs** of harmful vs. harmless instructions to isolate the refusal direction via the **difference-in-means** technique (Belrose, 2023; Marks & Tegmark, 2023; Panickssery et al., 2023).

For each layer `l` and each post-instruction token position `i` (tokens after the user instruction in the chat template, before the model's response):

```
mu_i^(l) = (1/|D_harmful_train|) * sum_{t in D_harmful_train} x_i^(l)(t)
nu_i^(l) = (1/|D_harmless_train|) * sum_{t in D_harmless_train} x_i^(l)(t)
```

The difference-in-means vector is:

```
r_i^(l) = mu_i^(l) - nu_i^(l)
```

This vector is meaningful in both:
1. **Direction** -- the axis along which harmful and harmless mean activations differ
2. **Magnitude** -- the distance between those means

### 1.2 Selecting a Single Best Vector

Computing `r_i^(l)` for every post-instruction token position `i` and every layer `l` yields `|I| x L` candidate vectors. Selection uses validation sets `D_harmful_val` (32 samples) and `D_harmless_val` (32 samples) to score each candidate on three criteria:

- **bypass_score**: Under directional ablation of the candidate, compute average "refusal metric" (log-odds of refusal token probability) across harmful validation prompts. Lower = better at bypassing refusal.
- **induce_score**: Under activation addition of the candidate, compute average refusal metric across harmless validation prompts. Higher = better at inducing refusal.
- **kl_score**: KL divergence between next-token distributions with and without directional ablation on harmless validation prompts. Lower = less disruption to normal behavior.

The selected direction is the one with **minimum bypass_score** subject to:
- `induce_score > 0` (must be sufficient to induce refusal)
- `kl_score < 0.1` (must not significantly change behavior on harmless prompts)
- `l < 0.8 * L` (must not be too close to the unembedding layer, to avoid trivially blocking output tokens rather than finding a higher-level feature)

### 1.3 The Refusal Metric (Efficient Proxy)

Rather than generating full completions, the authors define a fast proxy:
- For each model family, identify a small set of "refusal tokens" R (e.g., token for "I" in Gemma, tokens for "I'm sorry" and "As an AI" in Qwen).
- `P_refusal = sum of probabilities assigned to tokens in R`
- `refusal_metric = logit(P_refusal) = log(P_refusal / (1 - P_refusal))`

This metric efficiently separates harmful from harmless instructions and is used for filtering datasets and evaluating candidate directions during selection.

---

## 2. Which Layers Contain the Refusal Direction

From Table 5, the selected layer `l*` relative to total layers `L` for each model:

| Model | Token Position i* | Layer l*/L | Layer Fraction |
|-------|-------------------|-----------|----------------|
| Qwen 1.8B | -1 (last) | 15/24 | 0.625 |
| Qwen 7B | -1 | 17/32 | 0.531 |
| Qwen 14B | -1 | 23/40 | 0.575 |
| Qwen 72B | -1 | 62/80 | 0.775 |
| Yi 6B | -5 | 20/32 | 0.625 |
| Yi 34B | -1 | 37/60 | 0.617 |
| Gemma 2B | -2 | 10/18 | 0.556 |
| Gemma 7B | -1 | 14/28 | 0.500 |
| Llama-2 7B | -1 | 14/32 | 0.438 |
| Llama-2 13B | -1 | 26/40 | 0.650 |
| Llama-2 70B | -1 | 21/80 | 0.263 |
| Llama-3 8B | -5 | 12/32 | 0.375 |
| Llama-3 70B | -5 | 25/80 | 0.313 |

**Key observations:**
- The direction is typically found in **middle layers** (roughly 25%-75% of the way through the network).
- Most models select the **last token position** (`i* = -1`), but Llama-3 models and Yi 6B select position -5 (the `<|eot_id|>` token in Llama-3).
- The constraint `l < 0.8L` prevents selection from final layers, which would just be blocking unembedding directions.

---

## 3. Causal Evidence for the Direction's Role

The paper provides **necessity and sufficiency** evidence:

### 3.1 Necessity: Directional Ablation Bypasses Refusal
- Ablating the refusal direction `r_hat` from all layers and all token positions during inference.
- Evaluated on JailbreakBench (100 harmful instructions).
- Result: refusal rates drop dramatically across all 13 models (from ~0.8-1.0 refusal score to near 0), and safety scores drop correspondingly (models produce unsafe completions).

### 3.2 Sufficiency: Activation Addition Induces Refusal
- Adding the difference-in-means vector `r` to activations at layer `l*` across all token positions.
- Evaluated on 100 harmless instructions from Alpaca.
- Result: models refuse even harmless instructions (refusal scores jump from ~0 to ~0.8-1.0 across all 13 models).

### 3.3 Adversarial Suffix Analysis (Mechanistic)
- On Qwen 1.8B Chat, adversarial suffixes suppress the refusal direction in the residual stream.
- Cosine similarity of last-token activations with the refusal direction is high for harmful instructions, remains high with random suffixes, but drops to harmless-instruction levels with adversarial suffixes.
- The top 8 attention heads that write most to the refusal direction (identified via Direct Feature Attribution / DFA) have their output heavily suppressed when adversarial suffixes are present.
- Mechanism: adversarial suffixes **hijack attention** -- these key heads shift attention from the instruction region to the suffix region, preventing them from reading harmful content and thus suppressing the refusal direction.

---

## 4. How Steering/Ablation Works

### 4.1 Activation Addition (Inference-Time)
Add the difference-in-means vector to residual stream activations at a single layer:
```
x^(l)' <- x^(l) + r^(l)    # induces refusal
x^(l)' <- x^(l) - r^(l)    # bypasses refusal
```
Intervention is at layer `l*` only, across all token positions.

### 4.2 Directional Ablation (Inference-Time)
Project out the refusal direction from every residual stream activation:
```
x' <- x - r_hat * r_hat^T * x
```
Applied at every activation `x_i^(l)` and `x_tilde_i^(l)` (before and after attention), across **all layers** and **all token positions**. This completely prevents the model from ever representing the refusal direction.

### 4.3 Weight Orthogonalization (Permanent Weight Edit)
Equivalent to directional ablation but implemented as a direct weight modification. For every matrix `W_out` that writes to the residual stream:
```
W_out' <- W_out - r_hat * r_hat^T * W_out
```
This applies to: embedding matrix, positional embedding matrix, attention output matrices, MLP output matrices, and any output biases.

**Key insight:** This is mathematically equivalent to inference-time directional ablation (proven in Appendix E), but is a permanent rank-one weight edit requiring no runtime hooks.

### 4.4 Comparison: Ablation vs. Activation Addition
- Directional ablation is **more surgical** than activation addition.
- Activation addition in the negative direction shifts harmful activations toward harmless ones (good), but also pushes harmless activations off-distribution (bad, increases perplexity).
- Directional ablation zeroes out the direction for both, which shifts harmful toward harmless without pushing harmless off-distribution.
- CE loss measurements confirm: ablation causes much less increase in loss on harmless data than activation addition.

---

## 5. Datasets Used

### Training Data for Direction Extraction
- **D_harmful_train** (128 samples): Randomly sampled from AdvBench, MaliciousInstruct, TDC2023
- **D_harmless_train** (128 samples): Sampled from Alpaca
- **D_harmful_val** (32 samples): From HarmBench validation set (standard behaviors only)
- **D_harmless_val** (32 samples): Sampled from Alpaca
- All splits are pairwise disjoint with each other and with evaluation sets.

### Evaluation Datasets
- **JailbreakBench** (100 harmful instructions, 10 categories): Used for Section 3 evaluations
- **HarmBench test set** (159 standard behaviors, 6 categories): Used for Section 4 jailbreak comparisons
- **Alpaca** (100 harmless instructions): Used for inducing-refusal evaluations
- **The Pile**: Used for CE loss evaluation

### Model Coherence Benchmarks
- MMLU, ARC, GSM8K, TruthfulQA, WinoGrande, TinyHellaSwag

### Models Studied (13 chat models)
| Family | Sizes | Alignment Type |
|--------|-------|---------------|
| Qwen Chat | 1.8B, 7B, 14B, 72B | AFT (aligned by fine-tuning) |
| Yi Chat | 6B, 34B | AFT |
| Gemma IT | 2B, 7B | APO (aligned by preference optimization) |
| Llama-2 Chat | 7B, 13B, 70B | APO |
| Llama-3 Instruct | 8B, 70B | APO |

---

## 6. Results

### 6.1 Bypassing Refusal (Directional Ablation)
- Across all 13 models, ablating the refusal direction reduces refusal scores from ~0.8-1.0 to near 0.
- Safety scores drop correspondingly (models produce harmful content).

### 6.2 Inducing Refusal (Activation Addition)
- Adding the refusal direction at layer l* to harmless prompts causes models to refuse nearly all harmless instructions.

### 6.3 White-Box Jailbreak Comparison (HarmBench ASR)
- Weight orthogonalization (ORTHO) competitive with or superior to other general jailbreak methods.
- Qwen family: ORTHO achieves 78-84% ASR (comparable to prompt-specific GCG at 79-84%).
- Llama-2 family: ORTHO achieves 4.4-22.6% with system prompt, but 61-80% without system prompt.
- Llama-2 models are much more sensitive to system prompts than Qwen models.

### 6.4 Model Coherence After Orthogonalization
- MMLU, ARC, GSM8K: negligible performance changes (typically <1% absolute difference).
- TruthfulQA: **consistent drop** across all models (2-6% absolute). Likely because TruthfulQA contains questions about misinformation, stereotypes, conspiracies that are adjacent to refusal territory.
- CE loss: ablation causes minimal increase; activation addition causes larger increase.
- Qualitative: orthogonalized models produce virtually identical responses on harmless prompts.

### 6.5 Refusal Direction in Base Models
- The refusal direction (extracted from chat models) is **already expressed in corresponding base models**.
- Base models show high cosine similarity with the refusal direction on harmful prompts and low on harmless prompts.
- Implication: safety fine-tuning does not create this direction from scratch; it repurposes an existing representation.

---

## 7. Limitations (Stated by Authors)

1. **Generalization uncertainty**: Findings may not extend to untested models, larger scales, proprietary models, or future models.
2. **Methodology is heuristic**: The direction extraction relies on several heuristics (difference-in-means, selection criteria). The paper is an "existence proof" rather than an optimal extraction method.
3. **Adversarial suffix analysis is limited**: Restricted to a single model (Qwen 1.8B) and a single adversarial suffix.
4. **Coherence measurement is imperfect**: Each metric used has limitations; multiple varied metrics give a broad but incomplete picture.
5. **Semantic ambiguity of the direction**: The "refusal direction" may represent "harm," "danger," or something else entirely -- its true semantic meaning is unclear.
6. **System prompt sensitivity**: Llama-2 models show large ASR variation depending on system prompt; the method's effectiveness is model-dependent in the white-box jailbreak setting.
7. **TruthfulQA degradation**: Consistent drops suggest the direction may encode more than just refusal.

---

## 8. Relevance to Our Project: "Sounds Like AI" Direction

This paper's methodology is **directly applicable** to finding a "sounds like AI" direction in the residual stream. Key parallels and lessons:

### What to replicate:
1. **Contrastive dataset construction**: We need pairs of "AI-sounding" vs. "human-sounding" text (analogous to harmful vs. harmless). These could be:
   - AI-generated text vs. human-written text on similar topics
   - The same content rewritten in AI-typical vs. human-typical style
   - Model outputs before vs. after style-steering interventions

2. **Difference-in-means computation**: Apply the same formula -- compute mean activations for each class at each (layer, position) pair, take the difference.

3. **Direction selection**: Use similar validation criteria adapted for our task:
   - bypass_score analog: Does ablating this direction make AI-sounding text less AI-like?
   - induce_score analog: Does adding this direction make human-sounding text more AI-like?
   - kl_score: Does ablation preserve coherence?

4. **Interventions**: Both directional ablation (remove the "AI style" direction) and activation addition (amplify or suppress it) should work.

5. **Weight orthogonalization**: For permanent model edits that remove AI-sounding style.

### Key methodological details to adopt:
- Use **128 training + 32 validation** samples per class (small dataset is sufficient)
- Focus on **post-instruction token positions** (where the model formulates its response)
- Search across **all layers** and select via validation
- Apply ablation at **all layers and positions** for maximum effect
- Use the `l < 0.8L` constraint to avoid trivial token-level effects
- The direction is likely in **middle layers** (25-75% of network depth)
- **Base models already encode the feature** -- the direction probably exists before any fine-tuning

### Differences from refusal:
- Refusal is binary (refuse or comply); "AI-sounding" is more of a continuous spectrum.
- Refusal has clear behavioral markers (refusal phrases); "AI-sounding" style markers are more diffuse (hedging, formality, bullet points, etc.).
- We may need **multiple directions** or a higher-dimensional subspace (the paper found 1D sufficient for refusal, but style may be richer).
- Evaluation will be harder -- no simple string-matching; may need a classifier or human evaluation.

### Tools mentioned:
- TransformerLens (exploratory research)
- HuggingFace Transformers + PyTorch (experimental pipeline)
- vLLM (fast inference)
- LM Evaluation Harness (benchmarking)

---

## 9. Key Equations Reference

**Difference-in-means:**
```
r_i^(l) = mu_i^(l) - nu_i^(l)
```

**Activation addition (induce behavior):**
```
x^(l)' <- x^(l) + r^(l)
```

**Directional ablation (remove behavior):**
```
x' <- x - r_hat * r_hat^T * x
```

**Weight orthogonalization (permanent edit):**
```
W_out' <- W_out - r_hat * r_hat^T * W_out
```

**Refusal metric (efficient proxy):**
```
refusal_metric(p) = log(sum_{t in R} p_t) - log(sum_{t not in R} p_t)
```
