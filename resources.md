# Resources Catalog

## Research Question
**Is there a "sounds like AI" direction in the residual stream of transformer language models?**

---

## Papers

### Core Methodology Papers (Deep-Read, Notes Available)

| # | Paper | Year | Venue | arXiv | Key Contribution | Notes |
|---|-------|------|-------|-------|------------------|-------|
| 1 | Marks & Tegmark, "The Geometry of Truth" | 2024 | COLM | 2310.06824 | Mass-mean probing for truth direction; causal interventions with NIE up to 0.97 | [notes](papers/notes/geometry_of_truth_notes.md) |
| 2 | Panickssery et al., "Contrastive Activation Addition" | 2024 | ACL | 2312.06681 | Mean-difference steering vectors; 7 behaviors steered; generalizes to open-ended generation | [notes](papers/notes/caa_notes.md) |
| 3 | Arditi et al., "Refusal Direction" | 2024 | NeurIPS | 2406.11717 | Single direction mediates refusal across 13 models; directional ablation; weight orthogonalization | [notes](papers/notes/refusal_direction_notes.md) |
| 4 | Konen et al., "Style Vectors" | 2024 | arXiv | 2402.01618 | Activation-based style vectors steer sentiment/emotion/writing style; layers 18-20 of 33 optimal | [notes](papers/notes/style_vectors_notes.md) |
| 5 | Lai et al., "Style-Specific Neurons" | 2024 | arXiv | 2410.00593 | Style neurons in FFN; 95% overlap between opposing styles; style concentrated in last layers | [notes](papers/notes/style_specific_neurons_notes.md) |

### Foundational Theory Papers

| # | Paper | Year | arXiv | Key Contribution |
|---|-------|------|-------|------------------|
| 6 | Park et al., "Linear Representation Hypothesis" | 2023 | 2311.03658 | Formal framework for linear representations; causal inner product |
| 7 | Zou et al., "Representation Engineering" | 2023 | 2310.01405 | Top-down approach using contrast pairs for concept directions |
| 8 | Turner et al., "Activation Addition" | 2023 | 2308.10248 | Foundational activation steering with single word-pair contrasts |

### Additional Relevant Papers

| # | Paper | Year | arXiv | Relevance |
|---|-------|------|-------|-----------|
| 9 | Cunningham et al., "Sparse Autoencoders" | 2023 | 2309.08600 | SAEs for disentangling superimposed features |
| 10 | Engels et al., "Not All Features Linear" | 2024 | 2405.14860 | Circular features exist; not all concepts are 1D |
| 11 | Mallen et al., "Belief State Geometry" | 2024 | 2405.15943 | Geometric structure of latent beliefs |
| 12 | Kambhampati et al., "Steering with Conceptors" | 2024 | 2410.16314 | Alternative steering via conceptor-based projections |
| 13 | Chalnev et al., "Feature-Guided Activation Additions" | 2025 | 2501.09929 | SAE-guided targeted steering |
| 14 | "SAE Features for Classification" | 2025 | 2502.11367 | Using SAE features for downstream classification |
| 15 | "Representation Engineering Survey" | 2025 | 2502.17601 | Comprehensive survey of the field |
| 16 | "Register Analysis in Style Transfer" | 2025 | 2505.00679 | Linguistic register analysis for style |
| 17 | "Style Vectors with Human Evaluation" | 2026 | 2601.21505 | Human evaluation of style vector steering |

**All PDFs stored in:** `papers/`
**Detailed reading notes in:** `papers/notes/`

---

## Datasets

### Recommended (Highest Relevance)

| Dataset | HuggingFace ID | Size | Structure | License | Why Use It |
|---------|---------------|------|-----------|---------|------------|
| **HC3** | `Hello-SimpleAI/HC3` | ~37k QA pairs | Same questions answered by human AND ChatGPT | CC-BY-SA-4.0 | **Best for contrastive pairs** -- identical prompts, different authors |
| **Human/AI Generated Text** | `dmitva/human_ai_generated_text` | ~1M rows | Columns: `instructions`, `human_text`, `ai_text` | CC-BY-4.0 | **Largest paired dataset** -- each row has both human and AI response to same instruction |

### Large-Scale Detection Corpora

| Dataset | HuggingFace ID | Size | Models | License | Why Use It |
|---------|---------------|------|--------|---------|------------|
| **AI Text Detection Pile** | `artem9k/ai-text-detection-pile` | ~1.39M rows | GPT-2, GPT-3, ChatGPT, GPT-J | MIT | Multi-model, long-form text |
| **AI Text Pile (Cleaned)** | `srikanthgali/ai-text-detection-pile-cleaned` | ~722k rows | Same as above | MIT | Balanced, deduplicated version |
| **RAID** | `liamdugan/raid` | ~10M docs | 12 LLMs | MIT | Most comprehensive; 4 domains, adversarial attacks |

### Additional Datasets

| Dataset | HuggingFace ID | Size | Notes |
|---------|---------------|------|-------|
| AI-Human Text | `andythetechnerd03/AI-human-text` | ~58k+ | NYT articles + synthetic from 6 LLMs (arXiv:2510.22874) |
| M4 Benchmark | `github.com/mbzuai-nlp/M4` | Multi-domain | Multi-generator, multi-domain, multi-lingual (arXiv:2305.14902) |
| Human-Like DPO | `HumanLLMs/Human-Like-DPO-Dataset` | Variable | DPO dataset for human-like generation |

### Download Script
A download utility is provided at `datasets/download_datasets.py`:
```bash
cd datasets
pip install datasets
python download_datasets.py --list        # See all options
python download_datasets.py               # Download HC3 + dmitva_paired
python download_datasets.py --all         # Download everything
python download_datasets.py --dataset hc3 # Download specific dataset
```

---

## Code Repositories

### Cloned (in `code/` directory)

| Repository | Path | License | Key Features |
|-----------|------|---------|--------------|
| **steering-vectors** | `code/steering-vectors/` | MIT | Pip-installable library; `train_steering_vector()` API; mean/PCA/logistic aggregators; works with any HuggingFace model |
| **CAA** | `code/CAA/` | MIT | Nina Rimsky's Llama 2 CAA implementation; `generate_vectors.py` for steering vector extraction; 7 behavioral datasets included |
| **activation-steering** | `code/activation-steering/` | Apache 2.0 | IBM's general-purpose activation steering; ICLR 2025 |
| **awesome-representation-engineering** | `code/awesome-representation-engineering/` | -- | Curated resource list of papers, tools, and datasets |

### Key Code Patterns

**Difference-of-means (CAA, `generate_vectors.py:114`):**
```python
vec = (all_pos_layer - all_neg_layer).mean(dim=0)
```

**steering-vectors library API (`train_steering_vector.py`):**
```python
from steering_vectors import train_steering_vector
sv = train_steering_vector(model, tokenizer, [("positive text", "negative text"), ...], layers=[13])
sv.apply(model, multiplier=1.5)  # steer during generation
```

**Three aggregation methods (`aggregators.py`):**
- `mean_aggregator()`: `(pos - neg).mean(dim=0)` -- default, recommended
- `pca_aggregator()`: First PC of deltas, norm 1
- `logistic_aggregator()`: Logistic regression on centered activations, normalized

### Not Cloned but Relevant

| Repository | URL | Purpose |
|-----------|-----|---------|
| TransformerLens | `github.com/neelnanda-io/TransformerLens` | Fine-grained activation access, patching, probing |
| refusal_direction | `github.com/andyrdt/refusal_direction` | Refusal direction extraction and ablation |
| sNeuron-TST | `github.com/wenlai-lavine/sNeuron-TST` | Style-specific neuron identification |
| DLR style-vectors | `github.com/DLR-SC/style-vectors-for-steering-llms` | Style vector computation and evaluation |
| geometry-of-truth | `github.com/saprmarks/geometry-of-truth` | Truth direction probing and visualization |

---

## Key Equations Reference

### Direction Extraction
```
r = mean(activations_class_A) - mean(activations_class_B)
```

### Activation Addition (steer toward class A)
```
x' = x + λ * r̂     where r̂ = r / ||r||
```

### Directional Ablation (remove concept)
```
x' = x - r̂ * r̂ᵀ * x
```

### Weight Orthogonalization (permanent edit)
```
W' = W - r̂ * r̂ᵀ * W
```

### Mass-Mean Probing (classification)
```
p(x) = σ(θᵀ * x)    where θ = μ₊ - μ₋
```

---

## Recommended Experimental Pipeline

```
1. PREPARE DATA
   └─ HC3 or dmitva/human_ai_generated_text → 128+ contrastive pairs
   └─ Control for domain: multiple topics
   └─ Split: 128 train, 32 val per class

2. EXTRACT ACTIVATIONS
   └─ Load model (e.g., Llama 3 8B) via HuggingFace / TransformerLens
   └─ Forward pass each text, record residual stream at last token
   └─ Store activations per layer

3. FIND DIRECTION
   └─ Difference-in-means at each layer
   └─ PCA visualization to confirm linear separability
   └─ Select best layer via validation (ablation + addition scores)

4. VALIDATE
   └─ Cross-dataset generalization (train on ChatGPT, test on Claude)
   └─ Causal interventions: NIE measurement
   └─ Confound checks: is it just formality? fluency? length?

5. APPLY
   └─ Activation addition: make human text sound more AI / vice versa
   └─ Directional ablation: remove "AI-sounding" quality
   └─ Weight orthogonalization: permanent model edit
```

---

## Hardware Requirements

Based on the surveyed papers:
- **Minimum:** Single GPU with 16GB+ VRAM (for 7B models with 4-bit quantization)
- **Recommended:** Single A100 40GB or equivalent (for 7-8B models at full precision)
- **For 70B models:** Multi-GPU setup or 4-bit quantization on 2x A100
- **Activation extraction:** Can be done in batches; requires storing ~4096-dimensional vectors per layer per sample
- **Compute time:** Activation extraction for 256 texts across 32 layers takes minutes on single GPU
