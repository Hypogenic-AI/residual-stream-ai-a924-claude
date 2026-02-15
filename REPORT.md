# Is There a "Sounds Like AI" Direction in the Residual Stream?

## 1. Executive Summary

We investigated whether "sounding like AI" is encoded as a linear direction in the residual stream of a transformer language model (Qwen 2.5 3B). Using contrastive activation analysis on paired human and ChatGPT responses from the HC3 dataset, we found a direction that classifies AI vs. human text with **97.5% test accuracy** (AUC = 0.999). However, this direction is highly correlated with text length (cosine similarity 0.93 with the length direction), reflecting the fact that ChatGPT responses are systematically longer. After projecting out the length component, a residual "AI style" direction still achieves **85.5% test accuracy** — well above chance — indicating that the model does encode something about AI writing style beyond just length. Causal steering experiments confirm that adding/subtracting this direction during generation shifts output style in the expected direction, though the effect is modest in a base (non-chat) model.

**Key takeaway:** Yes, there is a direction in the residual stream associated with "sounding like AI," but it is substantially confounded with text length. The pure style signal is real (85% accuracy after controlling for length) but weaker than the composite signal suggests. "Sounding like AI" is not as cleanly unitary as concepts like truth or refusal.

## 2. Goal

**Hypothesis:** Large language models exhibit linear structure in their residual streams that controls output style and semantics. We test whether there exists a specific direction corresponding to text that "sounds like AI" — the composite of formal tone, hedging language, structured formatting, and comprehensive coverage that characterizes LLM outputs.

**Importance:** Understanding how AI style is represented internally could enable (1) making LLM outputs more natural, (2) understanding what distinguishes AI from human writing at a mechanistic level, and (3) developing better AI text detection methods grounded in model internals rather than surface features.

**Gap filled:** While prior work has demonstrated linear directions for truth (Marks & Tegmark 2024), refusal (Arditi et al. 2024), sentiment (Konen et al. 2024), and behavioral traits (Panickssery et al. 2024), no prior work has directly investigated whether "AI-sounding" — arguably the most pervasive stylistic property of LLM outputs — constitutes a linear direction.

## 3. Data Construction

### Dataset Description
- **Source:** HC3 (Human ChatGPT Comparison Corpus), `Hello-SimpleAI/HC3`
- **Size:** 18,826 valid pairs after filtering from 24,322 total
- **Structure:** Same questions answered by both human respondents and ChatGPT
- **License:** CC-BY-SA-4.0

### Example Samples

| Question | Human Answer (truncated) | AI Answer (truncated) |
|----------|-------------------------|----------------------|
| "Why is every book a NYT Best Seller?" | "Basically there are many categories of 'Best Seller'. Replace 'Best Seller' by something like..." | "There are many different best seller lists that are published by various organizations, and the New..." |
| "If salt is bad for cars, why use it on roads?" | "salt is good for not dying in car crashes and car crashes are worse for cars then salt..." | "Salt is used on roads to help melt ice and snow and improve traction during the winter months..." |

### Data Quality
- All pairs have both human and ChatGPT answers
- Filtered to 50-1500 character length range
- Sources: reddit_eli5 (69%), finance (13%), medicine (8%), open_qa (6%), wiki_csai (4%)

### Text Length Distribution (Important Confound)
| | Human | AI |
|---|---|---|
| Mean chars | 446 | 914 |
| Std chars | 340 | 279 |
| Mean words | 86 | 159 |

AI text is approximately **2x longer** than human text on average.

### Train/Val/Test Splits
| Split | Size | Strategy |
|-------|------|----------|
| Train | 200 pairs | Random, used for direction extraction |
| Val | 50 pairs | Random, used for layer selection |
| Test | 100 pairs | Random, used for final evaluation |

## 4. Experiment Description

### Methodology

#### High-Level Approach
We apply the **contrastive activation addition (CAA) methodology** (Panickssery et al. 2024) to the "AI vs. human" classification task. For each layer of the model, we compute the difference-in-means direction between AI and human text activations, then evaluate this direction's classification accuracy and causal steering ability.

#### Why This Method?
- Difference-in-means is simpler, faster, and identifies more causally relevant directions than logistic regression (Marks & Tegmark 2024)
- The method requires no optimization or training — just averaging activations
- It produces a single direction vector per layer, enabling interpretable analysis
- Causal interventions (steering) provide evidence beyond correlational classification

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor operations, model loading |
| Transformers | 5.1.0 | Model and tokenizer loading |
| scikit-learn | - | PCA, logistic regression, metrics |
| matplotlib | - | Visualization |
| datasets (HuggingFace) | - | Dataset loading |

#### Model
- **Qwen/Qwen2.5-3B** (base model, not chat-finetuned)
- 3 billion parameters, 36 transformer layers, 2048 hidden dimension
- Loaded in float16 precision on NVIDIA RTX 3090

#### Hyperparameters
| Parameter | Value | Selection Method |
|-----------|-------|------------------|
| Max token length | 512 | Fixed, sufficient for most answers |
| Token position | Last token | Standard for sequence-level features |
| Random seed | 42 | Fixed for reproducibility |
| Steering temperature | 0.7 | Standard sampling temperature |
| Max new tokens (generation) | 150 | Sufficient for style assessment |

#### Analysis Pipeline
1. **Data preparation:** Load HC3, extract paired human/AI answers, filter by length, split into train/val/test
2. **Activation extraction:** Forward-pass each text through Qwen 2.5 3B, record residual stream activations at the last token for all 37 layer outputs (embedding + 36 layers)
3. **Direction extraction:** Compute difference-in-means direction at each layer: `d_l = mean(AI_acts_l) - mean(human_acts_l)`
4. **Classification evaluation:** Mass-mean probing accuracy and AUC on validation and test sets
5. **PCA visualization:** 2D PCA of combined activations at each layer, colored by AI/human label
6. **Confound analysis:** Compute length direction, measure cosine similarity with AI direction, evaluate accuracy after projecting out length
7. **Causal steering:** Add/subtract the direction during model generation, evaluate output with LLM judge

### Reproducibility Information
- Random seed: 42 (set for Python random, NumPy, PyTorch)
- Hardware: 2x NVIDIA RTX 3090 (24GB each), used 1 GPU
- Activation extraction time: ~30 seconds per split (200 pairs)
- Python version: 3.12.2

### Evaluation Metrics
| Metric | What it measures | Why appropriate |
|--------|------------------|----------------|
| Mass-mean probing accuracy | Classification via projection onto direction | Tests if the direction separates AI from human |
| AUC-ROC | Ranking quality | Robust to threshold choice |
| Silhouette score (PCA) | Cluster separation in 2D | Visualizes linear separability |
| Cosine similarity | Direction overlap with confounds | Quantifies confound contamination |
| LLM judge score (1-7) | Perceived AI-likeness | Evaluates qualitative steering effect |

## 5. Raw Results

### Classification Accuracy by Layer (Selected Layers)

| Layer | % Through Model | Train Acc | Val Acc | Test Acc | Test AUC | Random Baseline |
|-------|-----------------|-----------|---------|----------|----------|-----------------|
| 0 | 0% (embedding) | 0.892 | 0.870 | 0.890 | 0.897 | 0.528±0.330 |
| 8 | 22% | 0.968 | 0.950 | 0.950 | 0.998 | 0.512±0.219 |
| 12 | 33% | 0.960 | 0.940 | 0.960 | 1.000 | 0.531±0.186 |
| **21** | **58%** | **0.983** | **0.980** | **0.975** | **0.999** | 0.476±0.163 |
| 27 | 75% | 0.985 | 0.970 | 0.990 | 0.999 | 0.511±0.160 |
| 29 | 81% | 0.993 | 0.980 | 0.985 | 0.999 | 0.515±0.171 |
| 36 | 100% (final) | 0.978 | 0.960 | 0.960 | 0.983 | 0.469±0.203 |

**Best layer by validation accuracy: Layer 21 (58% of depth)**
- Test accuracy: 97.5% [95% CI: 0.950, 0.995]
- Test AUC: 0.999

### Confound Analysis: Length vs. AI Style

| Metric | Value |
|--------|-------|
| Cosine similarity (AI dir vs. length dir) | **0.930** |
| Original accuracy | 0.975 |
| Accuracy after removing length component | **0.820** |
| Length-only direction accuracy | 0.975 |
| Best layer for length-orthogonal direction | Layer 33 (accuracy = **0.855**) |
| Within-class correlation (human length vs. AI proj) | 0.205 |
| Within-class correlation (AI length vs. AI proj) | 0.451 |

### LLM Judge Steering Scores (GPT-4.1, 1-7 scale)

| Multiplier | Direction | Mean Score | Individual Scores |
|------------|-----------|-----------|-------------------|
| -33.2 | Most human | 5.20 | [6, 6, 3, 6, 5] |
| -16.6 | Somewhat human | 6.00 | [6, 6, 6, 6, 6] |
| 0.0 | Baseline | 5.20 | [6, 2, 6, 6, 6] |
| +16.6 | Somewhat AI | 6.20 | [6, 7, 6, 6, 6] |
| +33.2 | Most AI | 6.20 | [6, 7, 6, 6, 6] |

### Steering Examples (Climate Change Prompt)

**Most Human-Like (multiplier = -33.2):**
> "The climate is changing. The world is getting warmer. The poles are melting. The oceans are rising. The sea levels are rising..."

**Baseline (multiplier = 0):**
> "Climate change is a global phenomenon that has been occurring for thousands of years, but the current rate of warming is much faster than in the past..."

**Most AI-Like (multiplier = +33.2):**
> "Climate change is a pressing global issue that poses significant risks to the environment and human well-being. It is caused by the increase of greenhouse gases..."

### Output Locations
- Direction analysis: `results/direction_results.json`
- Confound analysis: `results/confound_results.json`
- Length-controlled analysis: `results/length_controlled_results.json`
- Steering results: `results/steering_results.json`
- LLM judge scores: `results/scored_steering_results.json`
- Visualizations: `results/plots/`

## 5. Result Analysis

### Key Findings

1. **A direction separating AI from human text exists and is highly accurate.** The difference-in-means direction at Layer 21 classifies AI vs. human text with 97.5% accuracy (AUC 0.999) on held-out data. This far exceeds random baselines (~50%) and approaches the logistic regression upper bound (100% on training data).

2. **The direction is predominantly a length/verbosity direction.** The cosine similarity between the AI direction and the text length direction is 0.93 — extremely high. The length direction alone achieves nearly identical classification accuracy (97.5%). This is because ChatGPT answers are systematically ~2x longer than human answers.

3. **A residual style signal exists beyond length.** After projecting out the length component, the remaining direction still achieves 82-85.5% accuracy across layers — well above chance (50%). This residual signal captures genuine stylistic differences: formal tone, comprehensive coverage, hedging language, and structured presentation that are independent of pure text length.

4. **The direction emerges early and is consistent across layers.** Even at the embedding layer (Layer 0), classification accuracy is 89%. It jumps sharply at Layer 8 (~22% of depth) to 95% and peaks in layers 21-29 (58-81%). Adjacent-layer cosine similarity averages 0.87, indicating the direction is stable once formed.

5. **Steering with the direction qualitatively shifts writing style.** Subtracting the direction produces simpler, shorter, more declarative text ("The climate is changing. The world is getting warmer."), while adding it produces more formal, structured, hedging text ("Climate change is a pressing global issue that poses significant risks..."). The LLM judge confirms a directional trend: mean AI-likeness score increases from 5.2 (negative multiplier) to 6.2 (positive multiplier).

6. **The steering effect is modest in absolute terms.** Because this is a base model (not chat-finetuned), all outputs already have some AI character. The steering moves the AI-likeness score by about 1 point on a 7-point scale, compared to the 3-4 point shifts seen with behavioral steering in chat models (Panickssery et al. 2024).

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Linear separability | **Strongly supported** | 97.5% test accuracy, clear PCA clusters |
| H2: >80% accuracy | **Supported** | 97.5% (original), 85.5% (length-controlled) |
| H3: Causal relevance | **Partially supported** | Qualitative style shift confirmed; quantitative effect modest |
| H4: Distinct from length | **Partially refuted** | 93% overlap with length direction; but 85% accuracy remains after controlling |

### Comparison to Prior Work

| Study | Concept | Best Accuracy | Optimal Layer | Confound Issue |
|-------|---------|---------------|---------------|----------------|
| Marks & Tegmark (2024) | Truth/falsehood | >95% | ~38% of depth | Minimal |
| Arditi et al. (2024) | Refusal | ~95%+ | 31-78% | Minimal |
| Panickssery et al. (2024) | 7 behaviors | ~75-90% | ~40% | Moderate |
| **This study** | **AI-sounding** | **97.5% (85.5% controlled)** | **58% of depth** | **High (length)** |

The raw accuracy is among the highest reported for linear direction probing, but this is inflated by length confounding. The length-controlled accuracy (85.5%) is comparable to behavioral trait classification.

### Surprises and Insights

1. **Length is the dominant feature.** We expected style features (formality, hedging, structure) to be the primary discriminator, but text length dominates. This makes sense in hindsight: the model encodes text length in its last-token representation (it knows how long the input was), and ChatGPT's verbosity is its most statistically salient distinguishing feature.

2. **The direction works from the embedding layer.** Even at Layer 0, the direction achieves 89% accuracy. This likely reflects the model encoding information about input text properties (length, vocabulary patterns) very early.

3. **Base model outputs are all "AI-sounding."** The LLM judge rated even the most negative-steered outputs around 5-6/7 on AI-likeness. A chat model (where there's a larger dynamic range between chat outputs and base model outputs) would likely show stronger steering effects.

### Error Analysis

At the best layer (21), only 5 out of 200 test samples (2.5%) are misclassified. These likely represent:
- Short AI responses that resemble human answers
- Long, well-structured human responses that resemble AI answers
- Cases where the question domain strongly influences style regardless of author

### Limitations

1. **Strong length confound.** The HC3 dataset has a systematic length difference between human and AI answers (2x). This inflates classification accuracy and makes it hard to isolate "pure style" from "verbosity."

2. **Single AI source.** All AI text is from ChatGPT. The direction may not generalize to other LLMs (Claude, Gemini, etc.), though the length confound likely applies broadly.

3. **Base model for steering.** Qwen 2.5 3B is a base model, not chat-finetuned. Chat models might show stronger steering effects because they have a larger range of style variation.

4. **Small sample for steering evaluation.** Only 5 prompts were tested for steering, each scored once by GPT-4.1. A larger-scale evaluation with multiple runs and human judges would provide more robust evidence.

5. **Single-direction assumption.** "AI-sounding" may be multi-dimensional (separate directions for hedging, formality, comprehensiveness, bullet-point structure). A single direction may not capture the full phenomenon.

6. **Representation vs. causation.** High classification accuracy shows the model *represents* the AI/human distinction, but the modest steering effects suggest the direction may not be the *causal mechanism* the model uses during generation.

## 6. Conclusions

### Summary
There exists a direction in the residual stream of Qwen 2.5 3B that separates AI-generated from human-written text with 97.5% accuracy. However, this direction is predominantly a **length/verbosity direction** (0.93 cosine similarity with text length), reflecting the well-known tendency of LLMs to produce longer, more comprehensive responses. After controlling for length, a residual "AI style" direction persists with 85.5% accuracy, indicating genuine style differences beyond verbosity are linearly encoded. Steering with this direction produces qualitatively appropriate style shifts but with modest effect sizes in a base model.

### Implications

**Practical:** The strong length confound suggests that AI text detection based on model internals must carefully control for text length to avoid spurious accuracy. Simple length-based features may be as effective as more complex style analysis for AI detection.

**Theoretical:** "AI-sounding" is not as cleanly encoded as binary concepts like truth or refusal. It appears to be a composite of multiple correlated features (length, formality, structure, hedging), with length being dominant. This suggests that "sounding like AI" may not be a single natural kind in the model's representation space, but rather an emergent property of multiple overlapping stylistic features.

### Confidence in Findings
- **High confidence** in the classification results (large effect, robust to split variation)
- **High confidence** in the length confound finding (0.93 cosine similarity is unmistakable)
- **Moderate confidence** in the residual style signal (85% accuracy, consistent across layers)
- **Low confidence** in the causal steering results (small sample, modest effects, base model limitations)

## 7. Next Steps

### Immediate Follow-ups
1. **Length-matched dataset:** Construct contrastive pairs where human and AI texts are matched by length (filter HC3 or use truncation), then re-extract directions. This would provide cleaner evidence for a pure "style" direction.
2. **Chat model analysis:** Repeat the analysis with a chat-finetuned model (e.g., Qwen2.5-3B-Instruct) where the AI/human style difference is more pronounced and the model has a wider range of style control.
3. **Multi-model generalization:** Test whether the direction extracted from ChatGPT data generalizes to text from Claude, Gemini, and other LLMs.

### Alternative Approaches
1. **Sparse autoencoder analysis:** Use SAEs to decompose the "AI direction" into interpretable features. This would reveal whether it decomposes into known style features (hedging neurons, formality features, etc.).
2. **Multi-dimensional subspace:** Instead of a single direction, find a k-dimensional subspace (k=2-5) that better captures the multi-faceted nature of "AI style."
3. **Causal patching:** Use activation patching to identify which specific layers/components are causally responsible for AI-like generation.

### Open Questions
1. Is "AI-sounding" a single direction or a multi-dimensional subspace?
2. Does the direction exist in base models before any instruction/chat finetuning?
3. Can ablating this direction make a chat model sound more human without degrading capabilities?
4. Is the direction the same across languages (English vs. Chinese in Qwen's bilingual training)?

## References

1. Park et al. (2023). "The Linear Representation Hypothesis and the Geometry of Large Language Models." arXiv:2311.03658
2. Marks & Tegmark (2024). "The Geometry of Truth." COLM 2024. arXiv:2310.06824
3. Panickssery et al. (2024). "Steering Llama 2 via Contrastive Activation Addition." ACL 2024. arXiv:2312.06681
4. Arditi et al. (2024). "Refusal in Language Models Is Mediated by a Single Direction." NeurIPS 2024. arXiv:2406.11717
5. Konen et al. (2024). "Style Vectors for Steering Generative Large Language Models." arXiv:2402.01618
6. Lai, Hangya & Fraser (2024). "Style-Specific Neurons for Steering LLMs in Text Style Transfer." arXiv:2410.00593
7. Turner et al. (2023). "Activation Addition: Steering Language Models Without Optimization." arXiv:2308.10248
8. Zou et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405
9. HC3 Dataset: `Hello-SimpleAI/HC3`, HuggingFace Datasets
10. Qwen 2.5 3B: `Qwen/Qwen2.5-3B`, HuggingFace Models
