# Research Plan: Is There a "Sounds Like AI" Direction in the Residual Stream?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs produce text with distinctive stylistic markers — hedging language, list structures, over-formality, and characteristic phrases — that humans readily identify as "AI-sounding." Understanding whether this quality is encoded as a linear direction in the residual stream would (1) reveal how style is computationally represented in transformers, (2) enable steering models to produce more natural or more robotic text, and (3) provide insights into the fundamental geometry of style in neural language models.

### Gap in Existing Work
The literature review reveals extensive work on linear directions for truth/falsehood (Marks & Tegmark 2024), refusal behavior (Arditi et al. 2024), sentiment (Konen et al. 2024), and behavioral traits like sycophancy (Panickssery et al. 2024). However, **no prior work has directly investigated whether "sounding like AI" — the composite stylistic quality that makes LLM outputs identifiable — constitutes a linear direction in the residual stream.** This is a natural and important extension: we know style is linearly represented, we know AI text has detectable style markers, but the connection has not been empirically tested with mechanistic interpretability tools.

### Our Novel Contribution
We directly test whether the "AI-sounding" quality is a linear feature in the residual stream by:
1. Extracting a difference-in-means direction from contrastive AI/human text pairs
2. Validating its linear separability via PCA across layers
3. Testing causal relevance through activation addition (can we make human text sound more AI?) and directional ablation (can we make AI text sound more human?)
4. Checking whether this direction is distinct from simpler confounds like formality or text length

### Experiment Justification
- **Experiment 1 (Direction Extraction & Probe):** We need to establish whether the direction exists at all — do AI and human text activations separate linearly?
- **Experiment 2 (Layer-wise Analysis):** Where in the network does this direction emerge? This localizes the computation and informs steering.
- **Experiment 3 (Causal Intervention):** Classification accuracy alone doesn't prove the direction is causally relevant (Marks & Tegmark 2024). We must show that adding/removing the direction changes model behavior.
- **Experiment 4 (Confound Analysis):** We must disentangle "AI-sounding" from correlated features like formality, length, and fluency.

---

## Research Question
Does there exist a linear direction in the residual stream of transformer language models that corresponds to text "sounding like AI," and is this direction causally relevant to the model's generation of AI-like vs human-like text?

## Background and Motivation
LLMs exhibit linear structure in their residual streams for many high-level concepts. Truth, refusal, sycophancy, and sentiment are all represented as directions that can be extracted via difference-in-means and used for causal steering. "Sounding like AI" is arguably the most pervasive stylistic property of LLM outputs — it encompasses hedging ("I'd be happy to help"), list formatting, over-formality, and epistemic hedging. If this composite quality has a linear representation, it opens avenues for both understanding style encoding and practically improving LLM naturalness.

## Hypothesis Decomposition
1. **H1 (Linear Separability):** Residual stream activations for AI-generated and human-written text are linearly separable, with separation emerging in middle layers.
2. **H2 (Direction Extraction):** A difference-in-means direction can classify held-out AI vs. human text with >80% accuracy.
3. **H3 (Causal Relevance):** Adding the "AI direction" to human text activations during generation makes output sound more AI-like; subtracting it from AI text makes output sound more human-like.
4. **H4 (Distinctness from Confounds):** The AI direction is not simply a proxy for formality, text length, or perplexity.

## Proposed Methodology

### Approach
We use the **contrastive activation addition (CAA) methodology** — the established approach for extracting linear directions. We work with a medium-sized open model (Gemma 2 2B or Qwen 2.5 3B) that fits comfortably on our RTX 3090 GPUs. We use the HC3 dataset (paired human/ChatGPT answers to the same questions) as contrastive pairs.

**Why this approach:**
- Difference-in-means is simpler, faster, and more causally relevant than learned classifiers (Marks & Tegmark 2024)
- HC3 provides naturally controlled pairs — same question, different author
- Small-to-medium models allow rapid iteration and full activation storage
- The steering-vectors library provides a clean API for extraction and intervention

### Model Choice
We use **Gemma 2 2B** (google/gemma-2-2b) as the primary model:
- Small enough to fit on a single RTX 3090 with room for activation storage
- Modern architecture with good representational quality
- Well-supported by HuggingFace and TransformerLens
- 26 layers — enough depth for meaningful layer-wise analysis

If Gemma 2 2B is unavailable or has issues, fallback to **Qwen 2.5 3B** or **Pythia 2.8B**.

### Experimental Steps

#### Step 1: Data Preparation
- Download HC3 dataset using provided download script
- Extract paired (human_answer, chatgpt_answer) for same questions
- Filter for English, reasonable length (50-500 tokens)
- Create train (200 pairs), validation (50 pairs), test (100 pairs) splits
- Ensure topic diversity across splits

#### Step 2: Activation Extraction
- Load model with float16 precision
- Forward-pass each text through the model
- Record residual stream activations at all layers at the last token position
- Store activations as tensors for subsequent analysis

#### Step 3: Direction Extraction
- Compute difference-in-means direction at each layer: `d_l = mean(AI_acts_l) - mean(human_acts_l)`
- Compute classification accuracy using mass-mean probing on validation set
- Select best layer based on validation accuracy

#### Step 4: PCA Visualization
- For each layer, compute PCA of combined (AI + human) activations
- Plot 2D PCA showing AI vs human clusters
- Identify layers where clear separation emerges

#### Step 5: Causal Intervention (Steering)
- Use the extracted direction to steer model generation:
  - **Addition:** Add AI direction to human-style prompts → does output become more AI-like?
  - **Subtraction:** Subtract AI direction from AI-style prompts → does output become more human-like?
- Evaluate steering effects with both automated metrics and qualitative inspection
- Use an external classifier (or API-based LLM judge) to score "AI-likeness" of steered outputs

#### Step 6: Confound Analysis
- Compute cosine similarity between AI direction and:
  - Text length direction (long vs short text activations)
  - Formality direction (formal vs informal text activations)
- Test whether the AI direction predicts AI-likeness after controlling for these confounds

### Baselines
1. **Random direction:** Random unit vector in activation space — should give ~50% accuracy
2. **PCA first component:** First principal component of all activations — captures variance but may not correspond to AI/human distinction
3. **Logistic regression:** Trained classifier — upper bound on linear separability

### Evaluation Metrics
- **Classification accuracy** of the direction on held-out test set (mass-mean probing)
- **AUC-ROC** for the linear probe
- **PCA cluster separation** (visual + silhouette score)
- **Cosine similarity** between AI directions across layers (consistency)
- **Steering effectiveness:** Qualitative assessment + LLM-judge scoring of steered outputs
- **Confound orthogonality:** Cosine similarity with known confound directions

### Statistical Analysis Plan
- Report accuracy with 95% confidence intervals via bootstrap (1000 resamples)
- Compare against random baseline using permutation test
- Use cosine similarity distributions to assess cross-layer consistency
- Significance level: α = 0.05

## Expected Outcomes
- **Supporting H1:** PCA plots show clear AI/human clusters in middle layers (30-70% of depth)
- **Supporting H2:** Difference-in-means direction achieves >80% accuracy on held-out data
- **Supporting H3:** Adding AI direction to generation produces noticeably more AI-like text
- **Refuting hypothesis:** If accuracy is near chance, or if the direction is indistinguishable from formality/length, the "AI-sounding" quality may not have a unitary linear representation

## Timeline and Milestones
1. **Data prep & environment setup:** 15 min
2. **Activation extraction:** 20 min
3. **Direction extraction & PCA analysis:** 15 min
4. **Causal steering experiments:** 30 min
5. **Confound analysis:** 15 min
6. **Documentation & report:** 30 min

## Potential Challenges
1. **Model memory:** Storing activations for all layers × all samples requires careful batching
2. **Domain confounding:** HC3 answers may differ in topic coverage, not just style
3. **AI-sounding may be multi-dimensional:** May need PCA subspace rather than single direction
4. **Steering may hurt coherence:** Adding/subtracting directions can cause degenerate text

## Success Criteria
1. A clearly identifiable direction with >75% classification accuracy
2. PCA visualizations showing layer-dependent emergence of the direction
3. At least qualitative evidence that steering with this direction affects perceived AI-likeness
4. Evidence that the direction is at least partially distinct from simpler confounds
