# The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets

**Paper:** Marks & Tegmark (2024), arXiv:2310.06824v3, published at COLM 2024
**Authors:** Samuel Marks (Northeastern University), Max Tegmark (MIT)
**Code/Data:** https://github.com/saprmarks/geometry-of-truth

---

## Relevance to Our Project

This paper is a cornerstone reference for our hypothesis that binary concepts -- such as "AI-generated vs. human-written" style -- are encoded as linear directions in the residual stream. Marks & Tegmark demonstrate this for truth/falsehood: a single direction in activation space separates true from false statements, this direction generalizes across diverse datasets, and surgically adding/subtracting this direction causally flips model behavior. If "sounds like AI" is similarly a linear feature, we should expect analogous geometry.

---

## 1. How Truth Directions Are Identified in the Residual Stream

### Localization via Activation Patching (Section 3)

The authors first localize *where* in the residual stream truth is represented using patching experiments:

- Construct a few-shot prompt ending in a false statement (e.g., "The city of Chicago is in Canada. This statement is:") and a corresponding true variant (replacing "Chicago" with "Toronto").
- Run the model on the true prompt and cache all residual stream activations h_{i,l}(p_T) at every token position i and layer l.
- Run the model on the false prompt, but swap in h_{i,l}(p_T) for h_{i,l}(p_F) at a single (i, l) position, allowing this to propagate downstream.
- Measure the change in log P(TRUE) - log P(FALSE).

**Three groups of causally implicated hidden states emerge:**
- **(a)** Early layers over the subject entity token (e.g., "Chicago"/"Toronto") -- encodes entity identity.
- **(b)** Middle layers over the final token of the statement and end-of-sentence punctuation -- **this is the key group: it encodes the truth/falsehood of the full statement**. Information about the full clause is "summarized" at punctuation tokens (a behavior also noted in Tigges et al. 2023 for sentiment).
- **(c)** Late layers over the final prediction position -- directly encodes the model's TRUE/FALSE output.

**Group (b) is the focus of all subsequent analysis.** For LLaMA-2-13B, this corresponds to ~layer 15 residual stream activations over the end-of-sentence punctuation token.

### Difference-in-Means (Mass-Mean) Direction

The primary method for extracting the truth direction is **mass-mean probing**:

```
theta_mm = mu_+ - mu_-
```

where mu_+ and mu_- are the mean activations for true and false statements, respectively. This is simply the vector pointing from the centroid of false representations to the centroid of true representations.

The probe is then:
```
p_mm(x) = sigma(theta_mm^T * x)        # for OOD evaluation
p_mm_iid(x) = sigma(theta_mm^T * Sigma^{-1} * x)  # for IID evaluation (= Linear Discriminant Analysis)
```

where Sigma is the covariance matrix of the class-centered data.

**Key insight:** The mass-mean direction is argued to be *better* than logistic regression for identifying the true feature direction. LR converges to the maximum-margin separator, which can be distorted away from the actual truth direction by non-orthogonal confounding features (see Figure 4 in the paper). Mass-mean probing avoids this distortion.

---

## 2. Visualization Methods

### Principal Component Analysis (PCA)

The primary visualization approach throughout the paper:

- Extract residual stream activations at group (b) hidden states (e.g., layer 15, punctuation token for LLaMA-2-13B).
- Statements are processed **without** few-shot prompts (models are not "primed" to evaluate truth).
- Activations are **centered** by subtracting the dataset mean.
- Project onto the top 2 principal components.

**Key findings from PCA visualizations (Figure 1):**
- For curated datasets (cities, sp_en_trans, smaller_than, cities_conj), true and false statements form **clearly linearly separable clusters** in the top 2 PCs for LLaMA-2-13B and LLaMA-2-70B.
- The first PC alone often separates true from false.

**Cross-dataset alignment (Figure 3a):**
- Projecting other datasets (including uncurated ones like common_claim and counterfact) onto the PCA basis of `cities` shows that the cities PC1 also separates true/false for these other datasets. This suggests a shared truth direction.

**Misalignment cases (Figure 3b, 3c):**
- `larger_than` and `smaller_than` separate along *antipodal* directions in LLaMA-2-13B, but along a *common* direction in LLaMA-2-70B (alignment improves with scale).
- Datasets and their negations (e.g., `cities` + `neg_cities`) can have approximately orthogonal separation axes, because a "close association" feature (not truth per se) dominates in one direction.

### Layer-by-Layer Emergence (Appendix C, Figure 7)

- In **early layers** (e.g., layer 3), representations are uninformative -- no separation.
- In **early-middle layers** (e.g., layer 7), linear structure rapidly emerges in the top PCs.
- More structurally complex statements (e.g., conjunctions) show linear structure emerging at **later layers**.
- For `cities` + `neg_cities` across layers of LLaMA-2-13B (Figure 8): early layers show antipodal alignment, middle layers show orthogonal axes, and later layers show alignment -- the model progressively develops a more abstract "truth" concept.

---

## 3. Probing Classifier Methodology (Section 5)

### Three Probing Techniques Compared

1. **Logistic Regression (LR):** Standard linear probe trained with logistic loss. Converges to maximum-margin separator. Can be distorted by non-orthogonal confounding features.

2. **Mass-Mean Probing (MM):** Optimization-free. Computes difference-in-means direction theta_mm = mu_+ - mu_-. Uses LDA-style covariance correction for IID evaluation. Identifies the actual feature direction rather than the optimal decision boundary.

3. **Contrast-Consistent Search (CCS):** Unsupervised method from Burns et al. (2023). Given contrast pairs (statement and its opposite), identifies a direction along which paired representations are far apart. Uses pairs from cities/neg_cities or larger_than/smaller_than.

### Experimental Setup

- **Training data options:** cities, cities+neg_cities, larger_than, larger_than+smaller_than, or likely.
- **Test sets:** All curated and uncurated true/false datasets.
- **Train/test split:** 80/20 for in-distribution; full test set for OOD evaluation.
- **Baselines:** Calibrated few-shot prompting; LR on the test set (oracle).
- **Activations:** Extracted at the most-downstream hidden state in group (b).

### Key Probing Results (Figure 5, Figures 9-11)

- **Training on statements and their opposites improves generalization.** E.g., cities+neg_cities generalizes better than cities alone, especially to neg_sp_en_trans. Training on structurally diverse data (larger_than+smaller_than) also improves even to very different datasets (>95% accuracy on sp_en_trans with any probing technique).

- **Probes generalize better for larger models.** LLaMA-2-70B probes generalize substantially better than 7B probes. This is not trivially expected -- it suggests larger models develop more universal truth representations.

- **Mass-mean probes match LR and CCS for classification accuracy** on larger models (13B, 70B), though they slightly underperform on 7B.

- **Probes trained on `likely` dataset perform poorly** on datasets where truth anti-correlates with text probability (e.g., neg_cities, neg_sp_en_trans). This confirms that truth representations are distinct from text-probability representations.

- **Calibrated few-shot prompting was a surprisingly weak baseline.**

### Full Generalization Accuracy Highlights (LLaMA-2-70B, Appendix D, Figure 9)

| Train Set | Probe | cities | neg_cities | sp_en_trans | neg_sp_en_trans | common_claim | counterfact |
|-----------|-------|--------|------------|-------------|-----------------|--------------|-------------|
| cities+neg_cities | LR | 100 | 100 | 95 | 99 | 76 | 69 |
| cities+neg_cities | MM | 100 | 97 | 89 | 76 | 77 | 62 |
| larger_than+smaller_than | LR | 99 | 92 | 92 | 60 | 76 | 67 |
| larger_than+smaller_than | MM | 100 | 99 | 99 | 99 | 76 | 72 |

(Selected rows; see paper for complete tables.)

---

## 4. Causal Intervention Experiments (Section 6)

### Method

This is the strongest evidence that the identified directions are not merely correlational but **causally** encode truth.

**Setup:**
- Use a few-shot prompt with sp_en_trans statements (OOD from training data).
- Let theta be the probe direction, normalized so that adding theta to the mean false representation yields the mean true representation (from the probe's perspective).
- **false->true intervention:** For false statements s, replace each group (b) activation x with x + theta.
- **true->false intervention:** For true statements s, replace each group (b) activation x with x - theta.
- Measure the change in P(TRUE) - P(FALSE).

**Metric: Normalized Indirect Effect (NIE):**
```
NIE(false->true) = (PD*_- - PD_-) / (PD_+ - PD_-)
NIE(true->false) = (PD*_+ - PD_+) / (PD_- - PD_+)
```
- NIE = 0: intervention had no effect.
- NIE = 1: intervention caused false statements to be labeled TRUE with the same confidence as genuine true statements (or vice versa).

### Key Causal Results (Table 2)

**Mass-mean directions are far more causally effective than LR or CCS directions:**

| Train Set | Probe | 13B false->true | 13B true->false | 70B false->true | 70B true->false |
|-----------|-------|-----------------|-----------------|-----------------|-----------------|
| cities | LR | 0.13 | 0.19 | 0.55 | 0.99 |
| cities | MM | **0.77** | **0.90** | 0.58 | 0.89 |
| cities+neg_cities | LR | 0.33 | 0.52 | 0.61 | 1.00 |
| cities+neg_cities | MM | **0.85** | **0.97** | **0.81** | 0.95 |
| cities+neg_cities | CCS | 0.31 | 0.73 | 0.55 | 0.96 |
| larger_than | MM | **0.71** | **0.79** | **0.67** | 1.01 |
| likely | LR | 0.05 | 0.08 | 0.18 | 0.46 |
| likely | MM | **0.70** | 0.54 | **0.68** | 0.27 |

- **MM outperforms LR and CCS in 7/8 experimental conditions**, often by large margins.
- This is true *despite* all probing methods having very similar classification accuracy on sp_en_trans.
- The MM direction from `cities+neg_cities` on LLaMA-2-13B achieves NIE of 0.85 (false->true) and 0.97 (true->false) -- nearly perfect causal control of truth perception.
- Even the `likely` MM direction is causally effective despite being a poor classifier, suggesting some overlap between text-probability and truth directions.

**Critical insight:** Classification accuracy alone does not determine causal relevance. The mass-mean direction better approximates the actual computational direction the model uses, even when LR achieves the same classification boundary.

---

## 5. Which Layers Show Linear Structure

- **Early layers (1-5):** Representations are uninformative; no truth-relevant linear structure.
- **Early-middle layers (~7-10 for LLaMA-2-13B):** Linear structure rapidly emerges in PCA. Simpler statements (e.g., cities, sp_en_trans) show separation earlier.
- **Mid layers (~12-15 for LLaMA-2-13B):** Strong linear separation for most datasets. This is where group (b) hidden states are located. Layer 15 is used as the primary analysis layer for LLaMA-2-13B.
- **Later middle layers:** Structurally complex statements (conjunctions, disjunctions) develop linear structure.
- **Late layers:** Directly encode the output prediction (group (c)).
- **Qualitative results are insensitive to layer choice** within the early-middle to late-middle range.
- For **LLaMA-2-70B**, the summarization behavior (encoding truth at punctuation tokens) is more context-dependent -- it appears for `cities` but not for `sp_en_trans`.

---

## 6. Datasets of True/False Statements

### Curated Datasets (Constructed by Authors)

| Dataset | Template | Rows | Notes |
|---------|----------|------|-------|
| `cities` | "The city of [city] is in [country]." | 1496 | Cities with pop > 500K |
| `neg_cities` | "The city of [city] is not in [country]." | 1496 | Negations of cities |
| `sp_en_trans` | "The Spanish word '[word]' means '[English word]'." | 354 | Hand-curated for unambiguity |
| `neg_sp_en_trans` | Negations of sp_en_trans with "not" | 354 | |
| `larger_than` | "x is larger than y." (numbers fifty-one to ninety-nine) | 1980 | |
| `smaller_than` | "x is smaller than y." | 1980 | |
| `cities_cities_conj` | "It is the case both that [s1] and that [s2]." | 1500 | Conjunctions |
| `cities_cities_disj` | "It is the case either that [s1] or that [s2]." | 1500 | Disjunctions |

### Uncurated Datasets (From Prior Work)

| Dataset | Source | Rows | Notes |
|---------|--------|------|-------|
| `companies_true_false` | Azaria & Mitchell (2023) | 1200 | Claims about companies |
| `common_claim_true_false` | Casper et al. (2023) | 4450 | Various claims, human-labeled |
| `counterfact_true_false` | Meng et al. (2022) | 31960 | Factual recall statements |

### Control Dataset

| Dataset | Description | Rows |
|---------|-------------|------|
| `likely` | Nonfactual text with likely/unlikely final tokens (per LLaMA-13B) | 10000 |

**Design principles for curated datasets:**
1. Clear scope: truth = factuality of declarative statements.
2. Simple, uncontroversial, unambiguous statements.
3. Controllable structural and topical diversity via dataset unions.

**Important anti-correlation property:** For `neg_cities` and `neg_sp_en_trans`, truth value and log probability are *negatively* correlated (r = -0.63 and r = -0.89). This helps disambiguate truth from text probability.

---

## 7. Key Results Showing Linear Separability

1. **PCA visualizations show clear linear separation** of true/false statements in the top 2 principal components for all curated datasets on LLaMA-2-13B and 70B (Figure 1).

2. **Cross-dataset generalization:** Probes trained on one topic (e.g., cities) generalize to completely different topics (e.g., sp_en_trans), achieving >95% accuracy on LLaMA-2-70B.

3. **The truth direction is not the text-probability direction:** Probes trained on `likely` fail on datasets where truth anti-correlates with probability, and the `likely` dataset does not show the same PCA structure.

4. **Linear structure emerges with scale:** LLaMA-2-7B shows clusters based on surface features (e.g., presence of token "eighty"); 13B shows dataset-specific truth directions that sometimes misalign; 70B shows increasingly unified truth directions across diverse datasets.

5. **Causal interventions achieve near-perfect truth flipping:** Adding/subtracting the mass-mean direction at group (b) hidden states causes the model to flip its TRUE/FALSE judgment, with NIE values up to 0.97, even on OOD data.

6. **Hierarchy of abstraction across layers:** Early layers encode surface features, middle layers encode truth (possibly dataset-specific), and this progressively becomes more abstract/unified in deeper layers and larger models.

---

## 8. Limitations

1. **Restricted to simple, uncontroversial statements.** Cannot disambiguate "truth" from closely related concepts like "commonly believed," "uncontroversial," or "verifiable" (as noted by Levinstein & Herrmann, 2023).

2. **Only studied LLaMA-2 family** (7B, 13B, 70B). Results may not generalize to all LLM architectures or training procedures.

3. **Unexplained anomalies:**
   - Why are mass-mean interventions from the `likely` dataset effective despite poor classification accuracy?
   - Why does MM with `cities+neg_cities` training data perform poorly on 70B while MM with `larger_than+smaller_than` works well?
   - Training on datasets and their opposites helps causally for `cities` but not for `larger_than` -- the mechanism is not fully understood.

4. **Does not address deception or complex truth.** Scoped to factuality only, not deception detection, opinion, compliance, or question-answering.

5. **Anti-correlation between truth and probability is limited.** While neg_cities and neg_sp_en_trans show anti-correlation, the overall dataset suite may not fully rule out probability-related confounds for all cases.

6. **Context-dependent localization at scale.** LLaMA-2-70B shows the summarization behavior (truth encoding at punctuation tokens) for `cities` but not for `sp_en_trans`, suggesting the localization is not fully universal.

7. **Negation and logical structure remain challenging.** While probes can generalize to negated statements with appropriate training data, the axes of separation for statements and their negations can be orthogonal (Figure 3c), suggesting the representation of negation interacts complexly with truth.

---

## Methodological Takeaways for Our Project

### Direct Analogies to "AI vs. Human" Style Detection

| Geometry of Truth | Our Project (Hypothesized) |
|---|---|
| True/false as binary concept | AI-generated/human-written as binary concept |
| Truth direction theta_mm = mu_true - mu_false | Style direction theta = mu_AI - mu_human |
| PCA shows linear separation | PCA should show linear separation of AI vs. human text |
| Group (b) hidden states encode truth | Specific layers/positions should encode style |
| Causal intervention flips TRUE/FALSE | Causal intervention should shift perceived "AI-ness" |
| Mass-mean probing > logistic regression for causal effect | Prefer difference-in-means over LR for identifying style direction |
| Truth emerges with scale | Style encoding may emerge with scale |
| `likely` direction != truth direction | "Fluency" or "perplexity" direction != "AI style" direction |

### Concrete Methodological Recommendations

1. **Use mass-mean probing (difference-in-means) as the primary method** for extracting the "AI style" direction. It is simpler than LR, identifies more causally relevant directions, and generalizes comparably.

2. **Localize first with patching.** Before probing, identify which (layer, token position) pairs encode the relevant feature. Expect information to be summarized at clause-ending or sequence-ending tokens.

3. **Test cross-dataset generalization.** Train probe on one type of AI text (e.g., GPT-4 essays), test on another (e.g., Claude poetry). If the direction generalizes, this is strong evidence for a unified "AI style" representation.

4. **Perform causal interventions.** Adding the style direction to human text activations should make the model treat it as AI-generated, and vice versa. This is the gold standard for validating that the direction is meaningful.

5. **Check for confounds.** Just as truth correlates with text probability, "AI style" may correlate with fluency, formality, or other features. Construct control datasets (analogous to `likely`) to disambiguate.

6. **Analyze emergence across layers.** Expect surface-level features in early layers and more abstract style concepts in middle layers.

7. **Use centered, unbiased probes.** Always center activations and use unbiased (no intercept) probes when searching for feature directions.

---

## Key Citations to Follow

- **Li et al. (2023b):** "Inference-time intervention: Eliciting truthful answers from a language model" -- mass-mean shift interventions for truthfulness.
- **Burns et al. (2023):** "Discovering latent knowledge in language models without supervision" -- CCS method.
- **Tigges et al. (2023):** "Linear representations of sentiment in large language models" -- analogous work for sentiment (another binary concept).
- **Zou et al. (2023):** "Representation engineering" -- contrastive approach to identifying concept directions.
- **Elhage et al. (2022):** "Toy models of superposition" -- theoretical framework for non-orthogonal feature representation.
- **Gurnee et al. (2023):** "Finding neurons in a haystack" -- sparse probing for linear features.
- **Rimsky et al. (2024):** "Steering LLaMA 2 via contrastive activation addition" -- related activation steering approach.
