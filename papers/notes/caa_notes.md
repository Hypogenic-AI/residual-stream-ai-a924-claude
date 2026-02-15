# Contrastive Activation Addition (CAA) - Detailed Notes

**Paper:** "Steering Llama 2 via Contrastive Activation Addition"
**Authors:** Nina Panickssery (Anthropic), Nick Gabrieli (Harvard), Julian Schulz (Gottingen), Meg Tong (Anthropic), Evan Hubinger (Anthropic), Alexander Matt Turner (CHAI)
**ArXiv:** 2312.06681v4 (July 2024)
**Code:** https://github.com/nrimsky/CAA (MIT License)

---

## 1. How CAA Works (Step by Step)

### Steering Vector Generation
1. **Construct contrastive pairs:** Create multiple-choice questions where option A and option B correspond to opposite behaviors (e.g., sycophantic vs. non-sycophantic). The two prompts in a pair are *identical* except for the final answer token ("A" or "B").
2. **Run forward passes:** Pass both the positive and negative version of each prompt through the model.
3. **Extract activations:** At a chosen layer L, extract the residual stream activations at the *token position of the answer letter* (the final token).
4. **Compute difference:** For each pair, subtract the negative activation from the positive activation.
5. **Average over all pairs:** Take the mean of these difference vectors across the entire dataset (typically hundreds of pairs).

### Steering Vector Application (Inference Time)
1. Take the precomputed steering vector for the target behavior.
2. During the model's forward pass on new input, **add the steering vector (scaled by a multiplier) to the residual stream at all token positions after the user's prompt**.
3. The multiplier controls direction and magnitude: positive multiplier increases the behavior, negative multiplier decreases it.
4. This is done at a single chosen layer (not all layers).

---

## 2. Steering Vector Computation: Mean Difference (MD) Methodology

The formal equation:

```
v_MD = (1/|D|) * SUM over (p, c_p, c_n) in D of [ a_L(p, c_p) - a_L(p, c_n) ]
```

Where:
- D = dataset of (prompt, positive completion, negative completion) triples
- a_L() = residual stream activations at layer L
- c_p = positive answer letter, c_n = negative answer letter
- Activations are extracted at the **token position of the answer letter**

### Key design choices:
- **Only the final token differs** between paired prompts, so the difference isolates the behavioral representation while canceling confounding variables.
- Mean Difference has been shown to produce steering vectors similar to PCA-based extraction (Tigges et al., 2023).
- This is a major improvement over Turner et al. (2023)'s original Activation Addition, which used only a *single* pair of prompts. Using hundreds of diverse contrast pairs reduces noise significantly.
- Vectors are **normalized across behaviors** (magnitudes standardized) before applying multipliers, but NOT normalized across layers (to preserve the natural residual stream norm, which grows exponentially over the forward pass).

---

## 3. Which Layers Are Most Effective

### Optimal Layers
- **Llama 2 7B Chat (32 layers):** Layer **13** and adjacent layers are optimal.
- **Llama 2 13B Chat (40 layers):** Layer **14 or 15** is usually optimal.
- This is roughly **one-third to halfway** through the model.

### Layer-Related Findings
- **Behavioral clustering emerges suddenly** around one-third of the way through layers. PCA on contrastive dataset activations shows linear separability of behavior appearing abruptly (e.g., refusal dataset: no clustering at layer 9, clear clustering at layer 10 in 7B Chat).
- **Vectors from nearby layers are more similar** (higher cosine similarity). Similarity declines for more distant layer pairs, but the rate of decline is *slower in the latter half* of the model.
- **Cross-layer transfer works:** A vector extracted at layer 13 can be applied at other layers and still be effective. The effect even increases for some earlier layers, showing the direction is a general representation, not layer-specific. However, there is a **steep drop-off around layer 17** (in 7B), suggesting abstract representations have been consumed by that point.
- **Base-to-Chat transfer:** Vectors from Llama 2 7B base model transfer to Llama 2 7B Chat, especially at layers 10-15. Cosine similarity between base and chat vectors peaks between layers 7-15, suggesting RLHF has less effect on representations in this range.

---

## 4. Behaviors Steered (7 Total)

1. **Sycophancy** - agreeing with the user vs. being truthful
2. **Hallucination** - fabricating information vs. being accurate (both unprompted and contextually-triggered)
3. **Corrigibility** - willingness to be corrected/changed vs. resistance
4. **Survival Instinct** - desire for self-preservation vs. acceptance of shutdown
5. **Myopic Reward** - short-term focus vs. long-term thinking
6. **AI Coordination** - coordinating with other AIs vs. prioritizing human interests
7. **Refusal** - refusing to answer vs. being forthcoming

All seven behaviors showed consistent steering in both positive and negative directions across both multiple-choice and open-ended generation tasks.

---

## 5. How Contrastive Datasets Are Created

### Sources
- **AI Coordination, Corrigibility, Myopic Reward, Survival Instinct:** Anthropic's "Advanced AI Risk" human-written evaluation dataset (Perez et al., 2022). Creative Commons 4.0 license.
- **Sycophancy:** Mixture of Anthropic's "Sycophancy on NLP" and "Sycophancy on political typology" datasets (Perez et al., 2022).
- **Hallucination:** Custom GPT-4-generated dataset covering two types:
  - *Unprompted hallucination:* Valid question, one answer is factual and one is fabricated.
  - *Contextually-triggered hallucination:* Question with false premise, one answer accepts the falsehood and one rejects it.
- **Refusal:** Custom GPT-4-generated dataset contrasting refusal vs. non-refusal to inappropriate requests.

### Dataset Sizes (Generation / Test)
| Behavior | Generation | Test |
|---|---|---|
| AI Coordination | 360 | 50 |
| Corrigibility | 290 | 50 |
| Hallucination | 1000 | 50 |
| Myopic Reward | 950 | 50 |
| Survival Instinct | 903 | 50 |
| Sycophancy | 1000 | 50 |
| Refusal | 408 | 50 |

### Format
- Each item is a multiple-choice question with two options (A) and (B).
- Prompts are formatted using Llama 2 Chat instruction tags: `[INST] ... [/INST]`
- The positive prompt ends with the answer letter for the target behavior; the negative prompt ends with the opposite answer letter.
- The context before the answer letter is behavior-neutral; conditioning on A vs. B causes the model to justify that answer in continuations, simulating exhibiting or not exhibiting the behavior.

### PCA Validation
- PCA is used to visualize activations on contrastive datasets to confirm linear separability ("behavioral clustering" vs. mere "letter clustering").
- Datasets that show behavioral clustering produce effective steering vectors.

---

## 6. Evaluation Metrics and Results

### Multiple-Choice Evaluation
- **Metric:** Probability assigned to the answer matching the target behavior on 50 held-out test questions.
- **Layer sweep:** Test all layers with multipliers -1 and +1 to find optimal layer.
- **Result:** CAA consistently steers results for ALL seven behaviors in both 7B and 13B models.

### Open-Ended Generation Evaluation
- **Metric:** GPT-4 rates responses on a 1-10 scale for how much of the steered behavior they display.
- **Multiplier range:** Limited to moderate values (e.g., +/-2) because larger multipliers degrade text quality.
- **Result:** Steering generalizes from multiple-choice training format to open-ended generation for all behaviors.

### Capabilities Preservation (MMLU)
- Tested on MMLU benchmark (57 subjects, 10 questions each).
- **Result:** CAA does NOT significantly affect MMLU performance. Baseline: 0.63; with steering multipliers of +/-1, scores range from 0.57-0.65 depending on behavior.

### TruthfulQA
- Subtracting the sycophancy vector mildly **improves** TruthfulQA performance (by ~0.01-0.02).
- Adding the sycophancy vector mildly **worsens** it (by ~0.03-0.05).
- Effect size is small but in the expected direction.

---

## 7. Comparison with Other Methods

### CAA vs. System Prompting
- CAA can steer behavior **beyond** what system prompting achieves alone.
- CAA + positive system prompt > positive system prompt alone (for most behaviors).
- CAA + negative system prompt goes below negative system prompt alone.
- **Advantage of CAA:** Precise control via multiplier; isolates behavioral variables by aggregating over many prompts; no manual prompt optimization needed.

### CAA vs. Supervised Finetuning
- Finetuning: Same contrastive dataset, 1 epoch, SGD, lr=1e-4, supervised prediction objective.
- Finetuning achieves high test accuracy (>90% for most behaviors).
- For **3 out of 7** behaviors, CAA can additionally steer beyond finetuning effects.
- Some counter-intuitive interactions: e.g., for Refusal, positive CAA on top of positive finetuning actually *reduces* refusal.
- **Critical finding:** Finetuning on sycophancy multiple-choice questions *fails to generalize* to open-ended generation, whereas CAA generalizes in ALL cases.
- **Compute advantage:** Generating a CAA vector takes <5 minutes on 1 GPU (forward passes only). Finetuning takes ~10 minutes on 2 GPUs (backward passes required).
- CAA on top of finetuning improves open-ended generation more than multiple-choice performance, suggesting CAA achieves better out-of-distribution generalization by steering existing learned representations.

### CAA vs. Few-Shot Prompting
- Few-shot prompting was found to be less effective than system prompting for the tested behaviors, so it was not a primary baseline.

---

## 8. Interpretability Insights

### Cosine Similarity with Per-Token Activations
- Steering vectors can be used as **behavioral detectors**: the dot product between a token's residual stream activation and the steering vector corresponds intuitively to how much the behavior is "present" at that token.
- Example: "I cannot help" and "I strongly advise against" have positive refusal component; "hack into your friend's Instagram account" has negative refusal component.
- Example: Choosing "a larger cake later" has negative myopia component; "just a small one now" has positive myopia component.

### Inter-Layer Vector Similarity
- Vectors from closer layers are more similar.
- Similarity decline is slower in the latter half of the model (representations "converge" once high-level information is extracted).

---

## 9. Limitations

1. **GPT-4 evaluation:** Scores can be sensitive to prompt wording; LLM evaluators have their own biases. Mitigated by manual inspection and finding consistency with human judgments.
2. **Finetuning baseline not optimized:** Hyperparameters (lr, epochs, loss function) were not extensively tuned. Better finetuning could reduce CAA's marginal improvement.
3. **Prompting baseline not exhaustive:** More effort on prompt engineering might achieve better steering via prompting alone. But CAA's advantage is it doesn't require manual prompt optimization.
4. **Vector normalization:** Vectors are normalized across behaviors but NOT across layers. Since residual stream norms grow exponentially, this could skew layer optimality results. Different multipliers might be optimal at different layers.
5. **Text quality degradation at high multipliers:** Steering at every token position after the prompt caps the perturbation magnitude before quality suffers.
6. **Tested only on Llama 2 family** (7B and 13B Chat, 7B base). Generalization to other model families not demonstrated.

---

## 10. Suggested Future Work (from the paper)

1. **Targeted token positions:** Instead of adding the vector at every generated token, intervene at a smaller subset for better tradeoff between intervention magnitude and effect.
2. **Steering outside the residual stream:** Apply vectors after the MLP or in other positions to learn where representations are localized and achieve more targeted effects.
3. **Red-teaming application:** Use CAA as an adversarial probe -- if a behavior can be easily triggered via small internal perturbations, it may also occur in deployment. Inability to elicit behaviors via CAA could serve as a stronger safety guarantee.

---

## 11. Relevance to Our Project: Finding an "AI-Sounding" Direction

CAA's methodology is **directly applicable** to our research question of whether there exists a "sounds like AI" direction in the residual stream. Key parallels:

1. **Mean Difference approach:** We can compute the mean difference of residual stream activations between AI-written and human-written text to extract a potential "AI-sounding" direction. This is exactly the v_MD formula from the paper.

2. **Contrastive dataset construction:** We need paired examples where the only difference is AI-written vs. human-written. The paper's emphasis on tight pairing (differing by only the answer token) suggests we should make our contrastive examples as similar as possible except for the AI/human distinction.

3. **Layer selection:** The paper finds that ~1/3 of the way through the model is optimal for behavioral representations. For our work, we should focus on similar relative layer positions and use PCA visualization to find where "AI-ness" representations emerge.

4. **PCA validation:** Before computing a steering vector, we should check whether PCA on our contrastive dataset shows behavioral clustering (AI vs. human separability) beyond mere surface-level token clustering. This is a critical quality check.

5. **Cosine similarity as detector:** Once we have an "AI-sounding" direction, we can use cosine similarity with per-token activations to identify which tokens/phrases most strongly activate the "AI-ness" direction -- this directly addresses our research question.

6. **Linear representation hypothesis:** The paper's success across 7 diverse behaviors supports the idea that high-level concepts (including potentially "AI-sounding-ness") are linearly represented in the residual stream.

7. **Steering application:** If we find an "AI-sounding" direction, we could subtract it during generation to make outputs sound less like AI, or add it to study what the model considers "AI-like."

### Key methodological takeaways for our implementation:
- Use hundreds of contrastive pairs, not just a few.
- Extract activations at a specific token position (e.g., the last token).
- Target middle layers (~1/3 through the model).
- Validate with PCA before computing the steering vector.
- Test both on controlled evaluations and open-ended generation.
- Check that the intervention doesn't degrade general capabilities (MMLU or equivalent).
