# Paper Notes: Style-Specific Neurons for Steering LLMs in Text Style Transfer

**Citation:** Lai, Hangya, Fraser (2024). "Style-Specific Neurons for Steering LLMs in Text Style Transfer." arXiv:2410.00593v1.
**Authors:** Wen Lai, Viktor Hangya, Alexander Fraser (TU Munich / LMU Munich)
**Code:** https://github.com/wenlai-lavine/sNeuron-TST
**Relevance to our project:** Directly analogous to finding an "AI-sounding" direction in the residual stream -- they identify style-specific neurons/directions and use them to steer generation.

---

## 1. Core Idea

The paper introduces **sNeuron-TST**, a framework for text style transfer (TST) that identifies style-specific neurons in LLM feed-forward networks (FFNs) and manipulates them at inference time to steer generation toward a target style. The key insight is that deactivating source-style neurons increases the probability of generating target-style words, but this hurts fluency -- which they fix with a novel contrastive decoding method.

---

## 2. How Style-Specific Neurons Are Identified

### 2.1 Where they look: FFN neurons
- They focus on neurons in the **feed-forward network (FFN)** modules of the Transformer, which contain ~2/3 of model parameters.
- Activation values of layer j: `a^(j) = act_fn(W^(j) * a^(j-1) + b^(j))`
- A neuron i at layer j is considered **active** when its activation value `a_i^(j) > 0`.

### 2.2 Neuron selection procedure
1. Feed source-style corpus (style A) and target-style corpus (style B) separately through the LLM.
2. Collect activation values of all FFN neurons for both styles.
3. Select neurons with activation > 0, forming sets S_A and S_B.
4. Sort by activation value in descending order; select top-k neurons (k = 500n, n in {1,...,20}, tuned on validation set), yielding S'_A and S'_B.
5. **Critical step -- eliminate overlap:** Compute disjoint sets:
   - N_A = S'_A \ S'_B (neurons active only in style A)
   - N_B = S'_B \ S'_A (neurons active only in style B)

### 2.3 Why overlap elimination matters
- Using the naive method of Tang et al. (2024) for language-specific neurons, they found **massive overlap** between style neurons -- up to ~95% in the Politics benchmark (democratic vs. republican styles).
- This overlap is much higher than for language-specific neurons (~25% Chinese-English overlap).
- Ablation study (Table 3) shows that removing overlap consistently improves style transfer accuracy across all 12 transfer directions (e.g., informal->formal improves from 74.00 to 79.40).
- **Key insight for our project:** Style is encoded in a more distributed, overlapping manner than language. The polysemy problem (one neuron encoding multiple features) is severe for style attributes.

---

## 3. Which Layers Contain Style Information

### 3.1 Style neurons concentrate in later layers
- Figure 3 shows a clear pattern: **the last few layers (especially the final layer) contain significantly more style-specific neurons** than earlier layers.
- This holds for both formality and toxicity benchmarks in LLaMA-3.
- Earlier layers (0-26) show relatively flat, low counts of style neurons.
- The last ~4-5 layers show a dramatic increase.

### 3.2 JSD analysis across layers (Appendix C)
- JSD between the final layer and all previous layers remains nearly constant from layer 0 to ~26.
- From layer 27 to 31 (in 32-layer LLaMA-3), JSD distance from the final layer decreases but the **distance between consecutive layers increases** -- indicating active style processing.
- Target-style words (e.g., formal-style words) show larger JSD distance differences in these last layers, confirming style information is resolved there.

### 3.3 Implications
- They term the last few layers **"style layers"** and use the last 4 layers as candidate layers for contrastive decoding.
- **Key insight for our project:** Style-related computation happens predominantly in the latter layers of the network. If "AI-sounding" is a style attribute, we should look at later layers of the residual stream.

---

## 4. Style Attributes Studied

Six benchmarks, each with two styles (12 transfer directions total):

| Benchmark     | Styles                        | Dataset                          |
|---------------|-------------------------------|----------------------------------|
| Formality     | informal <-> formal           | GYAFC (Rao & Tetreault, 2018)    |
| Toxicity      | toxic <-> neutral             | ParaDetox (Logacheva et al., 2022)|
| Politics      | democratic <-> republican     | Political (Voigt et al., 2018)   |
| Politeness    | impolite <-> polite           | Politeness (Madaan et al., 2020) |
| Authorship    | shakespeare <-> modern        | Shakespeare (Xu et al., 2012)    |
| Sentiment     | positive <-> negative         | Yelp (Shen et al., 2017)        |

---

## 5. How They Steer Model Outputs

### 5.1 Step 1: Deactivate source-style neurons
- Set activation values of source-style-only neurons (N_A) to zero during forward pass.
- Effect: increases probability of target-style words during decoding.
- Problem: dramatically hurts fluency -- model generates concatenations of target-style words without grammatical coherence.
- Example: "Neither dishes were prepared with poor veggies" (all target-style words, poor fluency).

### 5.2 Step 2: Contrastive decoding (adapted from DoLa)
- **DoLa** (Chuang et al., 2024) contrasts outputs between the final layer and early layers to improve factuality.
- **Their adaptation to TST:**
  - Instead of contrasting with early layers, contrast with the **style layers** (last 4 layers).
  - At each time step, check if a target-style token's high probability is due to:
    - (a) Consistently high probability from early to final layer -> style-independent (function words), keep it.
    - (b) Low probability in early layers but "mutation" to high probability in last layers due to neuron deactivation -> use contrastive decoding to adjust.
  - Select the premature layer M from candidate style layers using max Jensen-Shannon Divergence (JSD).
  - Final token probability: `p_hat(x_t) = softmax(F(p^N(x_t), p^M(x_t)))` where F computes log-domain difference.
- Result: "Neither dish was prepared with quality veggies" (target-style structure, but fluent).

### 5.3 The full pipeline (sNeuron-TST)
1. Identify style-specific neurons (offline, using style corpora)
2. Deactivate source-style-only neurons during inference
3. Apply contrastive decoding comparing style layers vs. final layer

---

## 6. Datasets and Evaluation Metrics

### 6.1 Dataset sizes
| Benchmark   | Train | Valid | Test |
|-------------|-------|-------|------|
| Politeness  | 100k  | 2000  | 2000 |
| Toxicity    | 18k   | 2000  | 2000 |
| Formality   | 52k   | 500   | 500  |
| Authorship  | 27k   | 500   | 500  |
| Politics    | 100k  | 1000  | 1000 |
| Sentiment   | 100k  | 1000  | 1000 |

Preprocessing: removed sentences >120 chars, duplicates, special-symbol-heavy sentences.

### 6.2 Evaluation metrics
- **Style Accuracy:** Predicted by open-source style classifiers (one per benchmark, details in Table 8). Higher = better.
- **Content Preservation:** Cosine similarity of LaBSE sentence embeddings between source and generated text. Also BLEURT. Higher = better.
- **Fluency:** Perplexity of generated text using GPT-2. Lower = better.

### 6.3 Style classifiers used
- Politeness: xlm-roberta-large-tydip
- Toxicity: roberta_toxicity_classifier
- Formality: xlmr_formality_classifier
- Authorship: shakespeare_classifier_model
- Politics: distilbert-political-tweets
- Sentiment: distilbert-base-uncased-finetuned-sst-2-english

---

## 7. Key Results

### 7.1 Main findings (Table 2, LLaMA-3 8B)
- sNeuron-TST outperforms all baselines (LLaMA-3 vanilla, APE, AVF, PNMA) on **style accuracy** in 11/12 transfer directions.
- Also achieves best **fluency** (lowest perplexity) in most directions.
- Content preservation is competitive but not always best -- trade-off with style transfer strength.

### 7.2 Ablation study results
- **Overlap removal** (Table 3): Consistent improvement across all 12 directions when overlap between source and target neurons is eliminated.
- **Neuron deactivation + contrastive decoding** (Table 4):
  - Deactivation alone improves accuracy (toxic->neutral: 47.67 -> 52.63).
  - Contrastive decoding alone does NOT help (may even hurt: 47.67 -> 46.82).
  - Both together achieve best results (55.36).
  - CD only works when neurons are already deactivated (it needs the probability "mutation" in style layers to be effective).

### 7.3 Copy problem
- LLaMA-3 copies 34%+ of input text verbatim in some directions (e.g., polite->impolite).
- sNeuron-TST significantly reduces copy ratio across all tested directions.

### 7.4 Directional asymmetry
- Transfer from "negative" styles (impolite, toxic, informal) to "positive" styles is much easier (~80% accuracy) than the reverse (~12-30%).
- Attributed to LLM training data being predominantly "positive" (polite, neutral, formal).
- LLMs have a safety bias toward generating "safer" responses.

### 7.5 Scalability
- Method also works on LLaMA-3 70B (Table 10), with consistent improvements.
- Style layers approach outperforms original DoLa early-layer approach on all 12 directions (Table 6).

---

## 8. Baselines Compared

- **LLaMA-3 (vanilla):** Zero-shot with prompts, no neuron manipulation.
- **APE (Tang et al., 2024):** Activation probability entropy for neuron identification. Originally for language-specific neurons.
- **AVF (Tan et al., 2024):** Activation value frequency with threshold. Originally for multilingual MT.
- **PNMA (Kojima et al., 2024):** Neurons that activate on source but not target. Requires parallel data.

All baselines adapted from language-specific neuron methods; they underperform on style because style overlap is much higher than language overlap.

---

## 9. Limitations (stated by authors)

1. **Layer-uniform deactivation:** They deactivate style neurons across all layers uniformly. Different layers may have different roles (understanding vs. generation) -- deactivating selectively per layer could yield better results.
2. **Task scope:** Only evaluated on TST; could potentially apply to image style transfer, multilingual style transfer, domain adaptation, etc.
3. **Content preservation trade-off:** Their method is not optimal for content preservation metrics -- the semantic gap between different styles makes this inherently difficult to measure.

---

## 10. Relevance to Our "Sounds Like AI" Project

### Direct analogies
| Their concept | Our analog |
|---------------|------------|
| Style-specific neurons in FFN | "AI-sounding" direction in residual stream |
| Source style (e.g., formal) | AI-generated text style |
| Target style (e.g., informal) | Human-written text style |
| Neuron deactivation to suppress source style | Ablating/steering along "AI" direction |
| Contrastive decoding across style layers | Potential contrastive probing across layers |
| Overlap between source/target neurons | Shared features between AI and human text |

### Key takeaways for our work

1. **Style information is concentrated in later layers.** We should focus our probing on the latter portion of the residual stream (last ~4-5 layers for a 32-layer model, proportionally ~last 12-15%).

2. **Overlap is a critical problem.** For style attributes (as opposed to language), there is massive neuron overlap (~95%). Any method for finding "AI-style" neurons/directions must explicitly handle the overlap between AI-style and human-style features. Simply identifying "AI-active" neurons without removing those also active for human text would be ineffective.

3. **Individual neuron analysis may be insufficient.** Their approach works at the individual FFN neuron level. Our residual stream direction approach (using linear probes, PCA, or contrast vectors) may be more effective because it captures distributed representations rather than individual neuron activations.

4. **Deactivation hurts fluency.** Naively suppressing style features damages generation quality. Any steering approach needs a mechanism to preserve fluency/coherence -- their contrastive decoding is one solution; we might consider similar approaches.

5. **Directional asymmetry exists.** Transfer difficulty depends on direction. LLMs already bias toward "polite/formal/neutral" -- analogously, they likely bias toward certain stylistic patterns. The "AI-sounding" direction may be the default mode, making it easier to detect than to remove.

6. **Contrastive decoding with style layers outperforms early-layer contrasting.** The standard DoLa approach (contrasting early vs. final layers) does not work well for style. Contrasting within the style-processing layers (the last few) is more effective.

7. **Neuron-level analysis can transfer across model scales.** Their method works on both 8B and 70B LLaMA-3, suggesting that style encoding patterns are consistent across scales.
