# Style Vectors for Steering Generative Large Language Models

**Paper:** Konen et al. (2024), arXiv:2402.01618v1
**Authors:** Kai Konen, Sophie Jentzsch, Diaoule Diallo, Peer Schutt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, Tobias Hecking
**Affiliation:** Institute for Software Technology, German Aerospace Center (DLR)
**Code:** https://github.com/DLR-SC/style-vectors-for-steering-llms

---

## Relevance to Our Research

This paper is **directly relevant** to "Is there a 'sounds like AI' direction in the residual stream?" because it demonstrates that **style can be represented as a linear direction (vector) in activation space**, and that adding this vector to hidden layer activations during generation steers the model's output style. If sentiment, emotion, and writing style each have a direction, it is plausible that "AI-sounding" text also has a direction. The methodology here -- computing mean activation differences between style categories -- is exactly the kind of approach we could use to find an "AI style" direction.

---

## Core Methodology

### The Fundamental Equation

Style vectors v_s^(i) for style category s at layer i are added to activations during the forward pass:

```
a_hat^(i)(x) = a^(i)(x) + lambda * v_s^(i)     (Eq. 3)
```

Where:
- `a^(i)(x)` = original activation at layer i for input x
- `v_s^(i)` = style vector for style s at layer i
- `lambda` = weighting parameter controlling steering strength
- `a_hat^(i)(x)` = modified activation

The style vectors are added to **layers 18, 19, and 20** simultaneously during each forward pass (for Alpaca-7B / LLaMA-7B with 33 layers).

### Two Approaches Compared

#### 1. Training-based Style Vectors (Section 3.1)

Based on Subramani et al. (2022). For each target sentence x, a steering vector z_x is **learned via optimization** (up to 400 epochs, Adam optimizer, lr=0.01) such that adding z_x to a layer's activations causes the model to generate x from an empty BOS token. The objective maximizes log probability:

```
z_hat_steer = argmax_{z_steer} sum_{t=1}^{T} log p(x_t | x_{<t}, z_steer)
```

**Key adaptation by the authors:** Instead of requiring source-to-target style transfer (which needs n*(n-1) contrasting vectors for n classes), they:
1. Mean-aggregate all steering vectors for style s into z_bar_s^(i)
2. Mean-aggregate all *other* steering vectors into z_bar_{S\s}^(i)
3. Compute the style vector as the **difference of means**:

```
v_s^(i) = z_bar_s^(i) - z_bar_{S\s}^(i)     (Eq. 4)
```

This eliminates the need to know the source style.

**Drawback:** Extremely expensive. Only vectors with training loss < 5 were kept (to avoid grammatically broken outputs). Training took 150 hours per dataset (Yelp, Shakespeare) and 100 hours (GoEmotions) on a single NVIDIA A100-SXM4-80GB. Only samples < 50 characters could be used. Resulted in only 470 vectors/layer (Yelp), 89 (GoEmotions), 491 (Shakespeare).

#### 2. Activation-based Style Vectors (Section 3.2) -- THE PREFERRED METHOD

Based on Turner et al. (2023). Simply **record the hidden layer activations** when the model processes input sentences from each style category, then compute:

```
v_s^(i) = a_bar_s^(i) - a_bar_{S\s}^(i)     (Eq. 5)
```

Where:
- `a_bar_s^(i)` = mean activation at layer i across all input sentences from style s
- `a_bar_{S\s}^(i)` = mean activation at layer i across all input sentences NOT from style s

**Key insight:** This assumes LLMs internally adapt to the style of the input during the forward pass. The style information is already encoded in the hidden states.

**Advantages over training-based:**
- No optimization needed -- just a single forward pass per sentence
- Takes at most 8 hours per dataset (vs. 100-150 hours)
- No sample length restriction
- Works for ALL dataset samples
- Better or comparable performance

---

## Model and Architecture Details

- **Model:** Alpaca-7B (Stanford Alpaca, Taori et al. 2023), based on LLaMA-7B (Touvron et al. 2023)
- **Architecture:** 33 layers
- **Style vector dimension:** d = 4096 (matching hidden state dimension)
- **Hardware:** Single NVIDIA A100-SXM4-80GB
- **Critical layers:** i in {18, 19, 20} -- roughly layers 55-61% through the network

---

## Which Layers Are Most Effective

This is one of the most important findings for our research:

### Probing Study Results (Section 4.3)

A probing classifier was trained on activations/steering vectors at each layer to predict style class. Results:

**Yelp Sentiment (2-class: positive/negative):**
- Layer 0: AUC = 0.60 (near random)
- Layer 1: AUC = 0.95
- Layer 3: AUC = 0.98
- Layers 5-30: AUC = 0.97-0.98
- **Layers 18, 19, 20: AUC = 0.98-0.99** (activation vectors)
- Trained steering vectors at layers 18-20: AUC = 0.90-0.91

**Shakespeare (2-class: modern/Shakespearean):**
- Layer 0: AUC = 0.49 (random)
- Layer 1: AUC = 0.89
- Layer 3-30: AUC = 0.94-0.95
- Activation vectors layers 18-20: AUC = 0.94-0.96
- Trained steering vectors layers 18-20: AUC = 0.69-0.80

**GoEmotions (6-class emotions):**
- Activation vectors of 2k sentences, layers 18-20: micro-average AUC = 0.90-0.91
- Trained steering vectors layers 18-20: micro-average AUC = 0.81-0.82
- Per-emotion AUC (all activations, layer 19): joy=0.96, surprise=0.92, anger=0.91, sadness=0.90, fear=0.88-0.89, disgust=0.83-0.84

### Key Finding on Layer Selection

- Style information emerges strongly from layer 1 onward (dramatic jump from layer 0)
- Layers 18-20 (roughly 55-61% through the 33-layer model) were selected as most effective for steering
- The probing accuracy is very high from layer 3 onwards, but the **steering effectiveness** peaks at layers 18-20
- Single-layer vs. multi-layer classifiers showed minimal difference, suggesting style information is well-represented in individual layers
- Activations encode style more explicitly than trained steering vectors

---

## Styles Tested

### 1. Sentiment (Binary)
- **Positive** and **Negative** sentiment
- Source: Yelp Review Dataset

### 2. Emotion (6-class)
- **Sadness, Joy, Fear, Anger, Surprise, Disgust** (Ekman's basic emotions)
- Source: GoEmotions dataset

### 3. Writing Style
- **Modern** vs. **Shakespearean** English
- Source: Shakespeare dataset
- This is notably a whole writing style change, not just emotional valence

---

## Datasets Used

| Dataset | Size | Classes | Notes |
|---------|------|---------|-------|
| **Yelp Review** (Shen et al., 2017) | 542k samples (after dedup) | Positive, Negative | Restaurant reviews |
| **GoEmotions** (Demszky et al., 2020) | 5k samples used (from 58k total) | 6 basic emotions | Reddit comments; mapped from 27 fine-grained categories to Ekman's 6 |
| **Shakespeare** (Jhamtani et al., 2017) | 18,395 sentence pairs | Modern, Shakespearean | Paired translations |

For training-based vectors, only samples with < 50 characters were used. This limitation does not apply to activation-based vectors.

---

## Evaluation Methodology

### Evaluation Prompts
- **99 total prompts** designed for the study
  - **50 factual prompts** (e.g., "Who painted the Mona Lisa?", "What is the capital city of France?")
  - **49 subjective prompts** (e.g., "How do you define happiness?", "Share a personal anecdote about a vacation you had")

### Classification-based Evaluation

**For sentiment (Yelp):**
- VADER sentiment analyzer (Hutto and Gilbert, 2014) used to score output positivity/negativity

**For emotion (GoEmotions):**
- RoBERTa-based multi-class emotion classifier (Hartmann, 2022) -- `emotion-english-distilroberta-base`

**Prompting baseline:**
- Appending "Write the answer in a [positive/negative/angry/etc.] manner." to the input prompt
- This serves as the comparison against style vector steering

### Lambda (Steering Strength) Parameter
- Lambda = 0: no steering (original model output)
- Lambda values tested: 0 to ~2.0 (varies by experiment)
- Optimal range varies by style and prompt type
- Too large lambda produces nonsense/repetition (e.g., "sadly sadly sadly..." or "great great great...")

---

## Key Results

### Activation-based vs. Training-based
- **Activation-based style vectors are uniformly preferred** -- better performance, orders of magnitude cheaper
- Training-based vectors produce unstable outputs at lower lambda values, especially for GoEmotions (lambda >= 0.75 already breaks output)
- Activation-based vectors allow smooth, continuous modulation

### Sentiment Steering (Yelp)
- For **subjective prompts**: clear monotonic increase in target sentiment score as lambda increases
- Positive steering works somewhat better than negative steering
- Activation-based vectors show **stronger** steering effect than training-based
- **Factual prompts**: almost no change in sentiment -- model maintains neutral, factual responses (the authors view this positively as a safety property)
- Prompt baseline shows only **minimal effect** compared to style vector steering

### Emotion Steering (GoEmotions)
- Activation-based vectors (using all samples) successfully steer toward all 6 emotions on subjective prompts
- Steering to **anger, sadness, joy, and fear** works best
- Steering to **disgust and surprise** is weaker / harder
- Factual prompts again show almost no steering effect
- GoEmotions prompt baselines had a stronger effect than Yelp prompt baselines

### Writing Style (Shakespeare)
- Modern steering: effective at lower lambda (lambda = 0.8) because the model's default style is already closer to modern English
- Shakespearean steering: requires higher lambda (lambda = 1.6) to produce flowery, antiquated language
- Example: "Happiness is a state of contentment and joy, wherein the soul is freed from the bondage of sorrow..." (Shakespearean-steered)

### Style Vectors vs. Prompt Engineering
- Style vectors provide **smoother, more continuous** control than prompting
- Prompting offers only coarse step-wise control ("positive" vs. "very positive")
- Style vector steering outperforms the neutral prompt baseline for sentiment
- GoEmotions prompt baselines are more competitive

---

## Important Observations for Our Research

### 1. The Mean-Difference Method Works
Computing a style vector as `mean(target_class_activations) - mean(other_class_activations)` is simple yet highly effective. This is exactly the approach we should try for "AI-sounding" vs. "human" text.

### 2. Style is Linear in Activation Space
The fact that simply ADDING a vector to activations shifts style confirms that style is approximately **linearly represented** in the residual stream. This supports the hypothesis that "AI-sounding" could also be a linear direction.

### 3. Middle-to-Late Layers are Key
Layers 18-20 (out of 33, roughly 55-61%) are most effective for style representation. This suggests we should focus on similar proportional depths when looking for an "AI style" direction.

### 4. Lambda Controls Intensity Continuously
The ability to scale the style vector by lambda and get proportional style changes suggests these directions are genuine linear features, not binary on/off switches.

### 5. Domain Bias in Style Vectors
Yelp-derived style vectors carry topical bias toward food/restaurants, not just sentiment. Any "AI style" vector computed from specific datasets will likely carry domain-specific biases. This is critical to control for.

### 6. Factual vs. Subjective Robustness
Factual prompts resist style steering. This suggests the model's commitment to factual accuracy can override style directions. For our work: if "AI sounding" is primarily a style property, it should be more malleable on open-ended/subjective outputs.

---

## Limitations

1. **Training-based approach is computationally prohibitive** -- 100-150 hours on A100 per dataset, limited to < 50 character samples
2. **Only one model tested** -- Alpaca-7B (LLaMA-7B based). No verification on other architectures, though authors claim transferability
3. **Evaluation relies on automated classifiers** (VADER, RoBERTa) -- no human evaluation conducted. "Results only show a general tendency."
4. **Lambda tuning is not solved** -- no principled way to select lambda; it depends on prompt, domain, and target style. Too-high lambda causes degenerate outputs
5. **Strong focus on sentiment/emotion** -- writing style (Shakespeare) is only briefly explored. More complex styles (formality, register, genre) not tested
6. **English only** -- no experiments with other languages
7. **Domain bias problem** -- style vectors encode both target style AND domain-specific content (e.g., Yelp vectors bias toward food topics)
8. **Only tested on 99 prompts** -- relatively small evaluation set
9. **GoEmotions had very few usable training-based vectors** (only 89), making the comparison less reliable for that dataset
10. **No investigation of how style vectors interact** -- multidimensional composed styles acknowledged as a challenge but not addressed
11. **Ethics concern:** Style vectors could be used maliciously (targeted harassment, fake reviews, impersonation)

---

## Connections to Related Work

- **Subramani et al. (2022):** Original steering vectors paper (trained optimization approach). This paper extends it by mean-aggregating into class-level vectors.
- **Turner et al. (2023):** Activation Addition ("ActAdd") -- showed contrasting activations for opposed inputs (e.g., "love" vs. "hate") can steer LLMs. This paper extends it from single word-pairs to general style categories computed from datasets.
- **Hernandez et al. (2023):** Measuring and manipulating knowledge representations -- related activation manipulation approach.
- **Brack et al. (2022):** Analogous approach in image generation (latent space shifting during diffusion).

---

## Summary for Literature Review

Konen et al. (2024) demonstrate that style -- including sentiment, emotion, and writing style -- can be represented as a **linear direction in the residual stream** of transformer language models. Their key contribution is showing that **activation-based style vectors**, computed simply as the difference between mean activations for a target style class and mean activations for all other classes, can effectively steer LLM output when added to layers 18-20 of a 33-layer model (Alpaca-7B/LLaMA-7B). This approach requires no training or optimization, only recording activations during forward passes of style-labeled text. The method provides continuous, parameterizable control over output style via a scalar lambda coefficient, outperforming both a training-based alternative and simple prompt engineering baselines. The work provides strong evidence that abstract stylistic properties are linearly encoded in activation space, directly supporting the plausibility of our hypothesis that an "AI-sounding" direction exists in the residual stream.
