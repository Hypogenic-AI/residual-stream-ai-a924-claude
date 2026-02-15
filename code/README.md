# Code Repositories

Cloned repositories relevant to the "sounds like AI" direction research.

## Repositories

### steering-vectors
- **Path:** `steering-vectors/`
- **Source:** [github.com/steering-vectors/steering-vectors](https://github.com/steering-vectors/steering-vectors)
- **License:** MIT
- **Description:** Pip-installable Python library for training and applying steering vectors to any HuggingFace transformer model.
- **Key API:**
  ```python
  from steering_vectors import train_steering_vector
  sv = train_steering_vector(model, tokenizer, [("positive", "negative"), ...])
  sv.apply(model, multiplier=1.5)
  ```
- **Aggregators:** mean (default), PCA, logistic regression

### CAA (Contrastive Activation Addition)
- **Path:** `CAA/`
- **Source:** [github.com/nrimsky/CAA](https://github.com/nrimsky/CAA)
- **License:** MIT
- **Description:** Nina Rimsky's implementation for Llama 2. Includes `generate_vectors.py` for steering vector extraction and 7 behavioral evaluation datasets.
- **Key pattern:** `vec = (all_pos_layer - all_neg_layer).mean(dim=0)`

### activation-steering
- **Path:** `activation-steering/`
- **Source:** IBM Research
- **License:** Apache 2.0
- **Description:** General-purpose activation steering library. ICLR 2025.

### awesome-representation-engineering
- **Path:** `awesome-representation-engineering/`
- **Description:** Curated list of papers, code, and resources for representation engineering.
