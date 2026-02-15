# Is There a "Sounds Like AI" Direction in the Residual Stream?

## Overview

This project investigates whether the quality of "sounding like AI" — the formal, hedging, comprehensive style characteristic of LLM outputs — is encoded as a linear direction in the residual stream of transformer language models. We use contrastive activation analysis (CAA) on paired human/ChatGPT text to extract and validate such a direction.

## Key Findings

- **A direction separating AI from human text achieves 97.5% test accuracy** (AUC 0.999) at layer 21 of Qwen 2.5 3B (58% of model depth)
- **The direction is 93% cosine-similar to a text length direction** — ChatGPT's verbosity is the dominant distinguishing feature
- **After removing the length component, 85.5% accuracy remains** — genuine style differences (formality, hedging, structure) are linearly encoded beyond length
- **Steering with this direction shifts generation style** — subtracting it produces simpler, more casual text; adding it produces more formal, comprehensive text
- **"AI-sounding" is not as unitary as truth or refusal** — it's a composite of correlated features with length being dominant

## Repository Structure

```
├── REPORT.md                    # Full research report with all results
├── README.md                    # This file
├── planning.md                  # Research plan
├── literature_review.md         # Literature review (pre-gathered)
├── resources.md                 # Available resources catalog
├── src/
│   ├── prepare_data.py          # Data loading and preprocessing
│   ├── extract_activations.py   # Residual stream activation extraction
│   ├── find_direction.py        # Direction extraction and evaluation
│   ├── pca_analysis.py          # PCA visualization and silhouette analysis
│   ├── confound_analysis.py     # Length confound analysis
│   ├── length_controlled_analysis.py  # Length-orthogonal direction analysis
│   ├── steering_experiment.py   # Causal steering experiments
│   ├── llm_judge.py             # GPT-4.1 AI-likeness scoring
│   └── create_summary_plots.py  # Summary visualizations
├── results/
│   ├── direction_results.json   # Layer-wise classification results
│   ├── confound_results.json    # Confound analysis results
│   ├── length_controlled_results.json
│   ├── steering_results.json    # Generated text from steering
│   ├── scored_steering_results.json  # LLM judge scores
│   ├── best_direction.pt        # Best AI direction vector
│   ├── all_directions.pt        # Directions at all layers
│   └── plots/
│       ├── summary_figure.png       # 4-panel summary
│       ├── pca_grid.png             # PCA at multiple layers
│       ├── accuracy_silhouette_by_layer.png
│       ├── length_controlled_analysis.png
│       ├── confound_analysis.png
│       └── cross_layer_similarity.png
├── datasets/                    # HC3 dataset (not in git)
├── papers/                      # Reference papers (PDFs)
└── code/                        # Reference implementations
```

## How to Reproduce

### Setup
```bash
uv venv && source .venv/bin/activate
uv pip install torch transformers datasets accelerate numpy scikit-learn matplotlib seaborn tqdm
```

### Run Pipeline
```bash
# 1. Prepare data
python src/prepare_data.py

# 2. Extract activations (~30 seconds per split)
python src/extract_activations.py

# 3. Find direction and evaluate
python src/find_direction.py

# 4. PCA analysis
python src/pca_analysis.py

# 5. Confound analysis
python src/confound_analysis.py
python src/length_controlled_analysis.py

# 6. Steering experiment
python src/steering_experiment.py

# 7. LLM judge (requires OPENAI_API_KEY)
python src/llm_judge.py

# 8. Summary plots
python src/create_summary_plots.py
```

### Requirements
- Python 3.10+
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 3090)
- ~6GB disk space for model weights
- OpenAI API key for LLM judge (optional)

## See Also

- [Full Report](REPORT.md) for detailed methodology, results, and analysis
- [Literature Review](literature_review.md) for background on linear representations and style vectors
- [Planning Document](planning.md) for experimental design rationale
