# Datasets

Datasets for investigating whether there's a "sounds like AI" direction in the residual stream.

## Recommended Datasets

### Primary (Best for Contrastive Pairs)

1. **HC3** (`Hello-SimpleAI/HC3`)
   - ~37k QA pairs where both human and ChatGPT answer the same question
   - License: CC-BY-SA-4.0
   - Best for: Direct contrastive pair construction

2. **Human/AI Generated Text** (`dmitva/human_ai_generated_text`)
   - ~1M rows with columns: `instructions`, `human_text`, `ai_text`
   - License: CC-BY-4.0
   - Best for: Large-scale direction extraction with paired data

### Large-Scale Corpora

3. **AI Text Detection Pile** (`artem9k/ai-text-detection-pile`) - 1.39M rows, MIT
4. **AI Text Pile Cleaned** (`srikanthgali/ai-text-detection-pile-cleaned`) - 722k rows, MIT
5. **RAID** (`liamdugan/raid`) - 10M+ docs from 12 LLMs, MIT

### Additional

6. **AI-Human Text** (`andythetechnerd03/AI-human-text`) - NYT + 6 LLMs
7. **M4** (`github.com/mbzuai-nlp/M4`) - Multi-generator, multi-domain, multi-lingual

## Download

```bash
pip install datasets
python download_datasets.py --list   # See all options
python download_datasets.py          # Download HC3 + dmitva_paired (recommended)
python download_datasets.py --all    # Download everything
```

## Note

Large data files are excluded from git via `.gitignore`. Run the download script to fetch datasets locally.
