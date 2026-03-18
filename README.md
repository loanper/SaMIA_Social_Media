# SaMIA Membership Suspicion Scoring (Social Media, Unsupervised)

Project implementing a clean, reproducible pipeline around **SaMIA** (Sampling-based Pseudo-Likelihood for Membership Inference Attacks) to produce **per-message membership suspicion scores** on noisy social-media text.

This project focuses on a realistic setting where **ground-truth membership labels are unavailable** (e.g., Facebook/TikTok comments not tied to the model’s training set). Instead of forcing fake labels, the pipeline stays **scoring-first**: it ranks messages by how “memorization-like” the model’s continuations are.

## What This Repo Does

Given a dataset of texts, for each message it:

1) splits the message into a prefix/suffix (prompt/reference)
2) asks an LLM to generate continuations from the prefix (sampling)
3) compares generated continuations against the reference suffix
4) aggregates the similarity into a single **suspicion score** per message
5) outputs a ranked JSONL + CSV (+ optional metric summary table)

Key improvements over the original research-style code:

- **Membership unknown** data path (no arbitrary `label=0`)
- robust **candidate/reference alignment** via `source_input`
- modular metric registry + multi-metric export
- Windows-friendly JSONL reading (handles UTF-8 BOM)

## Implemented Similarity Metrics

The original SaMIA paper uses surface similarity (ROUGE). This repo supports multiple metrics to compare rankings:

- `rouge1_recall` (SaMIA core)
- `rouge1_recall_zlib` (paper variant)
- `jaccard` (token set overlap)
- `tfidf_cosine` (TF-IDF cosine similarity)
- `sbert_cosine` (Sentence-BERT embeddings cosine)
- `bertscore_f1` (contextual token similarity)

Important: these metrics are on different scales. Use them to compare **rankings** and stability, not as direct “percent membership”.

## Repository Layout

- `src/convert_data.py`: CSV → JSONL conversion for social datasets (membership unknown)
- `src/sampling.py`: generate candidate continuations with LLMs
- `src/eval_samia.py`: compute suspicion scores + export ranking
- `custom_data/`: generated JSONL references (ignored by git)
- `sample/`: generated candidates (ignored by git)
- `results/`: scoring outputs (ignored by git)

## Reproducibility (No Datasets Committed)

Datasets are intentionally excluded from git. To reproduce results:

### 0) Environment

Recommended: conda + Python 3.11.

```powershell
conda create -n samia python=3.11 -y
conda activate samia

# Core
pip install numpy tqdm transformers huggingface-hub

# PyTorch (choose ONE based on your machine)
# CPU-only:
pip install torch
# NVIDIA GPU (example; choose the correct CUDA index-url for your setup):
# pip install torch --index-url https://download.pytorch.org/whl/cu128

# Optional metrics
pip install scikit-learn            # TF-IDF
pip install sentence-transformers   # SBERT embeddings
pip install bert-score              # BERTScore
```

### 1) Provide a CSV dataset

`src/convert_data.py` expects a CSV column containing the text (default: `comment_text`).

Place your raw CSV(s) at:

- `../EDA/Facebook-datasets.csv`
- `../EDA/TikTok-datasets.csv`

Or run conversion with your own file paths by editing the defaults in `src/convert_data.py`.

### 2) Convert CSV → JSONL

```powershell
python src/convert_data.py --dataset facebook --text_length 32
python src/convert_data.py --dataset tiktok   --text_length 32
```

Outputs go to `custom_data/`.

### 3) Generate candidates (sampling)

Example (Qwen2.5-3B):

```powershell
python src/sampling.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 `
      --input_file custom_data/facebook_32.jsonl `
      --output_file sample/Qwen2.5-3B/facebook_32.jsonl
```

### 4) Score and export a ranking

Compute multiple metrics and export a summary table (one row per metric):

```powershell
python src/eval_samia.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 `
      --metrics rouge1_recall jaccard tfidf_cosine sbert_cosine bertscore_f1 `
      --cand_file sample/Qwen2.5-3B/facebook_32.jsonl `
      --ref_file  custom_data/facebook_32.jsonl `
      --output_file results/Qwen2.5-3B_facebook32_multi_scoring.jsonl `
      --output_csv  results/Qwen2.5-3B_facebook32_multi_scoring.csv `
      --summary_csv results/Qwen2.5-3B_facebook32_metrics_summary.csv
```

### 5) Alignment-safe “first N” subset (recommended for debugging)

On Windows:

```powershell
Get-Content custom_data/facebook_32.jsonl -TotalCount 25 | Set-Content custom_data/facebook_32_first25.jsonl -Encoding utf8
python src/sampling.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 --max_new_tokens 16 `
      --input_file custom_data/facebook_32_first25.jsonl `
      --output_file sample/Qwen2.5-3B/facebook_32_first25.jsonl
python src/eval_samia.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 `
      --metrics rouge1_recall jaccard tfidf_cosine sbert_cosine bertscore_f1 `
      --cand_file sample/Qwen2.5-3B/facebook_32_first25.jsonl `
      --ref_file  custom_data/facebook_32_first25.jsonl `
      --output_csv results/Qwen2.5-3B_first25_multi_scoring.csv `
      --summary_csv results/Qwen2.5-3B_first25_metrics_summary.csv
```

## Notes on Labels

For social datasets (Facebook/TikTok) membership is unknown. Supervised ROC/AUC is only meaningful if you have a benchmark where membership labels are real (e.g., a controlled train/test split or a dataset like WikiMIA).

## Attribution

This project is based on the SaMIA research idea described in:

- Kaneko et al., “Sampling-based Pseudo-Likelihood for Membership Inference Attacks” (arXiv:2404.11262)