# SaMIA Scoring Runbook (Social Media, Membership Unknown)

This file is a **practical runbook** for reproducing the end-to-end pipeline in this folder.

Goal: compute **per-message membership suspicion scores** (unsupervised ranking) on social-media text where seen/unseen labels are unknown.

## Pipeline Overview

1) Convert CSV → JSONL references (membership unknown)
2) Generate continuations (sampling)
3) Score candidates vs reference suffix (one score per message)
4) Export ranking (JSONL + CSV) + optional metrics comparison table

## A) Convert CSV → JSONL

Expected input CSVs (not committed):
- `../EDA/Facebook-datasets.csv`
- `../EDA/TikTok-datasets.csv`

Default text column: `comment_text`.

```powershell
python src/convert_data.py --dataset facebook --text_length 32
python src/convert_data.py --dataset tiktok   --text_length 32
```

Outputs:
- `custom_data/facebook_32.jsonl`
- `custom_data/tiktok_32.jsonl`

## B) Generate candidates (sampling)

Example (Qwen2.5-3B, small debug subset):

```powershell
# Create a deterministic first-25 subset for alignment-safe debugging
Get-Content custom_data/facebook_32.jsonl -TotalCount 25 | Set-Content custom_data/facebook_32_first25.jsonl -Encoding utf8

# Generate 1 continuation per message
python src/sampling.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 --max_new_tokens 16 `
  --input_file custom_data/facebook_32_first25.jsonl `
  --output_file sample/Qwen2.5-3B/facebook_32_first25.jsonl
```

Notes:
- `sampling.py` stores both the full `output_i` and the prompt-stripped `continuation_i`.
- If you rerun, it overwrites the output file by default (prevents accidental duplication).

## C) Score + export

Compute multiple metrics and export a summary table:

```powershell
python src/eval_samia.py --model_name Qwen2.5-3B --text_length 32 --num_samples 1 --prefix_ratio 0.5 `
  --metrics rouge1_recall jaccard tfidf_cosine sbert_cosine bertscore_f1 `
  --cand_file sample/Qwen2.5-3B/facebook_32_first25.jsonl `
  --ref_file  custom_data/facebook_32_first25.jsonl `
  --output_file results/Qwen2.5-3B_first25_multi_scoring.jsonl `
  --output_csv  results/Qwen2.5-3B_first25_multi_scoring.csv `
  --summary_csv results/Qwen2.5-3B_first25_metrics_summary.csv `
  --top_k 10
```

Outputs:
- Per-message ranking: `results/*.jsonl` and `results/*.csv`
- Per-metric summary: `results/*_metrics_summary.csv`

## GPU Check

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

If `torch.cuda.is_available()` is `False`, your current Python environment is using CPU-only PyTorch.

## Labels (Why ROC/AUC is Usually Skipped)

For Facebook/TikTok custom data, membership is unknown. ROC/AUC is only meaningful if true binary labels exist in both classes.
