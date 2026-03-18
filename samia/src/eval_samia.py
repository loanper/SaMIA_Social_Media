import argparse
import csv
import json
import os
import re
from typing import Dict, List

import numpy as np

from metrics import build_metric
from utils import load_jsonl


def split_prefix_suffix(text, prefix_ratio):
    words = text.split()
    if not words:
        return "", []
    cut = max(1, min(len(words) - 1, int(round(len(words) * prefix_ratio))))
    prefix = " ".join(words[:cut])
    suffix = words[cut:]
    return prefix, suffix


def clean_text(text, model_name):
    cleaned = text

    # GPT family
    cleaned = re.sub(r"<\|endoftext\|>", " ", cleaned)

    # Llama-2 and OPT
    cleaned = re.sub(r"<s>", " ", cleaned)
    cleaned = re.sub(r"</s>", " ", cleaned)

    # Llama-3
    cleaned = re.sub(r"<\|begin_of_text\|>", " ", cleaned)
    cleaned = re.sub(r"<\|end_of_text\|>", " ", cleaned)
    cleaned = re.sub(r"<\|eot_id\|>", " ", cleaned)

    # Qwen chat/control tokens
    cleaned = re.sub(r"<\|im_start\|>", " ", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", " ", cleaned)

    # Some tokenizers may emit additional generic special tokens.
    cleaned = re.sub(r"<\|.*?\|>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    _ = model_name
    return cleaned


def extract_continuation(candidate_line, output_index, prefix, model_name):
    continuation_key = f"continuation_{output_index}"
    output_key = f"output_{output_index}"

    if continuation_key in candidate_line:
        return clean_text(candidate_line[continuation_key], model_name)

    raw = clean_text(candidate_line.get(output_key, ""), model_name)
    if raw.startswith(prefix):
        return raw[len(prefix):].strip()
    return raw


def get_available_output_indices(candidate_line):
    indices = []
    for key in candidate_line:
        if key.startswith("output_"):
            suffix = key.split("output_", 1)[1]
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)


def compute_summary(scores):
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "q10": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def maybe_supervised_metrics(records):
    labeled = [r for r in records if r.get("label") in (0, 1)]
    classes = sorted({r.get("label") for r in labeled})
    if len(classes) < 2:
        return None

    y_true = np.array([r["label"] for r in labeled], dtype=int)
    y_score = np.array([r["mean_score"] for r in labeled], dtype=float)

    thresholds = np.unique(y_score)[::-1]
    tpr_values = []
    fpr_values = []

    positives = max(1, np.sum(y_true == 1))
    negatives = max(1, np.sum(y_true == 0))

    for threshold in thresholds:
        predicted = (y_score >= threshold).astype(int)
        tp = np.sum((predicted == 1) & (y_true == 1))
        fp = np.sum((predicted == 1) & (y_true == 0))
        tpr_values.append(tp / positives)
        fpr_values.append(fp / negatives)

    fpr = np.array([0.0] + fpr_values + [1.0], dtype=float)
    tpr = np.array([0.0] + tpr_values + [1.0], dtype=float)
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]

    roc_auc = float(np.trapz(tpr, fpr))
    idx = int(np.argmin(np.abs(fpr - 0.10)))
    return {
        "roc_auc": roc_auc,
        "tpr_at_10_fpr": float(tpr[idx]),
        "labeled_count": len(labeled),
    }


def write_jsonl(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Keep a stable baseline schema, then append any metric-specific columns.
    base = [
        "rank",
        "mean_score",
        "zlib_mean_score",
        "num_candidates_used",
        "input_text",
        "prefix",
        "reference_suffix",
        "membership",
        "label",
    ]

    extra = []
    if records:
        keys = set()
        for rec in records:
            keys.update(rec.keys())
        # Include additional per-metric mean columns in a predictable order.
        for key in sorted(keys):
            if key.startswith("mean_score_") and key not in base:
                extra.append(key)

    fieldnames = base + extra
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name) for name in fieldnames})


def write_summary_csv(summary_rows: List[dict], output_path: str):
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not summary_rows:
        return
    # Write all keys present across rows.
    keys = set()
    for row in summary_rows:
        keys.update(row.keys())
    fieldnames = [
        "metric",
        "count",
        "mean",
        "median",
        "std",
        "q10",
        "q25",
        "q75",
        "q90",
        "q95",
        "q99",
        "min",
        "max",
        "model_name",
        "text_length",
        "num_samples",
        "prefix_ratio",
        "cand_file",
        "ref_file",
    ]
    # Append any other keys deterministically.
    for key in sorted(keys):
        if key not in fieldnames:
            fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def build_default_output_path(model_name, text_length, metric_name):
    return f"results/{model_name}_{text_length}_{metric_name}_scoring.jsonl"


def _index_candidates(lines_cand):
    """Index candidates for robust alignment.

    Supports both newer sampling output (with 'source_input') and older formats.
    If duplicates exist (e.g., from accidental appends), the last occurrence wins.
    """
    by_source = {}
    by_prefix = {}
    for cand in lines_cand:
        source = cand.get("source_input")
        prefix = cand.get("input")
        if isinstance(source, str) and source.strip():
            by_source[source] = cand
        if isinstance(prefix, str) and prefix.strip():
            by_prefix[prefix] = cand
    return by_source, by_prefix


def _index_references(lines_ref):
    """Index references for one-to-one alignment.

    Returns a dict input_text -> list[ref_line] preserving original order.
    This supports duplicate texts in the reference file.
    """
    by_text = {}
    for ref in lines_ref:
        text = ref.get("input")
        if isinstance(text, str) and text.strip():
            by_text.setdefault(text, []).append(ref)
    return by_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SaMIA scoring mode: per-message unsupervised membership suspicion scores.",
    )

    parser.add_argument(
        "--model_name",
        default="gpt2",
        choices=[
            "gpt-j-6B",
            "opt-6.7b",
            "pythia-6.9b",
            "Llama-2-7b",
            "Llama-3-8b",
            "Qwen2.5-7B",
            "Qwen2.5-3B",
            "gpt2",
        ],
        type=str,
    )
    parser.add_argument("--text_length", default=32, choices=[32, 64, 128, 256], type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)
    parser.add_argument(
        "--zlib",
        action="store_true",
        help="Legacy flag: compute ROUGE-1 and ROUGE-1*zlib and rank by the zlib variant.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Metrics to compute (space-separated). Examples: --metrics rouge1_recall jaccard tfidf_cosine sbert_cosine bertscore_f1. "
            "If omitted, defaults to ROUGE-1 (and zlib variant when --zlib is set)."
        ),
    )
    parser.add_argument(
        "--metric_device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for embedding-based metrics (SBERT/BERTScore).",
    )
    parser.add_argument(
        "--sbert_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model name for sbert_cosine.",
    )
    parser.add_argument(
        "--tfidf_max_features",
        type=int,
        default=50000,
        help="Max features for TF-IDF (tfidf_cosine).",
    )
    parser.add_argument(
        "--bertscore_model",
        default="roberta-base",
        help="Model type for BERTScore (e.g., roberta-base, distilroberta-base).",
    )
    parser.add_argument(
        "--bertscore_lang",
        default="en",
        help="Language shortcut for BERTScore baseline selection (e.g., en, fr).",
    )
    parser.add_argument(
        "--bertscore_rescale",
        action="store_true",
        help="Enable BERTScore rescale_with_baseline (can help calibration across domains).",
    )
    parser.add_argument("--top_k", default=10, type=int)
    parser.add_argument("--cand_file", type=str, default=None, help="Path to candidate JSONL file")
    parser.add_argument("--ref_file", type=str, default=None, help="Path to reference JSONL file")
    parser.add_argument("--output_file", type=str, default=None, help="Per-message score output JSONL")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional CSV export")
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Optional summary CSV (one row per metric) for easy comparison.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    text_length = args.text_length
    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio

    # Metrics selection
    explicit_metrics = args.metrics is not None and len(args.metrics) > 0
    if explicit_metrics:
        metric_inputs = args.metrics
        rank_metric_name = metric_inputs[0]
        legacy_zlib_mode = False
    else:
        # Backward-compatible defaults
        if args.zlib:
            metric_inputs = ["rouge1_recall", "rouge1_recall_zlib"]
            rank_metric_name = "rouge1_recall_zlib"
            legacy_zlib_mode = True
        else:
            metric_inputs = ["rouge1_recall"]
            rank_metric_name = "rouge1_recall"
            legacy_zlib_mode = False

    cand_path = args.cand_file or f"sample/{model_name}/{text_length}.jsonl"
    ref_path = args.ref_file or f"wikimia/{text_length}.jsonl"
    # Keep output path naming stable for legacy modes.
    primary_metric_for_name = "rouge1_recall_zlib" if (args.zlib and not explicit_metrics) else metric_inputs[0]
    output_path = args.output_file or build_default_output_path(model_name, text_length, primary_metric_for_name)

    print(f"Reading candidates from {cand_path}")
    print(f"Reading references from {ref_path}")
    print(f"Scoring metrics: {metric_inputs}")
    if args.zlib and explicit_metrics:
        print("Note: --zlib is ignored because --metrics was provided.")

    # Build metric computers (dedupe by canonical metric.name)
    metric_objects = []
    seen_metric_names = set()
    for name in metric_inputs:
        metric = build_metric(
            name,
            device=args.metric_device,
            sbert_model=args.sbert_model,
            tfidf_max_features=args.tfidf_max_features,
            bertscore_model=args.bertscore_model,
            bertscore_lang=args.bertscore_lang,
            bertscore_rescale=args.bertscore_rescale,
        )
        if metric.name not in seen_metric_names:
            metric_objects.append(metric)
            seen_metric_names.add(metric.name)

    # Resolve ranking metric after aliases
    rank_metric = build_metric(
        rank_metric_name,
        device=args.metric_device,
        sbert_model=args.sbert_model,
        tfidf_max_features=args.tfidf_max_features,
        bertscore_model=args.bertscore_model,
        bertscore_lang=args.bertscore_lang,
        bertscore_rescale=args.bertscore_rescale,
    )

    lines_cand = load_jsonl(cand_path)
    lines_ref = load_jsonl(ref_path)

    if not lines_cand or not lines_ref:
        raise ValueError("Candidate or reference file is empty.")

    # Preferred: one-to-one alignment by iterating candidates.
    # This avoids over-counting when references contain duplicate texts.
    aligned_pairs = []
    cand_has_source = any(isinstance(c.get("source_input"), str) and c.get("source_input").strip() for c in lines_cand)
    if cand_has_source:
        ref_by_text = _index_references(lines_ref)
        missing = 0
        for cand in lines_cand:
            source = cand.get("source_input")
            if not isinstance(source, str) or not source.strip():
                missing += 1
                continue
            bucket = ref_by_text.get(source)
            if not bucket:
                missing += 1
                continue
            ref = bucket.pop(0)
            aligned_pairs.append((cand, ref))

        if missing or len(aligned_pairs) != len(lines_cand):
            print(
                "Note: candidate/reference alignment used 1-to-1 matching by source_input. "
                f"Aligned {len(aligned_pairs)} of {len(lines_cand)} candidates; {missing} candidates could not be matched."
            )
    else:
        by_source, by_prefix = _index_candidates(lines_cand)

        # Try robust matching first: ref.input -> candidate by source_input (exact) or by computed prefix.
        for ref in lines_ref:
            input_text = ref.get("input", "")
            if not isinstance(input_text, str) or not input_text.strip():
                continue
            prefix, _ = split_prefix_suffix(input_text, prefix_ratio)

            cand = by_source.get(input_text)
            if cand is None:
                cand = by_prefix.get(prefix)
            if cand is None:
                continue
            aligned_pairs.append((cand, ref))

    # Fallback to positional alignment if matching failed entirely.
    if not aligned_pairs:
        pair_count = min(len(lines_cand), len(lines_ref))
        if len(lines_cand) != len(lines_ref):
            print(
                f"Warning: candidate/reference size mismatch ({len(lines_cand)} vs {len(lines_ref)}). "
                f"Using first {pair_count} aligned pairs."
            )
        aligned_pairs = [(lines_cand[i], lines_ref[i]) for i in range(pair_count)]
    else:
        if not cand_has_source and (len(aligned_pairs) != len(lines_ref) or len(lines_cand) != len(lines_ref)):
            print(
                "Note: candidate/reference alignment used matching (by source_input/prefix). "
                f"Matched {len(aligned_pairs)} of {len(lines_ref)} references; candidates file has {len(lines_cand)} lines."
            )

    # Phase 1: collect per-record candidate continuations.
    items = []
    for idx, (line_cand, line_ref) in enumerate(aligned_pairs):
        input_text = line_ref.get("input", "")

        prefix, reference_suffix_tokens = split_prefix_suffix(input_text, prefix_ratio)
        reference_suffix = " ".join(reference_suffix_tokens)

        available = get_available_output_indices(line_cand)
        selected = available[:num_samples]
        if not selected:
            continue

        candidate_outputs = []
        for output_index in selected:
            continuation = extract_continuation(line_cand, output_index, prefix, model_name)
            candidate_outputs.append(continuation)

        membership_value = line_ref.get("membership")
        if membership_value is None:
            membership_value = "known" if line_ref.get("label") in (0, 1) else "unknown"

        items.append(
            {
                "index": idx,
                "input_text": input_text,
                "prefix": prefix,
                "reference_suffix": reference_suffix,
                "candidate_outputs": candidate_outputs,
                "num_candidates_used": len(candidate_outputs),
                "membership": membership_value,
                "label": line_ref.get("label"),
            }
        )

    if not items:
        raise ValueError("No valid scored records were produced (no candidates found).")

    # Phase 2: batch-score all (candidate, reference_suffix) pairs for each metric.
    cand_flat: List[str] = []
    ref_flat: List[str] = []
    slices: List[tuple] = []
    for item in items:
        start = len(cand_flat)
        for cand in item["candidate_outputs"]:
            cand_flat.append(cand)
            ref_flat.append(item["reference_suffix"])
        end = len(cand_flat)
        slices.append((start, end))

    metric_scores: Dict[str, np.ndarray] = {}
    for metric in metric_objects:
        metric_scores[metric.name] = metric.score_pairs(cand_flat, ref_flat)

    # Phase 3: build per-record output structure.
    results = []
    for item, (start, end) in zip(items, slices):
        record = dict(item)

        for metric in metric_objects:
            arr = metric_scores[metric.name][start:end]
            scores_list = [float(x) for x in arr.tolist()]
            record[f"candidate_scores_{metric.name}"] = scores_list
            record[f"mean_score_{metric.name}"] = float(np.mean(arr)) if arr.size else 0.0

        # Backward compatible fields
        if legacy_zlib_mode:
            record["candidate_scores"] = record.get("candidate_scores_rouge1_recall", [])
            record["mean_score"] = record.get("mean_score_rouge1_recall", 0.0)
            record["zlib_candidate_scores"] = record.get("candidate_scores_rouge1_recall_zlib", [])
            record["zlib_mean_score"] = record.get("mean_score_rouge1_recall_zlib", 0.0)
            record["metric"] = "rouge1_recall"
        else:
            primary = metric_objects[0].name
            record["candidate_scores"] = record.get(f"candidate_scores_{primary}", [])
            record["mean_score"] = record.get(f"mean_score_{primary}", 0.0)
            record["metric"] = primary

        results.append(record)

    ranking_key = "zlib_mean_score" if legacy_zlib_mode else "mean_score"
    ranked = sorted(results, key=lambda x: x.get(ranking_key, 0.0), reverse=True)
    for rank, rec in enumerate(ranked, start=1):
        rec["rank"] = rank

    all_scores = [rec[ranking_key] for rec in ranked]
    summary = compute_summary(all_scores)

    # Optional metric comparison summary (one row per metric)
    summary_rows: List[dict] = []
    for metric in metric_objects:
        key = f"mean_score_{metric.name}"
        metric_arr = [rec.get(key, 0.0) for rec in ranked]
        m_summary = compute_summary(metric_arr)
        summary_rows.append(
            {
                "metric": metric.name,
                **m_summary,
                "model_name": model_name,
                "text_length": text_length,
                "num_samples": num_samples,
                "prefix_ratio": prefix_ratio,
                "cand_file": cand_path,
                "ref_file": ref_path,
            }
        )

    write_jsonl(ranked, output_path)
    if args.output_csv:
        write_csv(ranked, args.output_csv)
    if args.summary_csv:
        write_summary_csv(summary_rows, args.summary_csv)

    print("\nUnsupervised membership suspicion summary")
    print(f"Scored records : {summary['count']}")
    print(f"Mean           : {summary['mean']:.6f}")
    print(f"Median         : {summary['median']:.6f}")
    print(f"Std            : {summary['std']:.6f}")
    print(
        f"Quantiles      : q10={summary['q10']:.6f}, q25={summary['q25']:.6f}, q75={summary['q75']:.6f}, "
        f"q90={summary['q90']:.6f}, q95={summary['q95']:.6f}, q99={summary['q99']:.6f}"
    )
    print(f"Min/Max        : min={summary['min']:.6f}, max={summary['max']:.6f}")
    print(f"Saved JSONL    : {output_path}")
    if args.output_csv:
        print(f"Saved CSV      : {args.output_csv}")
    if args.summary_csv:
        print(f"Saved summary  : {args.summary_csv}")

    print(f"\nTop-{min(args.top_k, len(ranked))} most suspicious messages")
    for rec in ranked[: args.top_k]:
        preview = rec["input_text"][:120].replace("\n", " ")
        print(f"rank={rec['rank']:>3} score={rec[ranking_key]:.6f} text={preview}")

    supervised = maybe_supervised_metrics(ranked)
    if supervised:
        print("\nSupervised metrics (available labels only)")
        print(f"ROC-AUC        : {supervised['roc_auc']:.6f}")
        print(f"TPR@10%FPR     : {supervised['tpr_at_10_fpr'] * 100:.2f}%")
        print(f"Labeled count  : {supervised['labeled_count']}")
    else:
        print("\nNo valid seen/unseen labels found. ROC/AUC intentionally skipped.")
