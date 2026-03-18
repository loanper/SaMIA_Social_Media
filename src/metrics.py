import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return list(zip(*[tokens[i:] for i in range(n)]))


def _rouge_n_recall(candidate_tokens: Sequence[str], reference_tokens: Sequence[str], n: int = 1) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0
    cand_ngrams = _ngrams(candidate_tokens, n)
    ref_ngrams = _ngrams(reference_tokens, n)
    if not cand_ngrams or not ref_ngrams:
        return 0.0
    ref_count = Counter(ref_ngrams)
    cand_count = Counter(cand_ngrams)
    overlap = ref_count & cand_count
    return float(sum(overlap.values()) / max(1, len(ref_ngrams)))


def _tokenize_simple(text: str) -> List[str]:
    # Simple, robust word tokenizer (letters/digits/underscore) lowercased.
    return re.findall(r"\b\w+\b", (text or "").lower())


def _cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b shape: (N, D)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(1e-12, a_norm * b_norm)
    return np.sum(a * b, axis=1) / denom


def _device_from_preference(device_pref: str) -> str:
    device_pref = (device_pref or "auto").lower()
    if device_pref in ("cpu", "cuda"):
        return device_pref
    # auto
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Metric:
    name: str

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class Rouge1Recall(Metric):
    name: str = "rouge1_recall"

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        scores: List[float] = []
        for c, r in zip(candidates, references):
            c_toks = (c or "").split()
            r_toks = (r or "").split()
            scores.append(_rouge_n_recall(c_toks, r_toks, n=1))
        return np.asarray(scores, dtype=float)


@dataclass
class Rouge1RecallZlib(Metric):
    name: str = "rouge1_recall_zlib"

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        scores: List[float] = []
        for c, r in zip(candidates, references):
            c_toks = (c or "").split()
            r_toks = (r or "").split()
            base = _rouge_n_recall(c_toks, r_toks, n=1)
            compressed = zlib.compress(" ".join(c_toks).encode("utf-8"))
            scores.append(float(base * len(compressed)))
        return np.asarray(scores, dtype=float)


@dataclass
class JaccardSimilarity(Metric):
    name: str = "jaccard"

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        scores: List[float] = []
        for c, r in zip(candidates, references):
            a = set(_tokenize_simple(c))
            b = set(_tokenize_simple(r))
            if not a or not b:
                scores.append(0.0)
                continue
            inter = len(a & b)
            union = len(a | b)
            scores.append(float(inter / max(1, union)))
        return np.asarray(scores, dtype=float)


@dataclass
class TfidfCosineSimilarity(Metric):
    name: str = "tfidf_cosine"
    max_features: Optional[int] = 50000
    ngram_range: Tuple[int, int] = (1, 2)

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as e:
            raise ImportError(
                "TF-IDF metric requires scikit-learn. Install with: pip install scikit-learn"
            ) from e

        # Fit on combined corpus so both sides share the same vocabulary.
        corpus = list(references) + list(candidates)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        x = vectorizer.fit_transform(corpus)
        x_ref = x[: len(references)]
        x_cand = x[len(references) :]

        # Row-wise cosine similarity for sparse matrices.
        # cosine = (a·b) / (||a|| ||b||)
        numer = x_ref.multiply(x_cand).sum(axis=1).A1
        ref_norm = np.sqrt(x_ref.multiply(x_ref).sum(axis=1)).A1
        cand_norm = np.sqrt(x_cand.multiply(x_cand).sum(axis=1)).A1
        denom = np.maximum(1e-12, ref_norm * cand_norm)
        return numer / denom


@dataclass
class SbertCosineSimilarity(Metric):
    name: str = "sbert_cosine"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SBERT embedding metric requires sentence-transformers. Install with: pip install sentence-transformers"
            ) from e

        device = _device_from_preference(self.device)
        model = SentenceTransformer(self.model_name, device=device)

        # encode returns (N, D)
        ref_emb = model.encode(list(references), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=False)
        cand_emb = model.encode(list(candidates), batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=False)
        return _cosine_rowwise(cand_emb.astype(np.float32), ref_emb.astype(np.float32))


@dataclass
class BertScoreF1(Metric):
    name: str = "bertscore_f1"
    model_type: str = "roberta-base"
    lang: str = "en"
    device: str = "auto"
    batch_size: int = 16
    rescale_with_baseline: bool = False

    def score_pairs(self, candidates: Sequence[str], references: Sequence[str]) -> np.ndarray:
        try:
            from bert_score import score as bert_score
        except ImportError as e:
            raise ImportError(
                "BERTScore metric requires bert-score. Install with: pip install bert-score"
            ) from e

        device = _device_from_preference(self.device)
        # bert_score returns torch tensors
        p, r, f1 = bert_score(
            cands=list(candidates),
            refs=list(references),
            lang=self.lang,
            model_type=self.model_type,
            device=device,
            batch_size=self.batch_size,
            rescale_with_baseline=self.rescale_with_baseline,
            verbose=False,
        )
        # Convert to numpy float
        try:
            f1_np = f1.detach().cpu().numpy()
        except Exception:
            f1_np = np.asarray(f1)
        return f1_np.astype(float)


def build_metric(name: str, *, device: str = "auto", sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 tfidf_max_features: Optional[int] = 50000, bertscore_model: str = "roberta-base",
                 bertscore_lang: str = "en", bertscore_rescale: bool = False) -> Metric:
    key = (name or "").strip().lower()
    registry: Dict[str, Metric] = {
        "rouge1_recall": Rouge1Recall(),
        "rouge1": Rouge1Recall(),
        "rouge1_recall_zlib": Rouge1RecallZlib(),
        "rouge1_zlib": Rouge1RecallZlib(),
        "jaccard": JaccardSimilarity(),
        "jaccard_similarity": JaccardSimilarity(),
        "tfidf_cosine": TfidfCosineSimilarity(max_features=tfidf_max_features),
        "tfidf": TfidfCosineSimilarity(max_features=tfidf_max_features),
        "sbert_cosine": SbertCosineSimilarity(model_name=sbert_model, device=device),
        "embedding_cosine": SbertCosineSimilarity(model_name=sbert_model, device=device),
        "bertscore_f1": BertScoreF1(model_type=bertscore_model, lang=bertscore_lang, device=device, rescale_with_baseline=bertscore_rescale),
        "bertscore": BertScoreF1(model_type=bertscore_model, lang=bertscore_lang, device=device, rescale_with_baseline=bertscore_rescale),
    }
    if key not in registry:
        raise ValueError(
            f"Unsupported metric: {name}. Supported: {sorted(set(registry.keys()))}"
        )
    return registry[key]
