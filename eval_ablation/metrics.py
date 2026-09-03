"""
Self-contained, dependency-free text metrics for the prune-ablation caption eval.

Only stdlib + (optional) numpy. Everything here operates on plain strings; the
tokenizer is a lowercase alphanumeric split so results are stable across machines
and need no nltk / rouge-score / sacrebleu install.

Exposed:
    caption_metrics(candidate, reference) -> dict   (candidate vs a gold caption)
    pair_metrics(text_a, text_b)          -> dict   (model-A caption vs model-B caption)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _prf(match: int, cand_total: int, ref_total: int) -> Dict[str, float]:
    p = match / cand_total if cand_total else 0.0
    r = match / ref_total if ref_total else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f": f}


def rouge_n(cand: List[str], ref: List[str], n: int) -> Dict[str, float]:
    c, r = _ngrams(cand, n), _ngrams(ref, n)
    match = sum(min(c[g], r[g]) for g in c)
    return _prf(match, max(sum(c.values()), 0), max(sum(r.values()), 0))


def _lcs(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(cand: List[str], ref: List[str]) -> Dict[str, float]:
    return _prf(_lcs(cand, ref), len(cand), len(ref))


def bleu(cand: List[str], ref: List[str], max_n: int = 4) -> float:
    """Sentence BLEU with add-1 smoothing and brevity penalty."""
    if not cand:
        return 0.0
    weights = 1.0 / max_n
    log_acc = 0.0
    for n in range(1, max_n + 1):
        c, r = _ngrams(cand, n), _ngrams(ref, n)
        match = sum(min(c[g], r[g]) for g in c)
        total = max(sum(c.values()), 1)
        prec = (match + 1.0) / (total + 1.0)          # add-1 smoothing
        log_acc += weights * math.log(prec)
    bp = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return bp * math.exp(log_acc)


def distinct_n(tokens: List[str], n: int) -> float:
    g = _ngrams(tokens, n)
    tot = sum(g.values())
    return len(g) / tot if tot else 0.0


def repetition(tokens: List[str]) -> Dict[str, float]:
    """Degeneration signals: low distinct-n and a high single-ngram repeat count
    both flag a compressor whose features let the LLM loop."""
    g4 = _ngrams(tokens, 4)
    max_rep = max(g4.values()) if g4 else 0
    return {
        "distinct_1": distinct_n(tokens, 1),
        "distinct_2": distinct_n(tokens, 2),
        "repeat_4gram_max": float(max_rep),
        # fraction of 4-grams that are duplicates of an earlier one
        "dup_4gram_rate": 1.0 - distinct_n(tokens, 4),
    }


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def caption_metrics(candidate: str, reference: str) -> Dict[str, float]:
    c, r = tokenize(candidate), tokenize(reference)
    out = {
        "cand_words": len(c),
        "ref_words": len(r),
        "length_ratio": (len(c) / len(r)) if r else 0.0,
        "rouge1_f": rouge_n(c, r, 1)["f"],
        "rouge2_f": rouge_n(c, r, 2)["f"],
        "rougeL_f": rouge_l(c, r)["f"],
        "bleu4": bleu(c, r, 4),
        "unigram_recall": rouge_n(c, r, 1)["r"],
    }
    out.update(repetition(c))
    return out


def pair_metrics(text_a: str, text_b: str) -> Dict[str, float]:
    a, b = tokenize(text_a), tokenize(text_b)
    return {
        "rougeL_f": rouge_l(a, b)["f"],
        "rouge1_f": rouge_n(a, b, 1)["f"],
        "jaccard": jaccard(a, b),
        "len_ratio_a_over_b": (len(a) / len(b)) if b else 0.0,
    }


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = dicts[0].keys()
    return {k: float(sum(d[k] for d in dicts) / len(dicts)) for k in keys}
