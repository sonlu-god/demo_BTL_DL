import math
from collections import Counter
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


def _corpus_bleu_fallback(hypotheses: List[str], references: List[str], max_n: int = 4) -> float:
    """Tính corpus BLEU thuần Python khi chưa cài sacrebleu."""
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    hyp_len, ref_len = 0, 0

    for hyp, ref in zip(hypotheses, references):
        h_toks = hyp.split()
        r_toks = ref.split()
        hyp_len += len(h_toks)
        ref_len += len(r_toks)
        for n in range(1, max_n + 1):
            h_ngrams = Counter(tuple(h_toks[i : i + n]) for i in range(len(h_toks) - n + 1))
            r_ngrams = Counter(tuple(r_toks[i : i + n]) for i in range(len(r_toks) - n + 1))
            total_counts[n - 1] += sum(h_ngrams.values())
            for ng, cnt in h_ngrams.items():
                clipped_counts[n - 1] += min(cnt, r_ngrams[ng])

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            precisions.append(1e-9)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])
    log_avg = sum(math.log(p) for p in precisions) / max_n
    return 100.0 * bp * math.exp(log_avg)


def compute_bleu(hypotheses: List[str], references: List[str]) -> float:
    """Tính corpus BLEU bằng thư viện chuẩn sacrebleu (có fallback thuần Python)."""
    if not hypotheses or not references:
        return 0.0
    if sacrebleu is not None:
        score = sacrebleu.corpus_bleu(hypotheses, [references])
        return float(score.score)
    return _corpus_bleu_fallback(hypotheses, references)


def plot_attention(
    src_tokens: List[str],
    tgt_tokens: List[str],
    attn_matrix: torch.Tensor,
    save_path: str,
    title: str = "Attention Map",
) -> None:
    """Vẽ và lưu hình Attention Map giữa câu nguồn (Anh) và câu đích (Việt)."""
    if attn_matrix.numel() == 0 or not tgt_tokens or not src_tokens:
        return

    mat = attn_matrix[: len(tgt_tokens), : len(src_tokens)].numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.55), max(4, len(tgt_tokens) * 0.45)))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tgt_tokens, fontsize=9)
    ax.set_xlabel("Source (English)")
    ax.set_ylabel("Target (Vietnamese)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
