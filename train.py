import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import PAD_IDX, prepare_dataloaders
from src.models import Seq2SeqAttention, TransformerNMT
from src.utils import compute_bleu, plot_attention


def build_model(model_name: str, src_vocab_size: int, tgt_vocab_size: int, device: torch.device):
    if model_name == "seq2seq":
        model = Seq2SeqAttention(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            emb_dim=256,
            enc_hid_dim=256,
            dec_hid_dim=256,
            dropout=0.3,
        )
    elif model_name == "transformer":
        model = TransformerNMT(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=256,
            n_heads=4,
            num_layers=3,
            d_ff=512,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")
    return model.to(device)


def train_epoch(model, loader, optimizer, criterion, device, clip: float = 1.0):
    model.train()
    total_loss = 0.0
    for src, tgt in tqdm(loader, desc="Train", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        logits = model(src, tgt)  # [B, tgt_len - 1, vocab_size]
        targets = tgt[:, 1:]      # bỏ <bos>
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        if isinstance(model, Seq2SeqAttention):
            logits = model(src, tgt, teacher_forcing_ratio=0.0)
        else:
            logits = model(src, tgt)
        targets = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_bleu(model, loader, src_vocab, tgt_vocab, device, max_eval_sents: int = 200):
    model.eval()
    hyps, refs = [], []
    count = 0
    for src, tgt in loader:
        for i in range(src.size(0)):
            src_i = src[i : i + 1].to(device)
            pred_ids, _ = model.translate(src_i)
            pred_str = " ".join(tgt_vocab.decode(pred_ids))
            ref_str = " ".join(tgt_vocab.decode(tgt[i].tolist()))
            hyps.append(pred_str)
            refs.append(ref_str)
            count += 1
            if max_eval_sents is not None and count >= max_eval_sents:
                return compute_bleu(hyps, refs)
    return compute_bleu(hyps, refs)


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Demo Dịch máy Anh-Việt")
    parser.add_argument("--model", type=str, default="seq2seq", choices=["seq2seq", "transformer"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Giới hạn số câu train nếu muốn chạy nhanh")
    parser.add_argument("--max_eval_sents", type=int, default=300, help="Số câu tối đa để tính BLEU mỗi epoch")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--dummy", action="store_true", help="Dùng dữ liệu giả lập nhỏ để test nhanh pipeline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device} | Mô hình: {args.model}")

    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = prepare_dataloaders(
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        use_dummy=args.dummy,
    )
    print(f"Vocab Anh: {len(src_vocab)} | Vocab Việt: {len(tgt_vocab)}")
    print(f"Số batch - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    model = build_model(args.model, len(src_vocab), len(tgt_vocab), device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Số tham số huấn luyện: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pt")

    best_bleu = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss = evaluate_loss(model, val_loader, criterion, device)
        va_bleu = evaluate_bleu(model, val_loader, src_vocab, tgt_vocab, device, args.max_eval_sents)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} ({dt:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | Val BLEU: {va_bleu:.2f}"
        )

        if va_bleu >= best_bleu:
            best_bleu = va_bleu
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                },
                ckpt_path,
            )

    # Đánh giá trên tập test với checkpoint tốt nhất
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_bleu = evaluate_bleu(model, test_loader, src_vocab, tgt_vocab, device, max_eval_sents=None)
    print(f"\n>>> Test BLEU ({args.model}): {test_bleu:.2f}")
    print(f">>> Đã lưu checkpoint tốt nhất tại: {ckpt_path}")

    # Sinh thử 1 ví dụ và lưu Attention Map
    sample_src, sample_tgt = next(iter(test_loader))
    src_one = sample_src[0:1].to(device)
    pred_ids, attn_mat = model.translate(src_one)

    src_tokens = src_vocab.decode(sample_src[0].tolist(), skip_special=False)
    # Cắt bỏ phần <pad> ở cuối câu nguồn để vẽ đẹp hơn
    src_tokens = [t for t in src_tokens if t != "<pad>"]
    pred_tokens = tgt_vocab.decode(pred_ids)
    ref_tokens = tgt_vocab.decode(sample_tgt[0].tolist())

    print("Ví dụ Test:")
    print("  Anh :", " ".join(src_tokens))
    print("  Đích:", " ".join(ref_tokens))
    print("  Dịch:", " ".join(pred_tokens))

    attn_path = os.path.join(args.save_dir, f"{args.model}_attention.png")
    plot_attention(src_tokens, pred_tokens, attn_mat, attn_path, title=f"{args.model.upper()} Attention")
    print(f">>> Đã lưu Attention Map tại: {attn_path}")


if __name__ == "__main__":
    main()
