import argparse
import os

import torch

from src.data import basic_tokenize
from src.models import Seq2SeqAttention, TransformerNMT
from src.utils import plot_attention


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    src_vocab = ckpt["src_vocab"]
    tgt_vocab = ckpt["tgt_vocab"]

    if model_name == "seq2seq":
        model = Seq2SeqAttention(len(src_vocab), len(tgt_vocab))
    elif model_name == "transformer":
        model = TransformerNMT(len(src_vocab), len(tgt_vocab))
    else:
        raise ValueError(f"Mô hình không hợp lệ trong checkpoint: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name, src_vocab, tgt_vocab


def translate_sentence(model, model_name, src_vocab, tgt_vocab, text: str, device: torch.device, out_dir: str):
    tokens = basic_tokenize(text)
    src_ids = src_vocab.encode(tokens, add_bos_eos=True)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    pred_ids, attn_mat = model.translate(src_tensor)
    pred_tokens = tgt_vocab.decode(pred_ids)
    src_display = ["<bos>"] + tokens + ["<eos>"]

    os.makedirs(out_dir, exist_ok=True)
    attn_path = os.path.join(out_dir, f"demo_{model_name}_attention.png")
    plot_attention(src_display, pred_tokens, attn_mat, attn_path, title=f"Demo {model_name.upper()}")
    return " ".join(pred_tokens), attn_path


def main():
    parser = argparse.ArgumentParser(description="Demo dịch câu Anh -> Việt và xuất Attention Map")
    parser.add_argument("--ckpt", type=str, required=True, help="Đường dẫn tới file checkpoint (.pt)")
    parser.add_argument("--text", type=str, default=None, help="Câu tiếng Anh cần dịch (bỏ trống để nhập tương tác)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, src_vocab, tgt_vocab = load_checkpoint(args.ckpt, device)
    print(f"Đã tải mô hình {model_name} trên {device}.")

    if args.text:
        vi_text, attn_path = translate_sentence(model, model_name, src_vocab, tgt_vocab, args.text, device, args.out_dir)
        print(f"Input (EN): {args.text}")
        print(f"Dịch  (VI): {vi_text}")
        print(f"Attention : {attn_path}")
        return

    print("Nhập câu tiếng Anh để dịch (gõ 'q' để thoát):")
    while True:
        text = input("\nEN > ").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break
        if not text:
            continue
        vi_text, attn_path = translate_sentence(model, model_name, src_vocab, tgt_vocab, text, device, args.out_dir)
        print(f"VI > {vi_text}")
        print(f"(Đã lưu Attention Map tại {attn_path})")


if __name__ == "__main__":
    main()
