import os
import re
from collections import Counter
from typing import List, Tuple, Dict

import requests
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

STANFORD_BASE_URL = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"
IWSLT_FILES = [
    "train.en", "train.vi",
    "tst2012.en", "tst2012.vi",
    "tst2013.en", "tst2013.vi",
]


def basic_tokenize(text: str) -> List[str]:
    """Tách từ đơn giản theo khoảng trắng và tách dấu câu phổ biến."""
    text = text.strip()
    text = re.sub(r"([.,!?;:\"()\[\]])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().split()


class Vocab:
    """Bộ từ điển ánh xạ giữa token và index."""

    def __init__(self, min_freq: int = 2, max_size: int = 30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos: List[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}

    def build(self, tokenized_sentences: List[List[str]]) -> None:
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)

        # Sắp xếp giảm dần theo tần suất
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for tok, freq in sorted_tokens:
            if freq < self.min_freq:
                continue
            if len(self.itos) >= self.max_size:
                break
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, tokens: List[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(tok, UNK_IDX) for tok in tokens]
        if add_bos_eos:
            return [BOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
        specials = {PAD_IDX, BOS_IDX, EOS_IDX}
        tokens = []
        for i in ids:
            if i == EOS_IDX and skip_special:
                break
            if skip_special and i in specials:
                continue
            tokens.append(self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN)
        return tokens

    def __len__(self) -> int:
        return len(self.itos)


def _write_hf_split(split_data, src_path: str, tgt_path: str) -> None:
    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for item in split_data:
            if "translation" in item:
                en_text = item["translation"]["en"]
                vi_text = item["translation"]["vi"]
            else:
                en_text = item["en"]
                vi_text = item["vi"]
            f_src.write(en_text.replace("\n", " ").strip() + "\n")
            f_tgt.write(vi_text.replace("\n", " ").strip() + "\n")


def download_iwslt15(data_dir: str = "data/iwslt15") -> str:
    """Tải bộ dữ liệu IWSLT'15 En-Vi (ưu tiên Hugging Face, fallback Stanford NMT)."""
    os.makedirs(data_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(data_dir, f)) for f in IWSLT_FILES):
        return data_dir

    # 1. Thử tải từ Hugging Face datasets (nhanh và ổn định trên Colab)
    try:
        from datasets import load_dataset
        print("Đang tải IWSLT'15 En-Vi từ Hugging Face (thainq107/iwslt2015-en-vi) ...")
        try:
            ds = load_dataset("thainq107/iwslt2015-en-vi")
        except Exception:
            print("Thử nguồn dự phòng IWSLT/mt_eng_vietnamese ...")
            ds = load_dataset("IWSLT/mt_eng_vietnamese", "iwslt2015-en-vi")

        split_map = {
            "train": ("train.en", "train.vi"),
            "validation": ("tst2012.en", "tst2012.vi"),
            "test": ("tst2013.en", "tst2013.vi"),
        }
        for split_name, (en_file, vi_file) in split_map.items():
            _write_hf_split(
                ds[split_name],
                os.path.join(data_dir, en_file),
                os.path.join(data_dir, vi_file),
            )
        return data_dir
    except Exception as e:
        print(f"Không tải được qua Hugging Face ({e}), chuyển sang Stanford NMT ...")

    # 2. Fallback: tải trực tiếp từ Stanford NMT
    for fname in IWSLT_FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            url = f"{STANFORD_BASE_URL}/{fname}"
            print(f"Đang tải {url} ...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(fpath, "wb") as f:
                f.write(resp.content)
    return data_dir


def read_parallel_file(
    src_path: str,
    tgt_path: str,
    max_len: int = 60,
    max_samples: int = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Đọc cặp file song ngữ, lọc câu rỗng hoặc quá dài."""
    src_sents, tgt_sents = [], []
    with open(src_path, "r", encoding="utf-8") as f_src, \
         open(tgt_path, "r", encoding="utf-8") as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            src_toks = basic_tokenize(src_line)
            tgt_toks = basic_tokenize(tgt_line)
            if not src_toks or not tgt_toks:
                continue
            if len(src_toks) > max_len or len(tgt_toks) > max_len:
                continue
            src_sents.append(src_toks)
            tgt_sents.append(tgt_toks)
            if max_samples is not None and len(src_sents) >= max_samples:
                break
    return src_sents, tgt_sents


def get_dummy_data() -> Tuple[
    Tuple[List[List[str]], List[List[str]]],
    Tuple[List[List[str]], List[List[str]]],
    Tuple[List[List[str]], List[List[str]]],
]:
    """Dữ liệu giả lập nhỏ để test nhanh pipeline không cần tải mạng."""
    pairs = [
        ("hello , how are you ?", "xin chào , bạn khỏe không ?"),
        ("i am fine , thank you .", "tôi khỏe , cảm ơn bạn ."),
        ("what is your name ?", "tên của bạn là gì ?"),
        ("my name is minh .", "tên tôi là minh ."),
        ("i love deep learning .", "tôi yêu học sâu ."),
        ("machine translation is very interesting .", "dịch máy rất thú vị ."),
        ("good morning everyone .", "chào buổi sáng mọi người ."),
        ("see you tomorrow .", "hẹn gặp lại bạn ngày mai ."),
        ("the weather is nice today .", "hôm nay thời tiết đẹp ."),
        ("she goes to school every day .", "cô ấy đi học mỗi ngày ."),
    ]
    train_pairs = pairs * 20
    val_pairs = pairs[:4]
    test_pairs = pairs[4:8]

    def split_pairs(p_list):
        return (
            [basic_tokenize(en) for en, _ in p_list],
            [basic_tokenize(vi) for _, vi in p_list],
        )

    return split_pairs(train_pairs), split_pairs(val_pairs), split_pairs(test_pairs)


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_sents: List[List[str]],
        tgt_sents: List[List[str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
    ):
        self.src_ids = [torch.tensor(src_vocab.encode(s), dtype=torch.long) for s in src_sents]
        self.tgt_ids = [torch.tensor(tgt_vocab.encode(t), dtype=torch.long) for t in tgt_sents]

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.src_ids[idx], self.tgt_ids[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


def prepare_dataloaders(
    data_dir: str = "data/iwslt15",
    batch_size: int = 64,
    max_len: int = 60,
    min_freq: int = 2,
    max_train_samples: int = None,
    use_dummy: bool = False,
):
    """Tải dữ liệu, xây dựng vocab và trả về DataLoader cho train/val/test."""
    if use_dummy:
        (tr_src, tr_tgt), (va_src, va_tgt), (te_src, te_tgt) = get_dummy_data()
        min_freq = 1
    else:
        download_iwslt15(data_dir)
        tr_src, tr_tgt = read_parallel_file(
            os.path.join(data_dir, "train.en"),
            os.path.join(data_dir, "train.vi"),
            max_len=max_len,
            max_samples=max_train_samples,
        )
        va_src, va_tgt = read_parallel_file(
            os.path.join(data_dir, "tst2012.en"),
            os.path.join(data_dir, "tst2012.vi"),
            max_len=max_len,
        )
        te_src, te_tgt = read_parallel_file(
            os.path.join(data_dir, "tst2013.en"),
            os.path.join(data_dir, "tst2013.vi"),
            max_len=max_len,
        )

    src_vocab = Vocab(min_freq=min_freq)
    tgt_vocab = Vocab(min_freq=min_freq)
    src_vocab.build(tr_src)
    tgt_vocab.build(tr_tgt)

    train_ds = TranslationDataset(tr_src, tr_tgt, src_vocab, tgt_vocab)
    val_ds = TranslationDataset(va_src, va_tgt, src_vocab, tgt_vocab)
    test_ds = TranslationDataset(te_src, te_tgt, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
