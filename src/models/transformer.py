import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import PAD_IDX, BOS_IDX, EOS_IDX


class PositionalEncoding(nn.Module):
    """Mã hóa vị trí dạng Sin/Cos theo bài báo 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, d_model]
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Tự cài đặt Scaled Dot-Product Multi-Head Attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads."
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Chiếu tuyến tính và tách thành n_heads: [B, n_heads, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, n_heads, q_len, k_len]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_weights)

        context = torch.matmul(attn_probs, V)  # [B, n_heads, q_len, d_k]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.fc_out(context), attn_weights


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_out))

        cross_out, cross_attn = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, cross_attn


class TransformerNMT(nn.Module):
    """Transformer nhỏ tự cài đặt cho dịch máy Anh-Việt."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_len: int = 200,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        # [B, 1, 1, src_len]: 1 ở vị trí hợp lệ, 0 ở PAD
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        # Kết hợp Padding Mask và Causal (Look-ahead) Mask
        batch_size, tgt_len = tgt.shape
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, tgt_len]
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=torch.bool))
        return pad_mask & causal_mask.unsqueeze(0).unsqueeze(1)  # [B, 1, tgt_len, tgt_len]

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        cross_attn = None
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc_out(x), cross_attn

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # Khi train: đầu vào decoder là tgt[:, :-1] (bỏ <eos> cuối)
        tgt_in = tgt[:, :-1]
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt_in)

        enc_out = self.encode(src, src_mask)
        logits, _ = self.decode(tgt_in, enc_out, tgt_mask, src_mask)
        return logits  # [B, tgt_len - 1, tgt_vocab_size]

    @torch.no_grad()
    def translate(self, src: torch.Tensor, max_len: int = 60) -> Tuple[List[int], torch.Tensor]:
        """Dịch greedy 1 câu và trả về danh sách token ID + Cross-Attention map (trung bình các head ở lớp cuối)."""
        self.eval()
        assert src.shape[0] == 1, "translate() hỗ trợ từng câu một (batch_size=1)."
        src_mask = self.make_src_mask(src)
        enc_out = self.encode(src, src_mask)

        tgt_ids = [BOS_IDX]
        last_cross_attn = None

        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=src.device)
            tgt_mask = self.make_tgt_mask(tgt_tensor)
            logits, cross_attn = self.decode(tgt_tensor, enc_out, tgt_mask, src_mask)

            next_tok = int(logits[0, -1].argmax(dim=-1).item())
            last_cross_attn = cross_attn
            if next_tok == EOS_IDX:
                break
            tgt_ids.append(next_tok)

        pred_ids = tgt_ids[1:]  # bỏ <bos>
        if last_cross_attn is not None and len(pred_ids) > 0:
            # Trung bình qua các head: [tgt_len, src_len]
            attn_matrix = last_cross_attn[0].mean(dim=0)[: len(pred_ids)].cpu()
        else:
            attn_matrix = torch.empty(0, src.shape[1])
        return pred_ids, attn_matrix
