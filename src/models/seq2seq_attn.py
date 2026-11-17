import random
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import PAD_IDX, BOS_IDX, EOS_IDX


class Encoder(nn.Module):
    """BiLSTM Encoder cho Seq2seq."""

    def __init__(self, vocab_size: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # src: [B, src_len]
        embedded = self.dropout(self.embedding(src))  # [B, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs: [B, src_len, enc_hid_dim * 2]
        # hidden, cell: [2, B, enc_hid_dim] -> ghép chiều thuận & nghịch
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)))  # [B, dec_hid_dim]
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)))        # [B, dec_hid_dim]
        return outputs, (hidden.unsqueeze(0), cell.unsqueeze(0))


class BahdanauAttention(nn.Module):
    """Additive Attention (Bahdanau et al., 2015)."""

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(
        self,
        dec_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # dec_hidden: [B, dec_hid_dim]
        # encoder_outputs: [B, src_len, enc_hid_dim * 2]
        # src_mask: [B, src_len] (True ở vị trí hợp lệ, False ở PAD)
        src_len = encoder_outputs.shape[1]
        dec_hidden_exp = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [B, src_len, dec_hid_dim]

        energy = torch.tanh(self.attn(torch.cat((dec_hidden_exp, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)  # [B, src_len]

        # Che vị trí PAD trước softmax
        scores = scores.masked_fill(~src_mask, float("-inf"))
        return F.softmax(scores, dim=1)  # [B, src_len]


class Decoder(nn.Module):
    """LSTM Decoder kết hợp Bahdanau Attention."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.attention = BahdanauAttention(enc_hid_dim, dec_hid_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.rnn = nn.LSTM(enc_hid_dim * 2 + emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(enc_hid_dim * 2 + dec_hid_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_tok: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_tok: [B]
        embedded = self.dropout(self.embedding(input_tok.unsqueeze(1)))  # [B, 1, emb_dim]

        # Tính attention weights dùng hidden state lớp trên cùng
        attn_weights = self.attention(hidden[-1], encoder_outputs, src_mask)  # [B, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)       # [B, 1, enc_hid_dim * 2]

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        pred_input = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        logits = self.fc_out(pred_input)  # [B, vocab_size]
        return logits, hidden, cell, attn_weights


class Seq2SeqAttention(nn.Module):
    """Mô hình Seq2seq LSTM + Bahdanau Attention hoàn chỉnh."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_dim: int = 256,
        enc_hid_dim: int = 256,
        dec_hid_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, emb_dim, enc_hid_dim, dec_hid_dim, dropout)
        self.tgt_vocab_size = tgt_vocab_size

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        # src: [B, src_len], tgt: [B, tgt_len] (gồm <bos> ... <eos>)
        batch_size, tgt_len = tgt.shape
        outputs = torch.zeros(batch_size, tgt_len - 1, self.tgt_vocab_size, device=src.device)

        src_mask = src != PAD_IDX
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_tok = tgt[:, 0]  # <bos>
        for t in range(1, tgt_len):
            logits, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs, src_mask)
            outputs[:, t - 1] = logits
            use_teacher = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)
            input_tok = tgt[:, t] if use_teacher else top1

        return outputs

    @torch.no_grad()
    def translate(self, src: torch.Tensor, max_len: int = 60) -> Tuple[List[int], torch.Tensor]:
        """Dịch greedy 1 câu (batch_size=1) và trả về danh sách token ID + Attention map [tgt_len, src_len]."""
        self.eval()
        assert src.shape[0] == 1, "translate() hỗ trợ từng câu một (batch_size=1)."
        src_mask = src != PAD_IDX
        encoder_outputs, (hidden, cell) = self.encoder(src)

        input_tok = torch.tensor([BOS_IDX], device=src.device)
        pred_ids: List[int] = []
        attns: List[torch.Tensor] = []

        for _ in range(max_len):
            logits, hidden, cell, attn_weights = self.decoder(
                input_tok, hidden, cell, encoder_outputs, src_mask
            )
            next_tok = int(logits.argmax(dim=1).item())
            attns.append(attn_weights.squeeze(0).cpu())
            if next_tok == EOS_IDX:
                break
            pred_ids.append(next_tok)
            input_tok = torch.tensor([next_tok], device=src.device)

        attn_matrix = torch.stack(attns, dim=0) if attns else torch.empty(0, src.shape[1])
        return pred_ids, attn_matrix
