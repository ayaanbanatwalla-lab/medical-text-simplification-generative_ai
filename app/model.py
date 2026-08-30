"""
GRU sequence-to-sequence model with Luong dot-product attention.

Architecture exactly matches the model trained and evaluated in the paper
"Accuracy-Aware Medical Text Simplification Using Generative AI":
- Shared 50-d embedding layer (encoder + decoder)
- Bidirectional GRU encoder, hidden size 144 per direction
- Unidirectional GRU decoder, hidden size 144
- Luong-style dot-product ("general") attention
- Greedy decoding with no-repeat-trigram blocking
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, SOS, EOS = 0, 1, 2, 3

EMB_DIM = 50
HID_DIM = 144


def tokenize(text: str):
    """Same regex tokenizer used during training/evaluation."""
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text.lower())


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=EMB_DIM, hid_dim=HID_DIM):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.enc_gru = nn.GRU(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.enc_reduce_h = nn.Linear(hid_dim * 2, hid_dim)
        self.enc_reduce_out = nn.Linear(hid_dim * 2, hid_dim)
        self.dec_gru = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.out_proj = nn.Linear(hid_dim * 2, vocab_size)

    def encode(self, src):
        e = self.emb(src)
        enc_out, h = self.enc_gru(e)
        h_cat = torch.cat([h[0], h[1]], dim=-1)
        h0 = torch.tanh(self.enc_reduce_h(h_cat)).unsqueeze(0)
        enc_out_proj = torch.tanh(self.enc_reduce_out(enc_out))
        return enc_out_proj, h0

    def decode_step(self, y_tok, h, enc_out_proj):
        e = self.emb(y_tok)
        dec_out, h = self.dec_gru(e, h)
        scores = torch.bmm(dec_out, enc_out_proj.transpose(1, 2))
        attn_w = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_w, enc_out_proj)
        logits = self.out_proj(torch.cat([dec_out, ctx], dim=-1))
        return logits, h

    @torch.no_grad()
    def generate(self, src, max_len=45, block_repeat_ngram=3):
        """Greedy decoding with no-repeat-trigram blocking (identical to
        the evaluation pipeline used for the paper's reported results)."""
        self.eval()
        enc_out_proj, h = self.encode(src)
        B = src.size(0)
        y = torch.full((B, 1), SOS, dtype=torch.long)
        outputs = [[] for _ in range(B)]
        out_tensors = []
        for _ in range(max_len):
            logits, h = self.decode_step(y, h, enc_out_proj)
            logits = logits.squeeze(1)
            for b in range(B):
                seq = outputs[b]
                if len(seq) >= block_repeat_ngram - 1:
                    banned = set()
                    prefix = tuple(seq[-(block_repeat_ngram - 1):])
                    for j in range(len(seq) - (block_repeat_ngram - 1)):
                        if tuple(seq[j:j + block_repeat_ngram - 1]) == prefix:
                            banned.add(seq[j + block_repeat_ngram - 1])
                    for tok in banned:
                        logits[b, tok] = -1e9
            next_tok = logits.argmax(-1)
            for b in range(B):
                outputs[b].append(int(next_tok[b]))
            out_tensors.append(next_tok.unsqueeze(1))
            y = next_tok.unsqueeze(1)
        return torch.cat(out_tensors, dim=1)
