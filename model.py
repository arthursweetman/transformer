import torch
import torch.nn as nn
from torch.nn import functional as F
import math

d_model=512
num_heads = 8
d_ff = 2048
dropout = 0.2
max_seq_len=256
batch_size=64
lr=3e-4
eval_iters = 200

class Head(nn.Module):

    def __init__(self, d_k: int, is_decoder):
        super().__init__()
        self.is_decoder=is_decoder
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)
        if is_decoder:
            self.register_buffer('tril', torch.tril(torch.ones(max_seq_len, max_seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Calculate attention scores for one head
        # x: (batch_size, seq_len, d_k)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        if self.is_decoder:
            scores = scores.masked_fill(self.tril[:T,:T] == 0, float('-inf'))  # (B,T,T)
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)
        v = self.value(x)
        scores = scores @ v

        return scores

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_k, is_decoder):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.heads = nn.ModuleList(Head(d_k, is_decoder) for _ in range(num_heads))
        self.w_o = nn.Linear(num_heads*d_k, d_model, bias=False)  # d_model should be equal to num_heads * d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.shape[-1] != self.num_heads*self.d_k:
            raise ValueError(f"Embedding length of input is not equal to num_heads*d_k: {x.shape[-1]} != {self.num_heads * self.d_k}.")
        scores = torch.cat([attention(x) for attention in self.heads], dim=-1)
        scores = self.dropout(self.w_o(scores))
        return scores

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Embedding length of input is not equal to d_model: {x.shape[-1]} != {self.d_model}.")
        l1 = self.layer1(x)
        out1 = torch.relu(l1)
        l2 = self.layer2(out1)
        out = self.dropout(l2)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, is_decoder=False):
        super().__init__()
        self.d_model = d_model
        d_k = d_model // num_heads
        self.mh_attention = MultiHeadAttention(num_heads, d_k, is_decoder)
        self.FFN = FeedForward(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_norm = self.ln1(x)
        att_scores = x + self.mh_attention(x_norm)  # x is the residual connection
        att_scores_norm = self.ln2(att_scores)
        logits = att_scores + self.FFN(att_scores_norm)  # attention scores are the residual connection

        return logits

class GPTDecoderModel(nn.Module):

    def __init__(self, vocab_size, d_model=d_model, num_heads=num_heads):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.PEs = PositionalEncoding(d_model)
        self.decoder_blocks = nn.Sequential(
            TransformerBlock(d_model, num_heads, is_decoder=True),
            TransformerBlock(d_model, num_heads, is_decoder=True),
            TransformerBlock(d_model, num_heads, is_decoder=True),
            TransformerBlock(d_model, num_heads, is_decoder=True),
            TransformerBlock(d_model, num_heads, is_decoder=True),
            TransformerBlock(d_model, num_heads, is_decoder=True)
        )
        self.ln_final = nn.LayerNorm(d_model)  # a final layernorm directly before the final linear layer
        self.lin_final = nn.Linear(d_model, vocab_size)  # The final linear layer

    def forward(self, idx, targets=None):
        embeds = self.embeddings(idx)
        pos_enc = self.PEs(idx)  # (B,T) --> (B,T,C)
        x = embeds + pos_enc
        x = self.decoder_blocks(x)
        x = self.ln_final(x)
        logits = self.lin_final(x)

        if targets is None:
            loss = None
        else:
            loss = self.loss(logits, targets)
        return logits, loss

    def loss(self, logits, targets):
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_end = idx[:, -max_seq_len:]
            logits, loss = self(idx_end)
            logits = logits[:, -1, :]
            probs = logits.softmax(dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = TransformerBlock(d_model, num_heads=num_heads, is_decoder=True)
    input_sequence = torch.randint(0, 65, (1, 8))  # Batch size of 1, sequence length of 8
    print(input_sequence)
    output_logits = model(input_sequence)
    print(output_logits.shape)
    print(output_logits)

