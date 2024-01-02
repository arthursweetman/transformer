import torch
import torch.nn as nn
from torch.nn import functional as F
import math

d_model=512
d_ff = 2048
dropout = 0.2
vocab_size=10000

class Head(nn.Module):

    def __init__(self, d_k: int):
        super().__init__()
        self.key = nn.Linear(d_model, d_k, bias=False)
        self.query = nn.Linear(d_model, d_k, bias=False)
        self.value = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x):
        # Calculate attention scores for one head
        # x: (batch_size, seq_len, d_k)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        scores = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        scores = scores.softmax(dim=-1)
        v = self.value(x)
        scores = scores @ v

        return scores

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, d_k):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.heads = nn.ModuleList(Head(d_k) for _ in range(num_heads))
        self.w_o = nn.Linear(num_heads*d_k, d_model, bias=False)  # d_model should be equal to num_heads * d_k

    def forward(self, x):
        if x.shape[-1] != self.num_heads*self.d_k:
            raise ValueError(f"Embedding length of input is not equal to num_heads*d_k: {x.shape[-1]} != {self.num_heads * self.d_k}.")
        scores = torch.cat([attention(x) for attention in self.heads], dim=-1)
        scores = self.w_o(scores)
        return scores

class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Embedding length of input is not equal to d_model: {x.shape[-1]} != {self.d_model}.")
        l1 = self.layer1(x)
        out1 = torch.relu(l1)
        l2 = self.layer2(out1)
        return l2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].detach()

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        d_k = d_model // num_heads
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.PE = PositionalEncoding(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.mh_attention = MultiHeadAttention(num_heads, d_k)
        self.FFN = FeedForward(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, seq):
        # seq is array of indices within the vocab corresponding to words in a sentence
        embeddings = self.embedding(seq)
        PEs = self.PE(embeddings)
        att_scores = self.mh_attention(PEs)
        att_scores_norm = self.ln1(att_scores + PEs)  # PEs are a residual connection at this point
        logits = self.FFN(att_scores_norm)
        logits_norm = self.ln2(logits + att_scores_norm)

        return logits_norm


if __name__ == "__main__":
    model = Transformer(vocab_size, d_model, num_heads=4)
    input_sequence = torch.randint(0, vocab_size, (1, 8))  # Batch size of 1, sequence length of 8
    print(input_sequence)
    output_logits = model(input_sequence)
    print(output_logits.shape)
    print(output_logits)

