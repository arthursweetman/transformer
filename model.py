import torch
import torch.nn as nn
from torch.nn import functional as F

d_model=512
d_ff = 2048
dropout = 0.2

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



if __name__ == "__main__":
    x = torch.randn((1, 8, d_model))
    print('input:', x)
    att = MultiHeadAttention(4, 128)
    scores = att(x)
    print('scores:', scores.shape)
    print(scores)
    ffn = FeedForward(d_model, d_ff)
    out_ffn = ffn(scores)
    print('out_ffn', out_ffn.shape)
    print(out_ffn)

