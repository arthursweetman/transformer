import torch
import torch.nn as nn
from torch.nn import functional as F
from model import *


def read_data():
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        return text

def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - max_seq_len, (batch_size,))
    x = torch.stack([data[i:i+max_seq_len] for i in ix])
    y = torch.stack([data[i+1:i+max_seq_len+1] for i in ix])
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == '__main__':

    max_iters = 50

    input = read_data()
    chars = sorted(list(set(input)))
    print(''.join(chars))
    print(len(chars))
    vocab_size = len(chars)

    # Begin with a character-level tokenizer
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(input), dtype=torch.long)

    # Train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    mod = GPTDecoderModel(vocab_size)
    mod = mod.to(device)
    print(sum(p.numel() for p in mod.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
    for iter in range(max_iters):

        if iter % 100 == 0 or iter == max_iters - 1:
            losses = estimate_loss(mod)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        logits, loss = mod(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(mod.generate(context, max_new_tokens=500)[0].tolist()))








