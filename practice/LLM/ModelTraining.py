# mini_gpt.py
# Tiny GPT-style LLM (character-level) 🔥

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

data_file = os.path.join(script_dir, 'data.txt')


# --------------------
# CONFIG
# --------------------
batch_size = 64       # how many sequences per batch
block_size = 128      # max context length
max_iters = 3000      # training steps (increase on Colab)
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 256          # embedding dimension
n_head = 8            # number of attention heads
n_layer = 6           # number of transformer blocks
dropout = 0.1
print("Using device:", device)

# --------------------
# DATA LOADING
# --------------------
with open(data_file, 'r', encoding='utf-8') as f:

    text = f.read()

print("Data length:", len(text))

# Get all unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

# char <-> id mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --------------------
# DATA BATCHING
# --------------------
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

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

# --------------------
# MODEL DEFINITIONS
# --------------------

class Head(nn.Module):
    """Single self-attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # attention scores
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # weighted sum of values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """Simple MLP after attention"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: attention + feedforward"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # residual + attention
        x = x + self.ffwd(self.ln2(x)) # residual + MLP
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd) # final norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)              # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb                                  # (B, T, C)

        x = self.blocks(x)                                     # (B, T, C)
        x = self.ln_f(x)                                       # (B, T, C)
        logits = self.lm_head(x)                               # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # flatten for cross entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]       # last time step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat((idx, next_id), dim=1)
        return idx

# --------------------
# TRAIN
# --------------------
model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # periodically report
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # get a batch
    xb, yb = get_batch('train')

    # forward
    logits, loss = model(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------
# SAMPLE FROM MODEL
# --------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # start with all-zero token
generated = model.generate(context, max_new_tokens=500)[0].tolist()
print("\n--- SAMPLE OUTPUT ---\n")
print(decode(generated))


# --------------------
# SAVE MODEL
# --------------------
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer
}

torch.save(checkpoint, os.path.join(save_dir, "mini_gpt.pth"))
print("✅ Model saved to saved_model/mini_gpt.pth")
