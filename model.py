import os
import time
from dataclasses import dataclass
from typing import Optional, List

# Set environment variables before importing tinygrad
os.environ['CUDA'] = '1'
# os.environ['DEBUG'] = '1' # Comment out to reduce noise unless crashing

import numpy as np
import requests
import tiktoken
from tinygrad import Tensor, nn, dtypes, Device, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

print(f"tinygrad's default device is: {Device.DEFAULT}")

# --- CONFIGURATION ---
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128

@dataclass
class TrainConfig:
    batch_size: int = 32  # Increased batch size slightly
    learning_rate: float = 1e-3
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20

# --- MODEL COMPONENTS ---
class Block:
    def __init__(self, config: GPTConfig):
        self.n_head = config.n_head
        self.head_dim = config.n_embd // self.n_head
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        # Layer Norm 1
        x_ln = self.ln1(x)
        
        # Self Attention
        qkv = self.attn(x_ln)
        # Reshape: (B, T, 3, n_head, head_dim)
        qkv = qkv.reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.chunk(3, dim=2)
        
        # Transpose for attention: (B, n_head, T, head_dim)
        q = q.squeeze(2).transpose(1, 2)
        k = k.squeeze(2).transpose(1, 2)
        v = v.squeeze(2).transpose(1, 2)
        
        # Standard Attention Math (let tinygrad compiler optimize this)
        # scale = 1.0 / sqrt(head_dim)
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        
        # Apply Causal Mask (fill -inf where mask is 0)
        # Note: We reshape mask to broadcast: (1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = scores.softmax()
        out = attn @ v
        
        # Reassemble
        out = out.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(out)
        
        # MLP / Feed Forward
        x = x + self.mlp_proj(self.mlp_fc(self.ln2(x)).gelu())
        return x

class GPT:
    def __init__(self, config: GPTConfig):
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        # Pre-compute causal mask
        self.mask = Tensor.ones((config.block_size, config.block_size)).tril().cast(dtypes.bool).reshape(1, 1, config.block_size, config.block_size)
        
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Bias False usually cleaner for head

    def __call__(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        pos = Tensor.arange(T, dtype=dtypes.int32).reshape(1, T)
        
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        # Slice mask to current sequence length
        mask = self.mask[:, :, :T, :T]
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            # Reshape for cross entropy: (B*T, vocab_size) vs (B*T)
            loss = logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(targets.reshape(-1))
            
        return logits, loss

    def generate(self, idx: Tensor, max_tokens: int, temperature: float = 0.8) -> Tensor:
        for _ in range(max_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass only (no gradients needed)
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax()
            
            # Note: We must realize() to pull data to CPU for sampling
            probs_np = probs.realize().numpy().flatten()
            
            # Simple sampling
            next_token_id = np.random.choice(len(probs_np), p=probs_np)
            next_token = Tensor([[next_token_id]], dtype=idx.dtype)
            idx = idx.cat(next_token, dim=1)
        return idx

# --- DATA HELPERS ---
class BPETokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.enc.n_vocab
    def encode(self, text): return self.enc.encode(text)
    def decode(self, tokens): return self.enc.decode(tokens)

def get_batch(data, block_size, batch_size):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return Tensor(x), Tensor(y)

# --- GRADIENT MONITORING ---
def print_grad_stats(params_list, iter_num):
    grad_norms = []
    for param in params_list:
        if param.grad is not None:
            # We must realize() the grad to read it on CPU
            g = param.grad.realize().numpy()
            grad_norms.append(np.linalg.norm(g))
    
    if grad_norms:
        print(f"[Iter {iter_num}] Grad Norms | Avg: {np.mean(grad_norms):.4f} | Max: {np.max(grad_norms):.4f}")

# --- TRAINING ---
if __name__ == "__main__":
    # 1. Data Prep
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    tokenizer = BPETokenizer()
    data = tokenizer.encode(text)
    train_data = np.array(data[:int(0.9*len(data))], dtype=np.int32)
    val_data = np.array(data[int(0.9*len(data)):], dtype=np.int32)

    # 2. Model & Optimizer
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(config)
    
    # Auto-fetch parameters
    params = get_parameters(model)
    print(f"Params: {sum(p.numel() for p in params):,}")
    
    optimizer = Adam(params, lr=1e-3)
    
    # 3. JIT Compiled Step (The Speedup)
    @TinyJit
    def train_step(x, y):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # 4. Training Loop
    start_time = time.time()
    for i in range(1000):
        bx, by = get_batch(train_data, config.block_size, 32)
        
        # Strategy: Run JIT for speed, but run raw python occasionally to check grads
        if i % 100 == 0 and i > 0:
            # Run without JIT to easily inspect intermediates/gradients if needed
            # (Or just to break the monotony and print stats)
            optimizer.zero_grad()
            logits, loss = model(bx, by)
            loss.backward()
            print_grad_stats(params, i)
            optimizer.step()
            print(f"Iter {i} | Loss: {loss.item():.4f}")
        else:
            # Fast path
            loss = train_step(bx, by)
            
    print(f"Training finished in {time.time()-start_time:.2f}s")

    # 5. Inference
    context = "ROMEO:"
    x = Tensor([tokenizer.encode(context)], dtype=dtypes.int32)
    print(f"\nGenerating from: {context}")
    out = model.generate(x, max_tokens=50)
    print(tokenizer.decode(out.numpy()[0].tolist()))

