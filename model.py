"""GPT Training with Gradient Monitoring"""

from dataclasses import dataclass
import os

# Set environment
os.environ['CUDA'] = '1'
os.environ['DEBUG'] = '1'

from typing import Optional
from tinygrad import Tensor, nn, dtypes, Device
from tinygrad.nn.optim import Adam
import numpy as np
import requests
import tiktoken

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
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_iters: int = 10000
    eval_interval: int = 100
    eval_iters: int = 20

# --- MODEL COMPONENTS ---
class FlashAttention:
    def __init__(self, softmax_scale: float, block_size: int = 256):
        self.softmax_scale = softmax_scale
        self.block_size = block_size

    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        B, H, T, D = q.shape
        out = Tensor.zeros((B, H, T, D))
        normalizers = Tensor.zeros((B, H, T, 1))
        m = Tensor.full((B, H, T, 1), float('-inf'))

        for block_start in range(0, T, self.block_size):
            block_end = min(block_start + self.block_size, T)
            k_block = k[:, :, block_start:block_end]
            v_block = v[:, :, block_start:block_end]
            scores = (q @ k_block.transpose(-2, -1)) * self.softmax_scale
            block_mask = mask[:, :, :, block_start:block_end]
            scores = scores.masked_fill(block_mask == 0, float('-inf'))
            block_m = scores.max(axis=-1, keepdim=True)
            new_m = block_m.maximum(m)
            exp_scale = (m - new_m).exp()
            exp_scores = (scores - new_m).exp()
            out = out * exp_scale + (exp_scores @ v_block)
            normalizers = normalizers * exp_scale + exp_scores.sum(axis=-1, keepdim=True)
            m = new_m

        return out / normalizers

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
        self.flash = FlashAttention(1.0 / (self.head_dim ** 0.5))

    def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        x_ln = self.ln1(x)
        qkv = self.attn(x_ln).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.chunk(3, dim=2)
        q, k, v = [t.squeeze(2).transpose(1, 2) for t in (q, k, v)]
        attn_out = self.flash(q, k, v, mask)
        x = x + self.proj(attn_out.transpose(1, 2).reshape(B, T, C))
        x = x + self.mlp_proj(self.mlp_fc(self.ln2(x)).gelu())
        return x

class GPT:
    def __init__(self, config: GPTConfig):
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self._causal_mask = Tensor.ones((config.block_size, config.block_size)).tril().cast(dtypes.bool).reshape(1, 1, config.block_size, config.block_size)
        
        self.blocks = []
        for i in range(config.n_layer):
            block = Block(config)
            setattr(self, f'block_{i}', block)
            self.blocks.append(block)
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def __call__(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(Tensor.arange(T))
        mask = self._causal_mask[:, :, :T, :T]
        
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        if targets is None:
            return logits, None
        
        loss = logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(targets.reshape(-1))
        return logits, loss

    def generate(self, idx: Tensor, max_tokens: int, temperature: float = 0.8) -> Tensor:
        for _ in range(max_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = logits.softmax().realize()
            probs_np = probs.numpy().flatten()
            next_token_id = np.random.choice(len(probs_np), p=probs_np)
            next_token = Tensor([[next_token_id]], dtype=idx.dtype)
            idx = idx.cat(next_token, dim=1)
        return idx

# --- DATA & TOKENIZER ---
class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

def prepare_data(text: str, tokenizer):
    data = tokenizer.encode(text)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return np.array(train_data, dtype=np.int32), np.array(val_data, dtype=np.int32)

def get_batch(data: np.ndarray, block_size: int, batch_size: int):
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    return Tensor(x, dtype=dtypes.int32), Tensor(y, dtype=dtypes.int32)

# --- GRADIENT MONITORING (FIXED) ---
def check_gradient_flow(params_list):
    """Monitor gradient magnitudes across all layers"""
    grad_norms = []
    grad_means = []
    grad_maxes = []
    
    for param in params_list:
        if param.grad is not None:
            grad_data = param.grad.realize().numpy()
            grad_norms.append(np.sqrt(np.sum(grad_data ** 2)))
            grad_means.append(np.mean(np.abs(grad_data)))
            grad_maxes.append(np.max(np.abs(grad_data)))
    
    return {
        'avg_norm': np.mean(grad_norms),
        'min_norm': np.min(grad_norms),
        'max_norm': np.max(grad_norms),
        'avg_mean': np.mean(grad_means),
        'max_abs': np.max(grad_maxes),
        'count': len(grad_norms),
        'all_norms': grad_norms  # Keep for detailed analysis
    }

# --- TRAINING & EVALUATION ---
def train(model: GPT, train_config: TrainConfig, train_data: np.ndarray, val_data: np.ndarray):
    trainable_parts = [model.tok_emb, model.pos_emb]
    trainable_parts.extend(model.blocks)
    trainable_parts.append(model.ln_f)
    trainable_parts.append(model.head)
    
    params_list = nn.state.get_parameters(trainable_parts)
    optimizer = Adam(params_list, train_config.learning_rate)
    
    total_params = sum(p.numel() for p in params_list)
    print(f"Total parameters: {total_params:,}")
    print(f"Number of parameter tensors: {len(params_list)}")
    print(f"Starting training for {train_config.max_iters} iterations...\n")
    
    for iter_num in range(train_config.max_iters):
        with Tensor.train():
            x, y = get_batch(train_data, model.config.block_size, train_config.batch_size)
            _, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients every 100 iterations
            if iter_num % 100 == 0:
                grad_info = check_gradient_flow(params_list)
                
                print(f"\n--- Gradient Stats at Iter {iter_num} ---")
                print(f"Avg norm: {grad_info['avg_norm']:.6f} | Min: {grad_info['min_norm']:.6f} | Max: {grad_info['max_norm']:.6f}")
                print(f"Avg mean: {grad_info['avg_mean']:.6f} | Max absolute: {grad_info['max_abs']:.6f}")
                print(f"Parameters with gradients: {grad_info['count']}/{len(params_list)}")
                
                # Show distribution of gradient norms
                norms = grad_info['all_norms']
                print(f"First 3 param norms: {norms[:3]}")
                print(f"Last 3 param norms: {norms[-3:]}")
            
            optimizer.step()
        
        if iter_num % train_config.eval_interval == 0 or iter_num == train_config.max_iters - 1:
            train_loss = loss.item()
            val_losses = []
            for _ in range(train_config.eval_iters):
                vx, vy = get_batch(val_data, model.config.block_size, train_config.batch_size)
                _, val_loss = model(vx, vy)
                val_losses.append(val_loss.item())
            
            print(f"\nIter {iter_num}: Train Loss = {train_loss:.4f}, Val Loss = {np.mean(val_losses):.4f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Loading data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    
    print("Initializing tokenizer...")
    tokenizer = BPETokenizer()
    
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    
    print("Preparing data...")
    train_data, val_data = prepare_data(text, tokenizer)
    
    print("Initializing model...")
    model = GPT(config)
    
    train_config = TrainConfig()
    train(model, train_config, train_data, val_data)
    
    print("\n--- Generation ---")
    context = "ROMEO:"
    x = Tensor([tokenizer.encode(context)], dtype=dtypes.int32)
    generated_tokens = model.generate(x, 100).numpy().flatten()
    print(f"Prompt: {context}")
    print(f"Generated text:\n{tokenizer.decode(generated_tokens.tolist())}")

