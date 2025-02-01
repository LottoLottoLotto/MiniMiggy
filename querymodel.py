from dataclasses import dataclass
import time
from typing import Counter, Optional
from tinygrad import Tensor, nn, dtypes
from tinygrad.nn.optim import Adam
#from tinygrad.nn import Module
import numpy as np
import requests
import tiktoken
import os
from tinygrad.nn.state import safe_save, safe_load

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 50257  # Should match tokenizer's vocab size
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128

@dataclass  # Add this dataclass
class TrainConfig:
    batch_size: int = 8
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    max_iters: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20

# Now you can create TrainConfig instances
train_config = TrainConfig(
    batch_size=8,
    learning_rate=1e-3,
    max_iters=1000,
    eval_interval=100
)

class BPETokenizer:
    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

class Block:
    def __init__(self, config: GPTConfig):
        self.n_head = config.n_head
        self.head_dim = config.n_embd // self.n_head

        # Layers (no mask storage!)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.mlp_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        # Create mask during forward pass
        mask = Tensor.ones((T, T)).tril().cast(dtypes.bool)
        current_mask = mask.reshape(1, 1, T, T)

        # Attention
        x_ln = self.ln1(x)
        qkv = self.attn(x_ln).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.chunk(3, dim=2)
        q, k, v = [t.squeeze(2).transpose(1, 2) for t in (q, k, v)]

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(current_mask == 0, float('-inf')).softmax()
        x = x + self.proj((att @ v).transpose(1, 2).reshape(B, T, C))

        # MLP
        x_ln = self.ln2(x)
        x = x + self.mlp_proj(self.mlp_fc(x_ln).gelu())
        return x

class GPT:
    def __init__(self, config: GPTConfig):
        self.config = config

        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # Blocks (no manual registration!)
        self.blocks = [Block(config) for _ in range(config.n_layer)]

        # Final layers
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=True)

    def __call__(self, idx: Tensor, targets: Optional[Tensor] = None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(Tensor.arange(T))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = logits.reshape(-1, logits.shape[-1]).sparse_categorical_crossentropy(targets.reshape(-1))
        return logits, loss

    def generate(self, idx: Tensor, max_tokens: int) -> Tensor:
        print("\nStarting generation:")
        original_length = idx.shape[1]
        print(f"Initial input shape: {idx.shape}")

        # Pad if needed
        if idx.shape[1] < self.config.block_size:
            padding = self.config.block_size - idx.shape[1]
            idx = idx.pad((None, (0, padding))).contiguous()
            print(f"Padded input to block_size: {idx.shape}")
        else:
            print("Input already fits block_size")

        for step in range(max_tokens):
            print(f"\nGeneration step {step+1}/{max_tokens}")
            idx_cond = idx[:, -self.config.block_size:]
            print(f"Processing context shape: {idx_cond.shape}")

            logits = self(idx_cond, None)[0]
            print(f"Raw logits shape: {logits.shape}")

            next_token = logits[:, -1].argmax().reshape(1, 1)
            print(f"Selected next token: {next_token.numpy().item()}")

            idx = idx.cat(next_token, dim=1)[:, -self.config.block_size:]
            print(f"Updated sequence length: {original_length + step + 1}")

        print("\nGeneration complete!")
        return idx

    def save_checkpoint(self, iter: int, optimizer: Adam, checkpoint_dir: str = "checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create flattened state dict
        state_dict = {}

        # Model parameters
        for k, v in nn.state.get_state_dict(self).items():
            state_dict[f"model.{k}"] = v.detach().contiguous()

        # Optimizer state
        for k, v in nn.state.get_state_dict(optimizer).items():
            state_dict[f"optimizer.{k}"] = v.detach().contiguous()

        # Training state
        state_dict["iter"] = Tensor([iter], dtype=dtypes.int32)

        # Config (convert to tensors)
        for k, v in self.config.__dict__.items():
            if isinstance(v, (int, float)):
                state_dict[f"config.{k}"] = Tensor([v], dtype=dtypes.int32 if isinstance(v, int) else dtypes.float32)
            else:
                # Handle other types as needed
                raise ValueError(f"Unsupported config type: {type(v)} for key {k}")

        checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{iter}.safetensors")
        safe_save(state_dict, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, path: str, optimizer: Optional[Adam] = None):
        state_dict = safe_load(path)

        # Load model parameters
        model_params = {k.split("model.", 1)[1]: v for k, v in state_dict.items() if k.startswith("model.")}
        nn.state.load_state_dict(self, model_params)

        # Load optimizer state
        if optimizer:
            optim_params = {k.split("optimizer.", 1)[1]: v for k, v in state_dict.items() if k.startswith("optimizer.")}
            nn.state.load_state_dict(optimizer, optim_params)

        # Load config (if needed)
        config_params = {}
        for k in list(self.config.__dict__.keys()):
            if f"config.{k}" in state_dict:
                config_params[k] = state_dict[f"config.{k}"].numpy().item()
        self.config = GPTConfig(**config_params)

        print(f"Loaded checkpoint from {path}, iteration {state_dict['iter'].numpy().item()}")

def prepare_data(text: str, tokenizer, config: GPTConfig):
    # Tokenize and split into train/val
    data = tokenizer.encode(text)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return np.array(train_data, dtype=np.uint16), np.array(val_data, dtype=np.uint16)

def check_gradients(params):
    for i, p in enumerate(params):
        if p.grad is None:
            print(f"Param {i} ({p.shape}): No gradient!")
        else:
            grad = p.grad.numpy()
            print(f"Param {i} ({p.shape}): Mean grad {np.abs(grad).mean():.2e}")

def get_batch(data: np.ndarray, block_size: int, batch_size: int):
    """Randomly select batch from data"""
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = np.stack([data[i:i+block_size] for i in ix])
    y = np.stack([data[i+1:i+block_size+1] for i in ix])
    # Change dtype specification to use strings instead of np types
    return Tensor(x, dtype='int32'), Tensor(y, dtype='int32')

def train(model: GPT, train_config: TrainConfig, train_data: np.ndarray):
    optimizer = Adam(nn.state.get_parameters(model), train_config.learning_rate)

    # Check parameters after fix
    params = nn.state.get_parameters(model)
    print(f"Total parameters: {len(params)}")
    print("Parameter types:")
    for p in params:
        print(f"- {p.shape} ({p.dtype})")
    print(f"Initial training mode: {Tensor.training}")  # Should be False

    for iter in range(train_config.max_iters):
        with Tensor.train():
            print(f"Iter {iter} training mode: {Tensor.training}")
            x, y = get_batch(train_data, model.config.block_size, train_config.batch_size)
            _, loss = model(x, y)
            print(f"Iter {iter}: Loss = {loss.numpy().item():.4f}")
            optimizer.zero_grad()
            loss.backward()
            check_gradients(nn.state.get_parameters(model))
            # Check gradients
            missing = [p for p in params if p.grad is None]
            if missing:
                print(f"\nMissing gradients for {len(missing)} parameters:")
                for p in missing:
                    print(f"- {p.shape} ({p.dtype})")
                raise RuntimeError("Parameters disconnected from computation graph")

            optimizer.step()
    model.save_checkpoint(train_config.max_iters, optimizer, checkpoint_dir="final")
    model_weights = {k: v.detach().contiguous() for k, v in nn.state.get_state_dict(model).items()}
    safe_save(model_weights, "gpt_model.safetensors")

if __name__ == "__main__":
    # Load text data
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text

    # Initialize tokenizer
    tokenizer = BPETokenizer()  # or BPETokenizer()

    # Update GPT config with tokenizer's vocab size
    config = GPTConfig(
        block_size=64,
        vocab_size=tokenizer.vocab_size,  # Use actual vocab size
        n_layer=4,
        n_head=4,
        n_embd=128
    )

    # Prepare data
    train_data, val_data = prepare_data(text, tokenizer, config)

    # Initialize and train model
    model = GPT(config)
    train_config = TrainConfig(
        batch_size=8,
        learning_rate=1e-3,
        max_iters=2000,
        eval_interval=100
    )
    train(model, train_config, train_data)

    # Generate sample text
    context = "ROMEO:"
    x = Tensor([tokenizer.encode(context)], dtype='int32')  # Use string dtype
    generated = model.generate(x, 100).numpy().flatten()
    print(f"\nGenerated text:\n{tokenizer.decode(generated.tolist())}")

# Create and run model
print("Creating model...")
model = GPT(GPTConfig(n_layer=2, n_head=2))  # Reduced size for demo
print("\nModel created!")

x = Tensor.randint((1, 64), low=0, high=256)
print(f"\nInput tensor shape: {x.shape}")
