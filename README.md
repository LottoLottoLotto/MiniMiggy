# MiniMiggy 🤖 - A Playful GPT Implementation in tinygrad

MiniMiggy is a compact, efficient implementation of GPT (Generative Pre-trained Transformer) built with tinygrad. Despite its small size, MiniMiggy packs a punch with efficient attention mechanisms, robust checkpointing, and creative text generation capabilities.

## Features

- 🚀 Minimalist GPT architecture powered by tinygrad
- ⚡ Flash Attention for lightning-fast computation
- 💾 Hassle-free checkpoint saving and loading
- 🔄 Built-in BPE Tokenization using tiktoken
- 📝 Creative text generation
- 📊 Real-time training insights

## Requirements

```
tinygrad
numpy
requests
tiktoken
```

## Installation

1. Clone MiniMiggy:
```bash
git clone [your-repo-url]
cd minimiggy
```

2. Install dependencies:
```bash
pip install tinygrad numpy requests tiktoken
```

## Usage

### Training MiniMiggy

```python
from minimiggy import GPT, GPTConfig, TrainConfig, BPETokenizer

# Initialize configurations
config = GPTConfig(
    block_size=64,
    vocab_size=50257,
    n_layer=4,
    n_head=4,
    n_embd=128
)

train_config = TrainConfig(
    batch_size=8,
    learning_rate=1e-3,
    max_iters=2000,
    eval_interval=100
)

# Create and train MiniMiggy
model = GPT(config)
train(model, train_config, train_data)
```

### Generating Text with MiniMiggy

```python
# Initialize tokenizer and model
tokenizer = BPETokenizer()
model = GPT(config)

# Let MiniMiggy create some text
context = "ROMEO:"
x = Tensor([tokenizer.encode(context)], dtype='int32')
generated = model.generate(x, max_tokens=100)
print(tokenizer.decode(generated.numpy().flatten().tolist()))
```

### Saving and Loading MiniMiggy's Brain

```python
# Save MiniMiggy's state
model.save_checkpoint(iter_num, optimizer, "checkpoints")

# Reload MiniMiggy
model.load_checkpoint("checkpoints/ckpt_1000.safetensors", optimizer)
```

## Model Architecture

MiniMiggy's brain consists of:
- Multi-head self-attention with Flash Attention
- Layer normalization
- Feed-forward neural networks
- Residual connections
- Positional embeddings
- Token embeddings

## Configuration Options

### GPT Configuration
```python
GPTConfig(
    block_size=64,      # Maximum sequence length
    vocab_size=50257,   # Size of vocabulary
    n_layer=4,          # Number of transformer layers
    n_head=4,          # Number of attention heads
    n_embd=128         # Embedding dimension
)
```

### Training Configuration
```python
TrainConfig(
    batch_size=8,          # Batch size for training
    learning_rate=1e-3,    # Learning rate
    warmup_steps=100,      # Number of warmup steps
    max_iters=1000,        # Maximum training iterations
    eval_interval=100,     # Evaluation frequency
    eval_iters=20          # Number of evaluation iterations
)
```

## Implementation Details

MiniMiggy features:
- Adam optimizer for learning
- Efficient Flash Attention mechanism
- Built-in gradient checking
- Dual mode support (training/inference)
- BPE tokenization powered by tiktoken

## Memory Management

MiniMiggy is designed to be memory-efficient with:
- Smart tensor allocation
- Automatic gradient cleanup
- Optional checkpointing system

## Contributing

Want to make MiniMiggy even better? Contributions are welcome! Feel free to submit a Pull Request.

## License

MIT

## Acknowledgments

- Built using the fantastic tinygrad framework
- Uses OpenAI's tiktoken for tokenization
- Inspired by Andrej Karpathy's work on GPT
