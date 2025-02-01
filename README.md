# TinyGPT - A Lightweight GPT Implementation in tinygrad

A minimalist implementation of GPT (Generative Pre-trained Transformer) using tinygrad. This implementation includes efficient attention mechanisms, checkpointing, and text generation capabilities.

## Features

- 🚀 Lightweight GPT implementation using tinygrad
- ⚡ Flash Attention for efficient attention computation
- 💾 Checkpoint saving and loading functionality
- 🔄 BPE Tokenization using tiktoken
- 📝 Text generation capabilities
- 📊 Training progress monitoring

## Requirements

```
tinygrad
numpy
requests
tiktoken
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd tinygpt
```

2. Install dependencies:
```bash
pip install tinygrad numpy requests tiktoken
```

## Usage

### Training the Model

```python
from gpt import GPT, GPTConfig, TrainConfig, BPETokenizer

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

# Create model and train
model = GPT(config)
train(model, train_config, train_data)
```

### Generating Text

```python
# Initialize tokenizer and model
tokenizer = BPETokenizer()
model = GPT(config)

# Generate text
context = "ROMEO:"
x = Tensor([tokenizer.encode(context)], dtype='int32')
generated = model.generate(x, max_tokens=100)
print(tokenizer.decode(generated.numpy().flatten().tolist()))
```

### Saving and Loading Checkpoints

```python
# Save checkpoint
model.save_checkpoint(iter_num, optimizer, "checkpoints")

# Load checkpoint
model.load_checkpoint("checkpoints/ckpt_1000.safetensors", optimizer)
```

## Model Architecture

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

- Uses the Adam optimizer
- Implements efficient Flash Attention
- Includes gradient checking functionality
- Supports both training and inference modes
- Uses BPE tokenization from tiktoken

## Memory Management

The implementation includes efficient memory management through:
- Dynamic tensor allocation
- Gradient cleanup after updates
- Optional checkpoint loading/saving

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT 

## Acknowledgments

- Inspired by Andrej Karpathy's work on GPT
- Built using the tinygrad framework
- Uses the tiktoken tokenizer from OpenAI
