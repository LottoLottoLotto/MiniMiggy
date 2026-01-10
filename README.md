# TinyGPT-Grad
A minimal, educational implementation of a GPT-2 style Transformer trained on the Tiny Shakespeare dataset, built entirely in tinygrad.

This project demonstrates how to implement modern LLM architectures using tinygrad's lazy computation graph, featuring JIT compilation for high-performance training and intermittent "eager" passes for gradient monitoring.

## ⚡ Features
Pure Tinygrad: No PyTorch dependencies.

### JIT Acceleration: 
Uses @TinyJit to compile the training step into a static graph, dramatically reducing Python overhead and kernel launch latency.

### Hybrid Monitoring: 
Implements a smart training loop that runs 99% of steps in JIT mode for speed, but drops into eager mode periodically to inspect gradient norms and debugging stats.

### Flash-Style Attention: 
Efficient broadcasted attention implementation compatible with tinygrad's compiler fusion.

## 🛠️ Installation
You will need Python 3.8+ and the following dependencies:

```
### Install tinygrad (installing from source is recommended for latest compiler features)
pip install git+https://github.com/tinygrad/tinygrad.git

### Install other requirements
pip install numpy tiktoken requests
Note: If you have a specific GPU backend (CUDA, AMD, Metal), ensure your environment supports it. Tinygrad usually auto-detects this.
```

## 🚀 Usage
Simply run the script. It will download the dataset, tokenize it, and begin training.

```
# Standard run (uses default device, usually GPU if available)
python gpt.py

# Force specific backend (e.g., CUDA or METAL)
CUDA=1 python gpt.py
METAL=1 python gpt.py

# Debug mode (see graph/kernel compilation)
DEBUG=1 python gpt.py
```
## 🧠 Architecture Implementation Details
### 1. The Model

The architecture mimics GPT-2 with the following default configuration (adjustable in GPTConfig):
```
Layers: 4
Heads: 4
Embedding Dim: 128
Block Size: 128 context length
```
### 2. The @TinyJit Optimization
In tinygrad, operations are lazy. If we run a standard Python loop, the graph is rebuilt every iteration. To solve this, we decorate the training step:
```
@TinyJit
def train_step(x, y):
    # This graph is captured once and replayed on device
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
```
### 3. Gradient Monitoring Strategy
You cannot easily inspect gradients inside a JIT-compiled function because the tensors remain on the GPU/Accelerator to avoid synchronization costs.

To solve this, this repo uses a Hybrid Loop:

Iter 0-99: Run train_step (JIT). Fast, no CPU sync.

Iter 100: Run standard model(x, y). Slower, but allows us to call .realize() on param.grad to check for vanishing/exploding gradients.

## 📊 Sample Output
```
tinygrad's default device is: CUDA
Params: 820,352
...
Iter 0 | Loss: 10.8241
[Iter 100] Grad Norms | Avg: 0.1523 | Max: 0.4102
Iter 100 | Loss: 2.5412
...
Generating from: ROMEO:
ROMEO: But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
```

## 📝 Configuration
You can modify the GPTConfig and TrainConfig dataclasses at the top of the file to change model size or training hyperparameters.
```
@dataclass
class GPTConfig:
    block_size: int = 256   # Context window
    n_layer: int = 6        # Transformer depth
    n_head: int = 6         # Attention heads
    n_embd: int = 384       # Embedding dimension
```
## Acknowledgements
Andrej Karpathy for the original nanoGPT and tinyshakespeare dataset.

George Hotz for tinygrad.
