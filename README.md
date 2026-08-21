# Large Language Models From Scratch: A Comprehensive Guide

Welcome to the **LLM Learning** repository. This guide serves as an expert-level, step-by-step walkthrough for building a Generative Pre-trained Transformer (GPT) from the ground up, heavily inspired by the methodologies in Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

This document abandons high-level abstractions in favor of raw PyTorch implementations, exploring the underlying math, memory complexities, and architectural decisions that power modern AI.

---

## Table of Contents
1. [Chapter 1: Data Processing & Tokenization](#chapter-1-data-processing--tokenization)
2. [Chapter 2: The Attention Mechanism](#chapter-2-the-attention-mechanism)
3. [Chapter 3: The GPT Architecture](#chapter-3-the-gpt-architecture)
4. [Chapter 4: The Training Loop](#chapter-4-the-training-loop)
5. [Chapter 5: Advanced Scaling Concepts](#chapter-5-advanced-scaling-concepts)

---

## Transformers vs. LLMs: The Architecture Connection

While the terms "Transformer" and "Large Language Model" (LLM) are often used interchangeably, they refer to different steps in the evolution of AI architecture.

**The Original Transformer:**
Introduced in the 2017 paper *"Attention Is All You Need"*, the original Transformer was designed for Sequence-to-Sequence (Seq2Seq) tasks like machine translation. It consists of two halves:
1. **An Encoder:** Processes the input text and builds a contextual representation using Self-Attention.
2. **A Decoder:** Generates the output text auto-regressively, using **Cross-Attention** to "look back" at the Encoder's representation.
👉 *See the classic architecture mapped out in the [Transformer Flowchart](Transformer_Architecture_Flowchart.md).*

**The Modern LLM (e.g., GPT):**
Modern Large Language Models like GPT-3, LLaMA, and Claude are scaled-up variations of the Transformer, but they **drop the Encoder entirely**. They rely strictly on a massive stack of **Decoder blocks**. By processing the prompt and the generated text through the same Decoder stack using masked self-attention, they are heavily optimized for auto-regressive next-token prediction.
👉 *See how the modern generative architecture functions in the [LLM/GPT Flowchart](LLM_GPT_Architecture_Flowchart.md).*

---

## Chapter 1: Data Processing & Tokenization

Before a neural network can process text, linguistic data must be discretized into a continuous integer space. While basic approaches use word-level splitting, modern LLMs use sub-word tokenization like **Byte-Pair Encoding (BPE)**.

### Sub-word Tokenization Example
Instead of writing a regex-based tokenizer, let's look at how we convert text to tensor inputs using a vocabulary:

```python
import torch

class SimpleTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        
    def encode(self, text):
        # A naive split for demonstration. In reality, we use BPE merges.
        tokens = text.lower().split()
        return [self.vocab.get(token, self.vocab["<|unk|>"]) for token in tokens]

# Example usage
vocab = {"hello": 0, "world": 1, "<|unk|>": 2}
tokenizer = SimpleTokenizer(vocab)

text = "hello world"
token_ids = tokenizer.encode(text) # Output: [0, 1]

# Convert to PyTorch Tensor
input_tensor = torch.tensor(token_ids).unsqueeze(0) # Shape: (1, 2)
print("Input Tensor:", input_tensor)
```

---

## Chapter 2: The Attention Mechanism

The computational bottleneck of the Transformer is **Multi-Head Self-Attention**. It computes relationships across the sequence simultaneously, introducing an $O(N^2 d)$ time and memory complexity.

### Scaled Dot-Product Attention
The math behind attention is defined as:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Here is the exact PyTorch implementation with causal masking (essential for auto-regressive generation):

```python
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Fused linear projection for Q, K, V for better GEMM utilization
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.size() # Batch, Time (Seq_Len), Channels (Embed_Dim)
        
        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head computation: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute Attention Scores (Q * K^T) / sqrt(d_k)
        # .transpose(-2, -1) swaps the last two dimensions for matrix multiplication
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply Causal Mask (Upper Triangular)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and multiply by Values
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        # .contiguous() is crucial here as transpose fractured the memory layout
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)
```

> **Expert Note on FlashAttention:** In production, materializing the `attn_scores` matrix in HBM (High Bandwidth Memory) causes memory limits to be hit rapidly. Hardware-aware algorithms like FlashAttention fuse the masking and softmax operations on the GPU SRAM, dropping memory complexity from $O(N^2)$ to $O(N)$.

---

## Chapter 3: The GPT Architecture

Modern LLMs utilize a **Decoder-only** architecture, dropping the Encoder from the original Vaswani et al. paper.

### Assembling the Transformer Block
A standard block uses Pre-Layer Normalization to maintain gradient stability at extreme depths.

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        
        # Multi-Layer Perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # Residual connections allow gradients to flow unimpeded
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### The Full GPT Model
```python
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        return self.lm_head(x) # Returns unnormalized logits
```

---

## Chapter 4: The Training Loop

Training an LLM requires optimizing the network against a Cross-Entropy loss objective via auto-regressive next-token prediction.

### Key Optimization Mechanics
- **AdamW:** We strictly use AdamW over Adam. Decoupling weight decay from the gradient update is necessary for proper regularization in Transformers.
- **Gradient Clipping:** Transformers are prone to loss spikes; clipping gradient norms prevents the optimizer from taking catastrophically large steps.

```python
import torch.optim as optim

def train_step(model, X_batch, Y_batch, optimizer):
    model.train()
    
    # Forward pass
    logits = model(X_batch)
    
    # Reshape for CrossEntropyLoss which expects (N, C)
    B, T, C = logits.size()
    logits_flat = logits.view(B * T, C)
    targets_flat = Y_batch.view(B * T)
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True) # Optimized memory clearing
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()
```

---

## Chapter 5: Advanced Scaling Concepts

To transition from a toy model to a billion-parameter production model, several architectural modifications are required:

1. **Rotary Positional Embeddings (RoPE):**
   Absolute positional embeddings (used above) struggle to generalize to longer context windows. RoPE applies a rotation matrix directly to the Queries and Keys, ensuring that the dot product strictly represents relative distances.
   
2. **KV-Caching for Inference:**
   During generation, recalculating attention for all past tokens is computationally redundant and renders inference $O(N^2)$. By caching the Keys and Values of past tokens, inference becomes a memory-bandwidth-bound $O(N)$ operation.

3. **ZeRO & FSDP Distributed Training:**
   A 70B parameter model requires ~140GB of VRAM just to store its FP16 weights, far exceeding a single A100 GPU (80GB). Frameworks like Fully Sharded Data Parallel (FSDP) shard the model weights, gradients, and optimizer states across multiple GPUs, performing Just-In-Time all-gather operations during the forward pass.
