# Modern LLM Architecture (Decoder-Only GPT)

This document visualizes the **Decoder-only** Transformer architecture. 

Modern Large Language Models (LLMs) such as GPT-3, LLaMA, and Claude discard the Encoder entirely. Instead, they rely on a massive stack of Decoder blocks optimized strictly for auto-regressive next-token prediction.

## Architecture & KV-Caching Flowchart

```mermaid
graph TD
    classDef input fill:#1f9d5a,stroke:#39ff96,stroke-width:2px,color:#fff;
    classDef decoder fill:#111a24,stroke:#ffb454,stroke-width:2px,color:#fff;
    classDef cache fill:#0d1117,stroke:#ffb454,stroke-width:2px,stroke-dasharray: 5 5,color:#fff;
    classDef output fill:#39ff96,stroke:#1f9d5a,stroke-width:2px,color:#000;

    Input["Context Prompt (Text)"]:::input --> BPE["BPE / Subword Tokenizer"]
    BPE --> Emb["Token Embeddings"]
    
    %% Positional info
    Emb --> Pos_RoPE["Rotary Positional Embeddings (RoPE)"]
    
    subgraph "Decoder Block Stack (Nx)"
        Pre_Norm1["Pre-LayerNorm"]:::decoder
        MHA["Masked Self-Attention"]:::decoder
        KVCache[("KV-Cache (Past Keys & Values)")]:::cache
        Add1["Residual Add"]:::decoder
        
        Pre_Norm2["Pre-LayerNorm"]:::decoder
        MLP["MLP (SwiGLU / GeGLU)"]:::decoder
        Add2["Residual Add"]:::decoder
    end

    Pos_RoPE --> Pre_Norm1
    Pre_Norm1 --> MHA
    
    %% KV Cache Logic
    MHA <-->|"Read/Write"| KVCache
    
    MHA --> Add1
    Pos_RoPE -->|"Skip Connection"| Add1
    
    Add1 --> Pre_Norm2
    Pre_Norm2 --> MLP
    
    MLP --> Add2
    Add1 -->|"Skip Connection"| Add2
    
    Add2 --> Final_Norm["Final LayerNorm"]
    Final_Norm --> Unembed["Unembedding / LM Head (Linear)"]
    Unembed --> Softmax["Softmax (Optional for sampling)"]
    Softmax --> Output["Generated Next Token"]:::output
    
    %% Auto-regressive loop
    Output -. "Appended to input sequence" .-> Input
```

### Core LLM Evolutions
1. **Decoder-Only Design:** The prompt and the generated text are processed uniformly through a single stack of masked self-attention blocks.
2. **KV-Caching (Inference Optimization):** During text generation, recalculating the attention matrices for past tokens is highly redundant. The $K$ and $V$ vectors for past tokens are stored in the **KV-Cache** in High Bandwidth Memory (HBM). For every new token, the model only computes the $Q$ vector and performs a dot product against the cached $K$, turning a compute-bound $O(N^2)$ operation into a memory-bound $O(N)$ operation.
3. **Pre-LayerNorm:** Moving the Layer Normalization to the *start* of the residual block (before Attention and MLP) significantly improves training stability at massive scales compared to the original Post-LayerNorm design.
4. **RoPE (Rotary Positional Embeddings):** Instead of adding static embeddings at the bottom, modern LLMs apply a rotation matrix to the $Q$ and $K$ vectors natively within the attention module, leading to superior context length extrapolation.
