# The Transformer Architecture (Original Encoder-Decoder)

This document visualizes the foundational architecture of the original Transformer, as introduced in the 2017 paper *"Attention Is All You Need"* by Vaswani et al. 

This architecture was designed primarily for Sequence-to-Sequence (Seq2Seq) tasks such as machine translation. It consists of two main stacks: the **Encoder** (which processes the input sequence) and the **Decoder** (which generates the output sequence auto-regressively).

## Flowchart & Data Routing

```mermaid
graph TD
    classDef input fill:#1f9d5a,stroke:#39ff96,stroke-width:2px,color:#fff;
    classDef encoder fill:#111a24,stroke:#4fd1ff,stroke-width:2px,color:#fff;
    classDef decoder fill:#111a24,stroke:#ffb454,stroke-width:2px,color:#fff;
    classDef output fill:#39ff96,stroke:#1f9d5a,stroke-width:2px,color:#000;

    subgraph "Encoder Stack (Nx)"
        E_MHA[Multi-Head Self-Attention]:::encoder
        E_AddNorm1[Add & LayerNorm]:::encoder
        E_FFN[Feed Forward Network]:::encoder
        E_AddNorm2[Add & LayerNorm]:::encoder
    end

    subgraph "Decoder Stack (Nx)"
        D_MHA[Masked Multi-Head Attention]:::decoder
        D_AddNorm1[Add & LayerNorm]:::decoder
        D_Cross[Cross-Attention (Encoder-Decoder)]:::decoder
        D_AddNorm2[Add & LayerNorm]:::decoder
        D_FFN[Feed Forward Network]:::decoder
        D_AddNorm3[Add & LayerNorm]:::decoder
    end

    Input_Source["Source Sequence (e.g. French)"]:::input --> Tok_Source["Input Embedding"]
    Tok_Source --> Pos_Source["+ Positional Encoding"]
    
    Pos_Source --> E_MHA
    E_MHA --> E_AddNorm1
    E_AddNorm1 --> E_FFN
    E_FFN --> E_AddNorm2
    
    Input_Target["Target Sequence (e.g. English)"]:::input --> Tok_Target["Output Embedding"]
    Tok_Target --> Pos_Target["+ Positional Encoding"]
    
    Pos_Target --> D_MHA
    D_MHA --> D_AddNorm1
    
    %% Cross attention link
    E_AddNorm2 -. "Keys (K) & Values (V)" .-> D_Cross
    D_AddNorm1 -- "Queries (Q)" --> D_Cross
    
    D_Cross --> D_AddNorm2
    D_AddNorm2 --> D_FFN
    D_FFN --> D_AddNorm3
    
    D_AddNorm3 --> Linear["Linear Projection"]
    Linear --> Softmax["Softmax"]
    Softmax --> Output_Prob["Next Token Probabilities"]:::output
```

### Key Architectural Concepts
1. **Self-Attention vs Cross-Attention:** The Encoder uses pure self-attention to build a contextual representation of the input. The Decoder uses **Cross-Attention**, where the Queries ($Q$) come from the Decoder, but the Keys ($K$) and Values ($V$) come from the output of the Encoder stack.
2. **Masked Attention:** The first attention mechanism in the Decoder is masked (upper-triangular mask) to ensure the model cannot "look ahead" at future tokens during training.
3. **Residual Connections (Add & Norm):** Gradients flow around the attention and feed-forward layers directly, solving the vanishing gradient problem in deep networks.
