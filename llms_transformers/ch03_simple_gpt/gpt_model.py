import torch
import torch.nn as nn
from ch02_attention_mechanism.attention import CausalSelfAttention

class TransformerBlock(nn.Module):
    """
    A standard Transformer Decoder block implementing Pre-Layer Normalization.
    Pre-LayerNorm ensures training stability at extreme depths (e.g., > 96 layers) 
    compared to the original Post-LayerNorm formulation in Vaswani et al.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        
        # Multi-Layer Perceptron (MLP) for non-linear dimensionality expansion.
        # SwiGLU or GeGLU are increasingly preferred over standard GELU in state-of-the-art models (like LLaMA),
        # but GELU remains standard for GPT architectures.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        
    def forward(self, x):
        # Apply self-attention with a residual (skip) connection.
        # Gradients can flow directly through the residual stream without attenuation.
        x = x + self.attn(self.ln_1(x))
        # Apply MLP with a residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

class SimpleGPT(nn.Module):
    """
    Decoder-only Transformer Architecture (GPT-style).
    Optimized for auto-regressive language modeling.
    """
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=128):
        super().__init__()
        # Token and Positional Embeddings
        # In advanced setups, Absolute Positional Embeddings are replaced by Rotary Positional Embeddings (RoPE)
        # which are injected directly into the Q and K matrices during the attention forward pass.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Deep stack of Transformer Blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        
        # Final Layer Normalization before the unembedding projection head
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Unembedding layer: projects hidden states to the vocabulary vocabulary dimension.
        # Weight tying (sharing weights between token_embedding and lm_head) can be utilized to save memory,
        # but modern LLMs often leave them untied for higher expressivity.
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx):
        # idx: Batched tensor of sequence integers, shape: (batch_size, seq_len)
        batch_size, seq_len = idx.size()
        
        # Dynamically generate positional indices based on sequence length
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        
        # Retrieve dense embeddings
        tok_emb = self.token_embedding(idx) # (batch_size, seq_len, embed_dim)
        pos_emb = self.pos_embedding(pos)   # (seq_len, embed_dim)
        
        # Aggregate embeddings to form the initial residual stream
        x = tok_emb + pos_emb
        
        # Execute forward pass through the entire Transformer stack
        x = self.blocks(x)
        
        # Final pre-projection layer normalization
        x = self.ln_f(x)
        
        # Project hidden states to vocabulary dimension to yield next-token logits
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Initialize architectural hyperparameters
    vocab_size = 10000 
    max_seq_len = 512
    embed_dim = 128
    
    # Instantiate the GPT architecture
    model = SimpleGPT(vocab_size=vocab_size, embed_dim=embed_dim, max_seq_len=max_seq_len)
    
    # Generate mock integer sequences (simulating batched tokenized datasets)
    dummy_input = torch.randint(0, vocab_size, (4, 128))
    print("Allocated Batch Tensor:\n", dummy_input.shape)
    
    # Execute Graph Forward Pass
    logits = model(dummy_input)
    
    print(f"\nLogits shape (Batch, Seq_Len, Vocab_Size): {logits.shape}")
    print("Forward pass successful. Logits define an unnormalized distribution over the vocabulary for each temporal step.")
