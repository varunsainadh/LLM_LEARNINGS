import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Expert Implementation of Multi-Head Causal Self-Attention.
    This module computes the exact Scaled Dot-Product Attention.
    
    Note: For production inference and training, this is typically replaced by 
    hardware-aware algorithms like FlashAttention (Dao et al., 2022) to avoid 
    materializing the O(N^2) attention score matrix in HBM (High Bandwidth Memory).
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be perfectly divisible by the number of heads to ensure balanced subspace allocation."
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections mapping the hidden state to Q, K, V subspaces.
        # In optimized frameworks, these are often fused into a single linear layer
        # (e.g., nn.Linear(embed_dim, 3 * embed_dim)) for better GEMM utilization.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection back to the residual stream dimensionality
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 1. Subspace projections via GEMM
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. Reshape and transpose for multi-head computation
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        # We use .transpose() which returns a view, ensuring zero-copy memory manipulation
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 3. Compute Scaled Dot-Product Attention Scores: (Q * K^T) / sqrt(d_k)
        # The scaling factor sqrt(d_k) mitigates vanishing gradients in the softmax backward pass
        # by forcing the variance of the dot products to approach 1.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 4. Apply Causal Auto-regressive Mask
        # We utilize an upper-triangular mask to strictly prevent information leakage 
        # from future tokens into the current timestep's representation.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 5. Softmax normalization to yield convex attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 6. Value aggregation weighted by attention distributions
        attn_output = torch.matmul(attn_weights, v)
        
        # 7. Concatenate heads
        # .contiguous() is crucial here because the previous transpose operations made the tensor 
        # non-contiguous in memory. view() requires a contiguous memory layout.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final linear projection into the residual stream
        output = self.out_proj(attn_output)
        
        return output

if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Tensor dimensionality configuration
    batch_size = 2
    seq_len = 8
    embed_dim = 64
    num_heads = 4
    
    # Simulate an embedded context window
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"Allocated Input Tensor (Batch, Seq_Len, Embed_Dim): {x.shape}")
    
    attention_module = CausalSelfAttention(embed_dim, num_heads)
    
    # Execute graph forward pass
    output = attention_module(x)
    print(f"Graph Output Tensor (Batch, Seq_Len, Embed_Dim): {output.shape}")
    print("\nAttention pass successful. Output retains strict dimensionality for residual injection.")
