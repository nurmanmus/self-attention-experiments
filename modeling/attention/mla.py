import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_rope_x

class RopelessMLA_Uncompressed(torch.nn.Module):
    """Multi-Head Latent Attention without RoPE (Uncompressed KV Cache).
    
    Implements MLA with dimension reduction for queries and keys/values through latent projections.
    This version maintains full KV cache without compression.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        
    Note:
        Projects inputs to lower-dimensional latent space before computing attention,
        reducing computation while maintaining model quality.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = (2*d_model) // 3
        self.dh = d_model // n_heads

        # Q projections
        self.W_dq = torch.nn.Parameter(0.02 * torch.randn((d_model, self.q_proj_dim)))
        self.W_uq = torch.nn.Parameter(0.02 * torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        # KV projections
        self.W_dkv = torch.nn.Parameter(0.02 * torch.randn((d_model, self.kv_proj_dim)))
        self.W_ukv = torch.nn.Parameter(0.02 * torch.randn((self.kv_proj_dim, 2*d_model)))
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # output projection
        self.W_o = torch.nn.Parameter(0.02 * torch.randn((d_model, d_model)))

    def forward(self, x, kv_cache=None, past_length=0):
        B, S, D = x.shape

        # Q Projections
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq

        # KV Projections
        compressed_kv = x @ self.W_dkv
        compressed_kv = self.kv_layernorm(compressed_kv)
        KV = compressed_kv @ self.W_ukv
        K, V = torch.split(KV, self.d_model, dim=-1)

        # split into multiple heads
        q_heads = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        k_heads = K.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        v_heads = V.view(B, -1, self.n_heads, self.dh).transpose(1, 2)

        # KV Cache logic
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k_heads = torch.cat([k_cache, k_heads], dim=2)
            v_heads = torch.cat([v_cache, v_heads], dim=2)

        S_full = k_heads.size(2)
        mask = torch.ones((S, S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        mask = mask[None, None, :, :]
        sq_mask = mask == 1

        x = torch.nn.functional.scaled_dot_product_attention(
            q_heads, k_heads, v_heads,
            attn_mask=sq_mask
        )
        x = x.transpose(1, 2).reshape(B, S, D)
        x = x @ self.W_o.T
        return x, (k_heads, v_heads)

class RopelessMLA(nn.Module):
    """Multi-Head Latent Attention without RoPE (Compressed KV Cache).
    
    Implements MLA with dimension reduction and compressed KV cache.
    Maintains latent projections in the cache to save memory.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        init_scale (float, optional): Initial scale for weight initialization. Defaults to 0.02.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, init_scale=0.02):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dh = self.head_dim  # For compatibility with other attention classes
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Low-rank projection dimensions
        self.q_proj_dim = d_model // 2
        self.kv_proj_dim = d_model
        
        # Q projections with low-rank factorization
        self.W_dq = nn.Parameter(init_scale * torch.randn(d_model, self.q_proj_dim))
        self.W_uq = nn.Parameter(init_scale * torch.randn(self.q_proj_dim, d_model))
        self.q_layernorm = nn.LayerNorm(self.q_proj_dim)
        
        # KV projections with low-rank factorization
        self.W_dkv = nn.Parameter(init_scale * torch.randn(d_model, self.kv_proj_dim))
        self.W_ukv = nn.Parameter(init_scale * torch.randn(self.kv_proj_dim, 2 * d_model))
        self.kv_layernorm = nn.LayerNorm(self.kv_proj_dim)
        
        # Additional normalization for position invariance
        self.pre_q_norm = nn.LayerNorm(d_model)
        self.pre_kv_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.wo = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        
        # Store last attention pattern for analysis
        self.last_attn_pattern = None
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        B, S, D = x.shape
        
        # Apply pre-normalization for position invariance
        x_q = self.pre_q_norm(x)
        x_kv = self.pre_kv_norm(x)
        
        # Q projections with low-rank factorization
        compressed_q = x_q @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # KV projections with low-rank factorization
        compressed_kv = x_kv @ self.W_dkv
        compressed_kv = self.kv_layernorm(compressed_kv)
        
        if kv_cache is not None:
            past_compressed_kv, _ = kv_cache
            if len(past_compressed_kv.shape) == 2:
                past_compressed_kv = past_compressed_kv.unsqueeze(0)
            compressed_kv = torch.cat([past_compressed_kv, compressed_kv], dim=1)
        
        # Project to full KV
        KV = compressed_kv @ self.W_ukv
        K, V = torch.chunk(KV, 2, dim=-1)
        
        # Split into heads
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scale Q for stable attention
        Q = Q / math.sqrt(self.head_dim)
        
        # Compute attention scores
        attn_pattern = torch.matmul(Q, K.transpose(-2, -1))
        
        # Store attention pattern for analysis
        self.last_attn_pattern = attn_pattern.detach()
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(S, K.size(2)), diagonal=past_length+1).bool()
        causal_mask = causal_mask.to(x.device)
        attn_pattern = attn_pattern.masked_fill(causal_mask[None, None], float('-inf'))
        attn_weights = F.softmax(attn_pattern, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = out @ self.wo.T
        
        if use_cache:
            if len(compressed_kv.shape) == 2:
                compressed_kv = compressed_kv.unsqueeze(0)
            return out, (compressed_kv, None)
        return out


class MLA(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, rope=True):
        super(MLA, self).__init__()
        self.embed_dim = embed_dim
        self.d_model = embed_dim  # For compatibility with other attention classes
        self.num_heads = num_heads
        self.n_heads = num_heads  # For compatibility with other attention classes
        self.latent_dim = latent_dim
        self.head_dim = embed_dim // num_heads
        self.dh = self.head_dim  # For compatibility with other attention classes
        self.rope = rope

        # Split dimensions for decoupled RoPE
        self.qk_nope_dim = self.head_dim // 4  # Changed from 8 to 4 to ensure proper split
        self.qk_rope_dim = self.head_dim - self.qk_nope_dim
        if self.qk_rope_dim % 2 == 1:
            self.qk_rope_dim -= 1
            self.qk_nope_dim += 1  # Adjust nope_dim to maintain total head_dim

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        assert (
            self.qk_nope_dim + self.qk_rope_dim == self.head_dim
        ), "RoPE dimensions must sum to head dimension"

        # Linear projections for queries
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Low-rank projections for keys and values
        self.kv_down_proj = nn.Linear(embed_dim, latent_dim, bias=False)
        self.k_up_proj = nn.Linear(latent_dim, embed_dim, bias=False)
        self.v_up_proj = nn.Linear(latent_dim, embed_dim, bias=False)

        if self.rope:
            # RoPE parameters
            self.max_seq_len = 1024
            self.rope_theta = 10000.0
            
            # Precompute RoPE matrices
            freqs = 1.0 / (self.rope_theta ** (torch.arange(0, self.qk_rope_dim, 2).float() / self.qk_rope_dim))
            emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
            cos_cached = emb.cos()  # [max_len, rope_dim//2]
            sin_cached = emb.sin()  # [max_len, rope_dim//2]
            
            self.register_buffer("cos_cached", cos_cached)
            self.register_buffer("sin_cached", sin_cached)

        # Store last attention pattern for analysis
        self.last_attn_pattern = None

    def apply_rope(self, x, past_length=0):
        """Apply RoPE to input tensor x."""
        # Get the relevant position encodings
        seq_len = x.size(2)
        cos = self.cos_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//2]
        sin = self.sin_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//2]
        
        # Reshape for broadcasting
        cos = cos.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//2]
        sin = sin.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//2]
        
        # Add batch and head dimensions
        cos = cos.unsqueeze(0).expand(x.size(0), -1, x.size(1), -1)  # [batch, seq_len, n_heads, rope_dim//2]
        sin = sin.unsqueeze(0).expand(x.size(0), -1, x.size(1), -1)  # [batch, seq_len, n_heads, rope_dim//2]
        
        # Transpose to match x shape
        cos = cos.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//2]
        sin = sin.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//2]
        
        # Apply RoPE to even and odd dimensions
        x_even = x[..., ::2]  # [batch, n_heads, seq_len, rope_dim//2]
        x_odd = x[..., 1::2]  # [batch, n_heads, seq_len, rope_dim//2]
        
        x = torch.cat([
            x_even * cos - x_odd * sin,
            x_odd * cos + x_even * sin
        ], dim=-1)
        
        return x

    def forward(self, x, kv_cache=None, use_cache=False, past_length=0, mask=None):
        batch_size, seq_len, _ = x.size()

        # Compute queries
        q = self.q_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Split Q into non-RoPE and RoPE parts
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        # Compute low-rank latent vectors for keys and values
        c_kv = self.kv_down_proj(x)

        # Handle KV cache if provided
        if kv_cache is not None:
            past_c_kv, past_k_rope = kv_cache
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)

        # Reconstruct keys and values
        k = self.k_up_proj(c_kv)
        v = self.v_up_proj(c_kv)

        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Split K into non-RoPE and RoPE parts
        k_nope, k_rope = torch.split(k, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)

        if self.rope:
            # Apply RoPE to queries and keys
            q_rope = self.apply_rope(q_rope, past_length)
            k_rope = self.apply_rope(k_rope, 0)

        # Combine RoPE and non-RoPE parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Store only the non-RoPE part of the attention pattern for analysis
        nope_attn = torch.matmul(q_nope, k_nope.transpose(-2, -1)) / math.sqrt(self.qk_nope_dim)
        self.last_attn_pattern = nope_attn.detach()

        # Apply causal mask if needed
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, k.size(2)), diagonal=past_length+1).bool()
            causal_mask = causal_mask.to(x.device)
            attn_scores = attn_scores.masked_fill(causal_mask[None, None], float('-inf'))
        else:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        if use_cache:
            if len(c_kv.shape) == 2:
                c_kv = c_kv.unsqueeze(0)
            if len(k_rope.shape) == 3:
                k_rope = k_rope.unsqueeze(0)
            return attn_output, (c_kv, k_rope)
        return attn_output

    
    
