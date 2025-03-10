import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_rope_x

class RopelessMQA(nn.Module):
    """Multi-Query Attention implementation without RoPE.
    
    Implements MQA where keys and values are shared across heads while queries are head-specific.
    This reduces memory and computation cost compared to standard MHA.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        init_scale (float, optional): Initial scale for weight initialization. Defaults to 0.02.
        
    Note:
        MQA typically reduces memory usage by a factor of n_heads compared to MHA,
        while maintaining most of the model quality.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, init_scale=0.02):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Initialize Q, K, V projections
        self.wq = nn.Parameter(init_scale * torch.randn(d_model, d_model))  # Head-specific Q
        self.wk = nn.Parameter(init_scale * torch.randn(d_model, self.head_dim))  # Shared K
        self.wv = nn.Parameter(init_scale * torch.randn(d_model, self.head_dim))  # Shared V
        self.wo = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        
        # Store last attention pattern for analysis
        self.last_attn_pattern = None
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = x @ self.wq.T  # [batch_size, seq_len, d_model]
        k = x @ self.wk  # [batch_size, seq_len, head_dim]
        v = x @ self.wv  # [batch_size, seq_len, head_dim]
        
        # Split Q into heads, expand K and V
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Expand to all heads
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Expand to all heads
        
        # Handle KV cache if provided
        if kv_cache is not None:
            past_key, past_value = kv_cache
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
            
        # Compute attention scores
        attn_pattern = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Store attention pattern for analysis
        self.last_attn_pattern = attn_pattern.detach()
        
        # Apply causal mask and softmax
        causal_mask = torch.triu(torch.ones(seq_len, k.size(2)), diagonal=past_length+1).bool()
        causal_mask = causal_mask.to(x.device)
        attn_pattern = attn_pattern.masked_fill(causal_mask[None, None], float('-inf'))
        attn_weights = F.softmax(attn_pattern, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = out @ self.wo.T
        
        if use_cache:
            return out, (k, v)
        return out


class Rope_MQA(nn.Module):
    """Multi-Query Attention with Rotary Position Embeddings (RoPE).
    
    Implements MQA with RoPE applied to queries and keys before attention computation.
    Keys and values are shared across heads while queries are head-specific.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        init_scale (float, optional): Initial scale for weight initialization. Defaults to 0.02.
        max_len (int, optional): Maximum sequence length for RoPE. Defaults to 1024.
        rope_theta (float, optional): Base for RoPE frequency computation. Defaults to 1000.0.
        
    Note:
        Combines the efficiency of MQA with the position-awareness of RoPE.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, init_scale=0.02, max_len=1024, rope_theta=1000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Initialize Q, K, V projections (MQA style)
        self.wq = nn.Parameter(init_scale * torch.randn(d_model, d_model))  # Head-specific Q
        self.wk = nn.Parameter(init_scale * torch.randn(d_model, self.head_dim))  # Shared K
        self.wv = nn.Parameter(init_scale * torch.randn(d_model, self.head_dim))  # Shared V
        self.wo = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        
        # RoPE parameters
        self.max_seq_len = max_len
        self.rope_theta = rope_theta
        
        # Precompute RoPE matrices with stronger position encoding
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()  # [max_len, head_dim//2]
        sin_cached = emb.sin()  # [max_len, head_dim//2]
        
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)
        
        # Store last attention pattern for analysis
        self.last_attn_pattern = None
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V (MQA style)
        q = x @ self.wq.T  # [batch_size, seq_len, d_model]
        k = x @ self.wk  # [batch_size, seq_len, head_dim]
        v = x @ self.wv  # [batch_size, seq_len, head_dim]
        
        # Split Q into heads, expand K and V
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Expand to all heads
        v = v.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # Expand to all heads
        
        # Handle KV cache if provided
        if kv_cache is not None:
            past_key, past_value = kv_cache
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
            
            # Apply RoPE to new key tokens only
            cos_k = self.cos_cached[past_length:past_length+seq_len]  # [seq_len, head_dim//2]
            sin_k = self.sin_cached[past_length:past_length+seq_len]  # [seq_len, head_dim//2]
            
            # Reshape for broadcasting
            cos_k = cos_k.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
            sin_k = sin_k.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
            
            # Add batch and head dimensions
            cos_k = cos_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
            sin_k = sin_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
            
            # Transpose to match k shape
            cos_k = cos_k.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
            sin_k = sin_k.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
            
            # Apply RoPE to even and odd dimensions of new key tokens
            k_new = k[:, :, -seq_len:]  # Get only new tokens
            k_even = k_new[..., ::2]  # [batch, n_heads, seq_len, head_dim//2]
            k_odd = k_new[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]
            
            k_new = torch.cat([
                k_even * cos_k - k_odd * sin_k,
                k_odd * cos_k + k_even * sin_k
            ], dim=-1)
            
            k = torch.cat([k[:, :, :-seq_len], k_new], dim=2)  # Combine with past keys
        else:
            # Apply RoPE to all key tokens
            cos_k = self.cos_cached[:seq_len]  # [seq_len, head_dim//2]
            sin_k = self.sin_cached[:seq_len]  # [seq_len, head_dim//2]
            
            # Reshape for broadcasting
            cos_k = cos_k.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
            sin_k = sin_k.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
            
            # Add batch and head dimensions
            cos_k = cos_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
            sin_k = sin_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
            
            # Transpose to match k shape
            cos_k = cos_k.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
            sin_k = sin_k.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
            
            # Apply RoPE to even and odd dimensions
            k_even = k[..., ::2]  # [batch, n_heads, seq_len, head_dim//2]
            k_odd = k[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]
            
            k = torch.cat([
                k_even * cos_k - k_odd * sin_k,
                k_odd * cos_k + k_even * sin_k
            ], dim=-1)
        
        # Apply RoPE to queries
        cos_q = self.cos_cached[past_length:past_length+seq_len]  # [seq_len, head_dim//2]
        sin_q = self.sin_cached[past_length:past_length+seq_len]  # [seq_len, head_dim//2]
        
        # Reshape for broadcasting
        cos_q = cos_q.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
        sin_q = sin_q.view(seq_len, 1, -1)  # [seq_len, 1, head_dim//2]
        
        # Add batch and head dimensions
        cos_q = cos_q.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
        sin_q = sin_q.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, head_dim//2]
        
        # Transpose to match q shape
        cos_q = cos_q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
        sin_q = sin_q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
        
        # Apply RoPE to even and odd dimensions
        q_even = q[..., ::2]  # [batch, n_heads, seq_len, head_dim//2]
        q_odd = q[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]
        
        q = torch.cat([
            q_even * cos_q - q_odd * sin_q,
            q_odd * cos_q + q_even * sin_q
        ], dim=-1)
        
        # Compute attention scores
        attn_pattern = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Store only the RoPE-affected attention pattern for analysis
        q_rope = q[..., self.head_dim//2:]  # Take the RoPE-affected half
        k_rope = k[..., self.head_dim//2:]  # Take the RoPE-affected half
        rope_attn = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / math.sqrt(self.head_dim//2)
        self.last_attn_pattern = rope_attn.detach()
        
        # Apply causal mask and softmax
        causal_mask = torch.triu(torch.ones(seq_len, k.size(2)), diagonal=past_length+1).bool()
        causal_mask = causal_mask.to(x.device)
        attn_pattern = attn_pattern.masked_fill(causal_mask[None, None], float('-inf'))
        attn_weights = F.softmax(attn_pattern, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Compute output
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = out @ self.wo.T
        
        if use_cache:
            return out, (k, v)
        return out
 
