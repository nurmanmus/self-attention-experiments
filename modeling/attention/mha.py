import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_rope, apply_rope_x

class MHA(nn.Module):
    """Multi-Head Attention implementation.
    
    Standard multi-head attention with parallel query, key, and value projections.
    Uses scaled dot-product attention with optional KV caching.
    
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
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Initialize Q, K, V projections
        self.wq = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        self.wk = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        self.wv = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        self.wo = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        
        # Store last attention pattern for analysis
        self.last_attn_pattern = None
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = x @ self.wq.T  # [batch_size, seq_len, d_model]
        k = x @ self.wk.T  # [batch_size, seq_len, d_model]
        v = x @ self.wv.T  # [batch_size, seq_len, d_model]
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
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

class Rope_MHA(nn.Module):
    """Multi-Head Attention with Rotary Position Embeddings (RoPE).
    
    Implements MHA with RoPE applied to queries and keys before attention computation.
    Uses parallel QKV projections and supports KV caching.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        init_scale (float, optional): Initial scale for weight initialization. Defaults to 0.02.
        max_len (int, optional): Maximum sequence length for RoPE. Defaults to 1024.
        rope_theta (float, optional): Base for RoPE frequency computation. Defaults to 1000.0.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, init_scale=0.02, max_len=1024, rope_theta=1000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Initialize Q, K, V projections
        self.wq = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        self.wk = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        self.wv = nn.Parameter(init_scale * torch.randn(d_model, d_model))
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
        
    def apply_rope(self, x, cos, sin, past_length=0):
        """Apply RoPE to input tensor x."""
        x_even = x[..., ::2]  # [batch, n_heads, seq_len, head_dim//2]
        x_odd = x[..., 1::2]  # [batch, n_heads, seq_len, head_dim//2]
        
        # Get RoPE matrices for current position
        cos = cos[past_length:past_length+x.size(2)]  # [seq_len, head_dim//2]
        sin = sin[past_length:past_length+x.size(2)]  # [seq_len, head_dim//2]
        
        # Reshape for broadcasting
        cos = cos.view(x.size(2), 1, -1)  # [seq_len, 1, head_dim//2]
        sin = sin.view(x.size(2), 1, -1)  # [seq_len, 1, head_dim//2]
        
        # Add batch and head dimensions
        cos = cos.unsqueeze(0).expand(x.size(0), -1, x.size(1), -1)  # [batch, seq_len, n_heads, head_dim//2]
        sin = sin.unsqueeze(0).expand(x.size(0), -1, x.size(1), -1)  # [batch, seq_len, n_heads, head_dim//2]
        
        # Transpose to match x shape
        cos = cos.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
        sin = sin.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim//2]
        
        return torch.cat([
            x_even * cos - x_odd * sin,
            x_odd * cos + x_even * sin
        ], dim=-1)
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = x @ self.wq.T  # [batch_size, seq_len, d_model]
        k = x @ self.wk.T  # [batch_size, seq_len, d_model]
        v = x @ self.wv.T  # [batch_size, seq_len, d_model]
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache if provided
        if kv_cache is not None:
            past_key, past_value = kv_cache
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Apply RoPE to queries and keys
        q = self.apply_rope(q, self.cos_cached, self.sin_cached, past_length)
        k = self.apply_rope(k, self.cos_cached, self.sin_cached, 0 if kv_cache is None else past_length)
        
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

class Decoupled_Rope_MHA(nn.Module):
    """Multi-Head Attention with Decoupled Rotary Position Embeddings.
    
    Implements MHA with decoupled RoPE, where only part of Q/K dimensions use RoPE.
    This allows the model to capture both absolute and relative position information.
    
    Args:
        d_model (int): Model dimension
        n_heads (int): Number of attention heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        init_scale (float, optional): Initial scale for weight initialization. Defaults to 0.02.
        max_len (int, optional): Maximum sequence length for RoPE. Defaults to 1024.
        rope_theta (float, optional): Base for RoPE frequency computation. Defaults to 10000.0.
    
    Note:
        The implementation splits Q/K into RoPE and non-RoPE parts:
        - RoPE part: Processes relative positional information
        - Non-RoPE part: Processes content and absolute position information
    """

    def __init__(self, d_model, n_heads, dropout=0.1, init_scale=0.02, max_len=1024, rope_theta=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.init_scale = init_scale
        
        # Split dimensions for decoupled RoPE
        self.qk_nope_dim = self.head_dim // 2
        self.qk_rope_dim = self.head_dim // 2
        
        # Initialize Q, K, V projections
        self.qkv = nn.Parameter(init_scale * torch.randn(3 * d_model, d_model))
        self.wo = nn.Parameter(init_scale * torch.randn(d_model, d_model))
        
        # RoPE parameters
        self.max_seq_len = max_len
        self.rope_theta = rope_theta

        # Precompute RoPE matrices
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.qk_rope_dim, 2).float() / self.qk_rope_dim))
        emb = torch.outer(torch.arange(self.max_seq_len).float(), freqs)
        cos_cached = emb.cos()  # [max_len, rope_dim//4]
        sin_cached = emb.sin()  # [max_len, rope_dim//4]

        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

        # Store last attention pattern for analysis
        self.last_attn_pattern = None
        
    def forward(self, x, kv_cache=None, use_cache=False, past_length=0):
        batch_size, seq_len, _ = x.shape
        
        # Project QKV
        qkv = x @ self.qkv.T  # [batch_size, seq_len, 3*d_model]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Split Q and K for RoPE
        q_nope, q_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        k_nope, k_rope = torch.split(k, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        
        # Apply RoPE to Q and K rope parts
        cos_q = self.cos_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//4]
        sin_q = self.sin_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//4]
        
        # Reshape for broadcasting
        cos_q = cos_q.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
        sin_q = sin_q.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
        
        # Add batch and head dimensions
        cos_q = cos_q.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
        sin_q = sin_q.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
        
        # Transpose to match q_rope shape
        cos_q = cos_q.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
        sin_q = sin_q.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
        
        # Apply RoPE to even and odd dimensions
        q_rope_even = q_rope[..., ::2]  # [batch, n_heads, seq_len, rope_dim//2]
        q_rope_odd = q_rope[..., 1::2]  # [batch, n_heads, seq_len, rope_dim//2]
        
        q_rope = torch.cat([
            q_rope_even * cos_q - q_rope_odd * sin_q,
            q_rope_odd * cos_q + q_rope_even * sin_q
        ], dim=-1)

        if kv_cache is None:
            cos_k = self.cos_cached[:seq_len]  # [seq_len, rope_dim//4]
            sin_k = self.sin_cached[:seq_len]  # [seq_len, rope_dim//4]
            
            # Reshape for broadcasting
            cos_k = cos_k.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
            sin_k = sin_k.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
            
            # Add batch and head dimensions
            cos_k = cos_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
            sin_k = sin_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
            
            # Transpose to match k_rope shape
            cos_k = cos_k.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
            sin_k = sin_k.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
            
            # Apply RoPE to even and odd dimensions
            k_rope_even = k_rope[..., ::2]  # [batch, n_heads, seq_len, rope_dim//2]
            k_rope_odd = k_rope[..., 1::2]  # [batch, n_heads, seq_len, rope_dim//2]
            
            k_rope = torch.cat([
                k_rope_even * cos_k - k_rope_odd * sin_k,
                k_rope_odd * cos_k + k_rope_even * sin_k
            ], dim=-1)
        else:
            past_key, past_value = kv_cache
            past_k_nope, past_k_rope = torch.split(past_key, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
            
            cos_k = self.cos_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//4]
            sin_k = self.sin_cached[past_length:past_length+seq_len]  # [seq_len, rope_dim//4]
            
            # Reshape for broadcasting
            cos_k = cos_k.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
            sin_k = sin_k.view(seq_len, 1, -1)  # [seq_len, 1, rope_dim//4]
            
            # Add batch and head dimensions
            cos_k = cos_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
            sin_k = sin_k.unsqueeze(0).expand(batch_size, -1, self.n_heads, -1)  # [batch, seq_len, n_heads, rope_dim//4]
            
            # Transpose to match k_rope shape
            cos_k = cos_k.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
            sin_k = sin_k.transpose(1, 2)  # [batch, n_heads, seq_len, rope_dim//4]
            
            # Apply RoPE to even and odd dimensions
            k_rope_even = k_rope[..., ::2]  # [batch, n_heads, seq_len, rope_dim//2]
            k_rope_odd = k_rope[..., 1::2]  # [batch, n_heads, seq_len, rope_dim//2]
            
            k_rope = torch.cat([
                k_rope_even * cos_k - k_rope_odd * sin_k,
                k_rope_odd * cos_k + k_rope_even * sin_k
            ], dim=-1)
            
            k_nope = torch.cat([past_k_nope, k_nope], dim=2)
            k_rope = torch.cat([past_k_rope, k_rope], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Combine nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        
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
            k = torch.cat([k_nope, k_rope], dim=-1)
            return out, (k, v)
        return out