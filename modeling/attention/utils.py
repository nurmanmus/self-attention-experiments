import torch

"""
RoPE taken from GPT-NeoX style, via llama in transformers.

References (this was way too hard to do without looking up substantially)

https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247

https://github.com/lucidrains/rotary-embedding-torch/tree/main
"""

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def apply_rope_x(x, cos, sin):
    """Apply rotary position embeddings to input tensor x.
    
    Args:
        x: Input tensor of shape [batch_size, n_heads, seq_len, rope_dim]
        cos: Cosine part of rotary embeddings [batch_size, n_heads, seq_len, rope_dim]
        sin: Sine part of rotary embeddings [batch_size, n_heads, seq_len, rope_dim]
        
    Returns:
        Tensor with rotary position embeddings applied
    """
    # Split input into even and odd dimensions
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    
    # Apply rotary embeddings
    x_rotated = torch.cat([
        x_even * cos - x_odd * sin,
        x_odd * cos + x_even * sin
    ], dim=-1)
    
    return x_rotated

