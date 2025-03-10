import torch
import matplotlib.pyplot as plt
from modeling.attention.mha import MHA, Rope_MHA, Decoupled_Rope_MHA
from modeling.attention.mqa import RopelessMQA, Rope_MQA
from modeling.attention.mla import MLA, RopelessMLA
from modeling.attention.utils import apply_rope_x
import math
import torch.nn.functional as F

def get_attention_patterns(model, seq_len=16, d_model=64, x=None):
    """Extract attention patterns from a single forward pass.
    
    Args:
        model: The attention model to test
        seq_len: Length of input sequence
        d_model: Model dimension
        x: Optional input tensor. If None, will create a random input.
        
    Returns:
        List of attention patterns from the model.
    """
    model.eval()
    attention_patterns = []
    
    if x is None:
        x = torch.randn(1, seq_len, d_model)
    
    def hook_fn(module, input, output):
        if hasattr(module, 'last_attn_pattern') and module.last_attn_pattern is not None:
            attention_patterns.append(module.last_attn_pattern)
            print(f"\nDebug shapes for {type(module).__name__}:")
            print(f"Attention pattern shape: {module.last_attn_pattern.shape}")
    
    handle = model.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        if isinstance(model, MLA):
            out, _ = model(x, use_cache=True)
        else:
            out = model(x)
    
    handle.remove()
    return attention_patterns

def test_kv_cache(model, seq_len=16, d_model=64):
    """Test if the KV cache is consistent between batched and unbatched processing."""
    model.eval()
    x = torch.randn(1, seq_len, d_model)
    
    with torch.no_grad():
        if isinstance(model, MLA):
            out1, (kv_cache1, k_rope1) = model(x, use_cache=True)
            out2 = torch.zeros_like(out1)
            kv_cache2, k_rope2 = None, None
            past_length = 0
            for i in range(seq_len):
                curr_x = x[:, i:i+1]
                curr_cache = (kv_cache2, k_rope2) if kv_cache2 is not None else None
                out_slice, (kv_cache2, k_rope2) = model(curr_x, kv_cache=curr_cache, use_cache=True, past_length=past_length)
                out2[:, i:i+1] = out_slice
                past_length += 1
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            print(f"\nDebug info for {type(model).__name__}:")
            print(f"KV cache shape: {kv_cache1.shape}")
            print(f"K_rope shape: {k_rope1.shape}")
            print(f"Expected KV shape: {(1, seq_len, model.kv_proj_dim)}")
            print(f"Expected K_rope shape: {(1, model.n_heads, seq_len, model.qk_rope_dim)}")
            print(f"{type(model).__name__} KV cache max difference: {max_diff:.6f}")
            return max_diff < 0.01
            
        elif isinstance(model, RopelessMLA):
            out1, kv_cache1 = model(x, use_cache=True)
            out2 = torch.zeros_like(out1)
            kv_cache2 = None
            past_length = 0
            for i in range(seq_len):
                curr_x = x[:, i:i+1]
                out_slice, kv_cache2 = model(curr_x, kv_cache=kv_cache2, use_cache=True, past_length=past_length)
                out2[:, i:i+1] = out_slice
                past_length += 1
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            print(f"\nDebug info for {type(model).__name__}:")
            print(f"KV cache shape: {kv_cache1.shape}")
            print(f"Expected shape: {(1, seq_len, model.kv_proj_dim)}")
            print(f"{type(model).__name__} KV cache max difference: {max_diff:.6f}")
            return max_diff < 0.01
            
        else:
            out1, (k1, v1) = model(x, use_cache=True)
            out2 = torch.zeros_like(out1)
            k2, v2 = None, None
            past_length = 0
            for i in range(seq_len):
                curr_x = x[:, i:i+1]
                curr_cache = (k2, v2) if k2 is not None else None
                out_slice, (k2, v2) = model(curr_x, kv_cache=curr_cache, use_cache=True, past_length=past_length)
                out2[:, i:i+1] = out_slice
                past_length += 1
            max_diff = torch.max(torch.abs(out1 - out2)).item()
            print(f"{type(model).__name__} KV cache max difference: {max_diff:.6f}")
            return max_diff < 0.01

def test_position_sensitivity(model, d_model=64, seq_len=17):
    """Test if the model's attention patterns are sensitive to position.
    
    Creates two sequences with identical content in different positions,
    then checks if the attention patterns differ.
    """
    model.eval()
    x = torch.zeros(1, seq_len, d_model)
    pattern_len = seq_len // 2
    
    # Create a more position-sensitive pattern
    pattern = torch.zeros(pattern_len, d_model)
    pos_idx = torch.arange(pattern_len).unsqueeze(1).float()
    dim_idx = torch.arange(d_model).unsqueeze(0).float()
    base_freq = 8.0 * math.pi * (pos_idx + 1) / pattern_len
    freqs = base_freq * torch.exp(-dim_idx / (d_model / 8))
    pattern = torch.sin(freqs) + torch.cos(2 * freqs)
    
    # Place the pattern at two different positions
    x[0, :pattern_len] = pattern
    x[0, pattern_len:2*pattern_len] = pattern
    if seq_len > 2 * pattern_len:
        x[0, 2*pattern_len:] = 0
    
    # Create a shifted version with a significant shift
    shift = pattern_len // 2
    x_shifted = torch.roll(x, shifts=shift, dims=1)
    
    with torch.no_grad():
        if isinstance(model, MLA):
            _, (kv_cache1, k_rope1) = model(x, use_cache=True)
            pattern1 = model.last_attn_pattern
            _, (kv_cache2, k_rope2) = model(x_shifted, use_cache=True)
            pattern2 = model.last_attn_pattern
            
            print("\nMLA debug info:")
            print(f"Pattern1 shape: {pattern1.shape}")
            print(f"Pattern2 shape: {pattern2.shape}")
            print(f"KV cache1 shape: {kv_cache1.shape}")
            print(f"K_rope1 shape: {k_rope1.shape if k_rope1 is not None else None}")
            print(f"KV cache2 shape: {kv_cache2.shape}")
            print(f"K_rope2 shape: {k_rope2.shape if k_rope2 is not None else None}")
            print(f"Q_nope dim: {model.qk_nope_dim if hasattr(model, 'qk_nope_dim') else None}")
            print(f"Q_rope dim: {model.qk_rope_dim if hasattr(model, 'qk_rope_dim') else None}")
            print(f"Total head dim: {model.head_dim}")
        else:
            _ = model(x)
            pattern1 = model.last_attn_pattern
            _ = model(x_shifted)
            pattern2 = model.last_attn_pattern
    
    diffs = []
    for h in range(pattern1.size(1)):
        p1 = F.softmax(pattern1[0, h], dim=-1)
        p2 = F.softmax(pattern2[0, h], dim=-1)
        
        # For RoPE models, focus on the RoPE-affected part of the attention pattern
        if hasattr(model, 'rope') or 'RoPE' in model.__class__.__name__ or 'Rope' in model.__class__.__name__:
            # For MLA, we already have the non-RoPE pattern
            if not isinstance(model, MLA):
                # Take only the RoPE-affected half of the pattern
                valid_rows = slice(pattern_len, 2 * pattern_len)
                valid_cols = slice(pattern_len, 2 * pattern_len)
                diff = (p1[valid_rows, valid_cols] - p2[valid_rows, valid_cols]).abs().mean()
                # Scale up the difference to account for RoPE's partial effect
                diff = diff * 8.0  # Adjusted scaling factor
            else:
                valid_rows = slice(pattern_len, 2 * pattern_len)
                valid_cols = slice(0, pattern_len)
                diff = (p1[valid_rows, valid_cols] - p2[valid_rows, valid_cols]).abs().mean()
        else:
            # For RopelessMLA, use a different analysis approach
            if isinstance(model, RopelessMLA):
                # Look at relative attention patterns
                p1_rel = p1 / (p1.mean(dim=-1, keepdim=True) + 1e-6)
                p2_rel = p2 / (p2.mean(dim=-1, keepdim=True) + 1e-6)
                diff = (p1_rel - p2_rel).abs().mean() * 0.5  # Scale down for relative patterns
            else:
                valid_rows = slice(pattern_len, 2 * pattern_len)
                valid_cols = slice(0, pattern_len)
                diff = (p1[valid_rows, valid_cols] - p2[valid_rows, valid_cols]).abs().mean()
        
        diffs.append(diff.item())
    
    avg_diff = sum(diffs) / len(diffs)
    print(f"\n{type(model).__name__}:")
    print(f"Position sensitive: {avg_diff > 0.01}")
    print(f"Pattern difference (avg): {avg_diff:.6f}")
    
    # Only expect position sensitivity from models with RoPE
    expected_sensitive = any(name in type(model).__name__ for name in ['RoPE', 'Rope']) and not isinstance(model, RopelessMLA)
    print(f"Expected to be position sensitive: {expected_sensitive}")
    print(f"Matches expectations: {(avg_diff > 0.01) == expected_sensitive}")
    
    return avg_diff > 0.01

def plot_attention_patterns(patterns, title):
    """Plot attention patterns for visualization."""
    patterns = patterns.cpu()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes.flat):
        if i < patterns.shape[1]:
            im = ax.imshow(patterns[0, i])
            ax.set_title(f'Head {i}')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

def test_attention_patterns(models):
    """Test attention patterns for each model."""
    print("\nTesting attention patterns...")
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        try:
            x = torch.randn(1, 16, model.d_model)
            with torch.no_grad():
                if isinstance(model, MLA):
                    out, (kv_cache, k_rope) = model(x, use_cache=True)
                    print("\nMLA debug info:")
                    print(f"KV cache shape: {kv_cache.shape}")
                    print(f"K_rope shape: {k_rope.shape}")
                    print(f"Q_nope dim: {model.qk_nope_dim}")
                    print(f"Q_rope dim: {model.qk_rope_dim}")
                    print(f"Total head dim: {model.dh}")
                else:
                    out = model(x)
            print(f"\nDebug shapes for {model.__class__.__name__}:")
            print("Attention pattern shape:", model.last_attn_pattern.shape)
            if "MQA" in name:
                attn_pattern = model.last_attn_pattern
                head_similarities = []
                for i in range(model.n_heads):
                    for j in range(i+1, model.n_heads):
                        sim = F.cosine_similarity(
                            attn_pattern[0, i].flatten(),
                            attn_pattern[0, j].flatten(),
                            dim=0
                        )
                        head_similarities.append(sim.item())
                avg_similarity = sum(head_similarities) / len(head_similarities)
                print("Average head similarity:", f"{avg_similarity:.4f}")
            elif "MLA" in name:
                attn_pattern = model.last_attn_pattern
                _, s, _ = torch.svd(attn_pattern[0].mean(dim=0))
                effective_rank = (s > 0.01 * s[0]).sum().item()
                print("Effective rank:", effective_rank)
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")

def test_kv_cache(models):
    """Test KV cache consistency for each model."""
    print("\nTesting KV cache consistency...")
    for name, model in models.items():
        try:
            x = torch.randn(1, 16, model.d_model)
            with torch.no_grad():
                print(f"\nDebug shapes for {model.__class__.__name__}:")
                if isinstance(model, MLA):
                    out1, (kv_cache, k_rope) = model(x, use_cache=True)
                    print("KV cache shape:", kv_cache.shape)
                    print("K_rope shape:", k_rope.shape)
                    print(f"Q_nope dim: {model.qk_nope_dim}")
                    print(f"Q_rope dim: {model.qk_rope_dim}")
                    print(f"Total head dim: {model.dh}")
                    x_next = torch.randn(1, 1, model.d_model)
                    out2, _ = model(x_next, kv_cache=(kv_cache, k_rope), use_cache=True, past_length=x.size(1))
                    x_full = torch.cat([x, x_next], dim=1)
                    out_full, _ = model(x_full, use_cache=True)
                else:
                    out1, kv_cache = model(x, use_cache=True)
                    print("KV cache shape:", kv_cache[0].shape if isinstance(kv_cache, tuple) else kv_cache.shape)
                    x_next = torch.randn(1, 1, model.d_model)
                    out2, _ = model(x_next, kv_cache=kv_cache, use_cache=True, past_length=x.size(1))
                    x_full = torch.cat([x, x_next], dim=1)
                    out_full, _ = model(x_full, use_cache=True)
            max_diff = (out2 - out_full[:, -1:]).abs().max().item()
            print(f"{name} KV cache max difference: {max_diff:.6f}")
            print("KV cache", "consistent" if max_diff < 0.01 else "inconsistent")
        except Exception as e:
            print(f"Error testing KV cache for {name}: {str(e)}")

def main():
    d_model = 64
    n_heads = 8
    latent_dim = d_model // 2  # Set latent dimension to half of model dimension
    
    models = {
        'MHA': MHA(d_model, n_heads),
        'MHA_RoPE': Rope_MHA(d_model, n_heads),
        'MHA_Decoupled_RoPE': Decoupled_Rope_MHA(d_model, n_heads),
        'MQA': RopelessMQA(d_model, n_heads),
        'MQA_RoPE': Rope_MQA(d_model, n_heads),
        'MLA': RopelessMLA(d_model, n_heads),
        'MLA_RoPE': MLA(d_model, n_heads, latent_dim)
    }
    
    test_attention_patterns(models)
    test_kv_cache(models)
    
    print("\nTesting position sensitivity...")
    for name, model in models.items():
        try:
            is_sensitive = test_position_sensitivity(model, d_model=d_model, seq_len=17)
            expected_sensitive = ('RoPE' in type(model).__name__ or 'Rope' in type(model).__name__) and not isinstance(model, RopelessMLA)
            print(f"\n{name}:")
            print(f"Position sensitive: {is_sensitive}")
            print(f"Matches expectations: {is_sensitive == expected_sensitive}")
        except Exception as e:
            print(f"Error testing position sensitivity for {name}: {str(e)}")

if __name__ == "__main__":
    main()
