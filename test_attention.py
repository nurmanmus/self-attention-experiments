import torch
import matplotlib.pyplot as plt
from modeling.attention.mha import MHA, Rope_MHA, Decoupled_Rope_MHA
from modeling.attention.mqa import RopelessMQA, Rope_MQA
from modeling.attention.mla import MLA, RopelessMLA
from modeling.attention.utils import apply_rope_x

def get_attention_patterns(model, seq_len=16, d_model=64):
    """Extract attention patterns from a single forward pass.
    
    Args:
        model: The attention module to test
        seq_len: Length of the test sequence
        d_model: Model dimension
        
    Returns:
        torch.Tensor: Attention patterns for each head [batch, n_heads, seq_len, seq_len]
    """
    # Create a simple input sequence
    x = torch.randn(1, seq_len, d_model)
    
    # Register a hook to capture attention weights
    attention_patterns = []
    def hook_fn(module, input, output):
        # Get attention weights from the output
        # The output should be (attn_output, kv_cache)
        # We need to compute attention weights from the module's Q and K
        head_dim = d_model // module.n_heads  # Define head_dim here for all variants
        
        if isinstance(module, (RopelessMQA, Rope_MQA)):
            # MQA case
            Q = x @ module.wq.T  # Shape: [B, S, D]
            KV = x @ module.wkv  # Shape: [B, S, 2*Dh]
            K, _ = torch.chunk(KV, 2, -1)  # K shape: [B, S, Dh]
            # Expand K for each head
            K_expand = K.unsqueeze(2).expand(-1, -1, module.n_heads, -1)  # [B, S, H, Dh]
            # Reshape Q into heads
            q_heads = Q.view(Q.shape[0], Q.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
            k_heads = K_expand.transpose(1, 2)  # [B, H, S, Dh]
        elif isinstance(module, (MHA, Rope_MHA)):
            # MHA case
            QKV = x @ module.qkv.T  # Shape: [B, S, 3D]
            Q, K, V = torch.chunk(QKV, 3, -1)  # Each shape: [B, S, D]
            # Split into heads
            q_heads = Q.view(Q.shape[0], Q.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
            k_heads = K.view(K.shape[0], K.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
        elif isinstance(module, RopelessMLA):
            # RopelessMLA case
            compressed_q = x @ module.W_dq  # [B, S, q_proj_dim]
            compressed_q = module.q_layernorm(compressed_q)
            Q = compressed_q @ module.W_uq  # [B, S, D]
            
            compressed_kv = x @ module.W_dkv  # [B, S, kv_proj_dim]
            compressed_kv = module.kv_layernorm(compressed_kv)
            KV = compressed_kv @ module.W_ukv  # [B, S, 2D]
            K, _ = torch.split(KV, module.d_model, dim=-1)  # [B, S, D]
            
            q_heads = Q.view(Q.shape[0], Q.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
            k_heads = K.view(K.shape[0], K.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
        elif isinstance(module, MLA):
            # MLA case with decoupled RoPE
            B, S, D = x.shape
            
            # Q projections with RoPE
            compressed_q = x @ module.W_dq  # [B, S, q_proj_dim]
            compressed_q = module.q_layernorm(compressed_q)
            Q = compressed_q @ module.W_uq  # [B, S, D]
            Q = Q.view(B, S, module.n_heads, module.dh).transpose(1, 2)  # [B, H, S, D/H]
            Q, Q_for_rope = torch.split(Q, [module.qk_nope_dim, module.qk_rope_dim], dim=-1)
            
            # Apply RoPE to Q
            cos_q = module.cos_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            sin_q = module.sin_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)
            
            # KV projections with RoPE
            compressed_kv = x @ module.W_dkv  # [B, S, kv_proj_dim + rope_dim]
            KV_for_lora, K_for_rope = torch.split(compressed_kv, [module.kv_proj_dim, module.qk_rope_dim], dim=-1)
            KV_for_lora = module.kv_layernorm(KV_for_lora)
            
            # Project and split KV
            KV = KV_for_lora @ module.W_ukv  # [B, S, D + H*nope_dim]
            KV = KV.view(B, S, module.n_heads, module.dh + module.qk_nope_dim).transpose(1, 2)
            K, V = torch.split(KV, [module.qk_nope_dim, module.dh], dim=-1)
            
            # Apply RoPE to K
            K_for_rope = K_for_rope.view(B, S, 1, module.qk_rope_dim).transpose(1, 2)
            cos_k = module.cos_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            sin_k = module.sin_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
            K_for_rope = K_for_rope.repeat(1, module.n_heads, 1, 1)
            
            # Combine Q and K heads
            q_heads = torch.cat([Q, Q_for_rope], dim=-1)  # [B, H, S, D/H]
            k_heads = torch.cat([K, K_for_rope], dim=-1)  # [B, H, S, D/H]
        elif isinstance(module, Decoupled_Rope_MHA):
            # Decoupled RoPE MHA case
            B, S = x.shape[:2]
            QV = x @ module.qkv.T  # [B, S, 2D]
            K = x @ module.wk.T  # [B, S, n_heads*nope_dim + rope_dim]
            
            Q, V = torch.chunk(QV, 2, -1)  # Q: [B, S, D], V: [B, S, D]
            Q = Q.view(B, S, module.n_heads, module.dh).transpose(1,2)  # [B, H, S, D/H]
            
            # Split Q into RoPE and non-RoPE parts
            Q, Q_for_rope = torch.split(Q, [module.qk_nope_dim, module.qk_rope_dim], dim=-1)
            
            # Split K into RoPE and non-RoPE parts
            K, K_for_rope = torch.split(K, [module.n_heads * module.qk_nope_dim, module.qk_rope_dim], dim=-1)
            K = K.view(B, S, module.n_heads, module.qk_nope_dim).transpose(1,2)
            K_for_rope = K_for_rope.view(B, S, 1, module.qk_rope_dim).transpose(1,2)
            
            # Apply RoPE
            cos = module.cos_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            sin = module.sin_cached[:, :, :S, :module.qk_rope_dim//2].repeat(1, 1, 1, 2)
            Q_for_rope = apply_rope_x(Q_for_rope, cos, sin)
            K_for_rope = apply_rope_x(K_for_rope, cos, sin)
            K_for_rope = K_for_rope.repeat(1, module.n_heads, 1, 1)
            
            # Combine Q and K parts
            q_heads = torch.cat([Q, Q_for_rope], dim=-1)
            k_heads = torch.cat([K, K_for_rope], dim=-1)
        else:
            raise ValueError(f"Unknown attention type: {type(module)}")
        
        # Print shapes for debugging
        print(f"\nDebug shapes for {type(module).__name__}:")
        print(f"q_heads shape: {q_heads.shape}")
        print(f"k_heads shape: {k_heads.shape}")
        
        # Compute attention weights with proper scaling
        attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, S, S]
        attention_patterns.append(attn_weights.detach())
    
    # Register hook directly on the model since it is the attention module
    model.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(x)
    
    if not attention_patterns:
        raise ValueError(f"No attention patterns captured for {type(model).__name__}")
    
    return attention_patterns[0]

def test_kv_cache(model, seq_len=16, d_model=64):
    """Test KV cache consistency between sequential and parallel processing."""
    x = torch.randn(1, seq_len, d_model)
    
    # Process full sequence at once
    with torch.no_grad():
        full_output, _ = model(x)
    
    # Process sequence sequentially with caching
    cached_outputs = []
    kv_cache = None
    with torch.no_grad():
        for i in range(seq_len):
            output, kv_cache = model(x[:, i:i+1, :], kv_cache=kv_cache, past_length=i)
            cached_outputs.append(output)
    
    sequential_output = torch.cat(cached_outputs, dim=1)
    
    # Compare outputs
    max_diff = (full_output - sequential_output).abs().max().item()
    
    # Print debug info for MLA variants
    if isinstance(model, (RopelessMLA, MLA)) and kv_cache is not None:
        print(f"\nDebug info for {type(model).__name__}:")
        print(f"KV cache shape: {kv_cache.shape}")
        if isinstance(model, MLA):
            # For MLA, we expect compressed KV with RoPE dimensions
            expected_shape = (1, seq_len, model.kv_proj_dim + model.qk_rope_dim)
            print(f"Expected shape: {expected_shape}")
            assert kv_cache.shape == expected_shape, f"KV cache shape mismatch: {kv_cache.shape} != {expected_shape}"
        else:
            # For RopelessMLA, we expect compressed KV
            expected_shape = (1, seq_len, model.kv_proj_dim)
            print(f"Expected shape: {expected_shape}")
            assert kv_cache.shape == expected_shape, f"KV cache shape mismatch: {kv_cache.shape} != {expected_shape}"
    
    print(f"{type(model).__name__} KV cache max difference: {max_diff:.6f}")
    print(f"KV cache {'consistent' if max_diff < 1e-5 else 'inconsistent'}")
    
    return max_diff

def test_position_sensitivity(model, seq_len=16, d_model=64):
    """Test if RoPE models are position-sensitive while non-RoPE are not."""
    # Create two different sequences with the same content but different positions
    x1 = torch.randn(1, seq_len, d_model)
    x2 = torch.clone(x1)  # Use clone to ensure identical content
    
    # Shift the sequence by 2 positions
    shift = 2
    x2 = torch.roll(x2, shifts=shift, dims=1)
    
    # Process both sequences
    with torch.no_grad():
        output1, _ = model(x1)
        output2, _ = model(x2)
    
    # Compare outputs for the non-shifted portion
    # RoPE models should produce different outputs even for identical tokens
    # due to position-dependent encoding
    valid_range = slice(shift, -shift) if shift > 0 else slice(None)
    mid_output_diff = (output1[:, valid_range] - output2[:, valid_range]).abs().mean().item()
    
    # Higher threshold for RoPE variants as they should show position sensitivity
    is_rope = isinstance(model, (Rope_MHA, Rope_MQA, MLA))
    threshold = 0.01  # Lower threshold since we're using mean difference
    
    is_sensitive = mid_output_diff > threshold
    expected_sensitive = is_rope
    
    print(f"\n{type(model).__name__}:")
    print(f"Position sensitivity score: {mid_output_diff:.6f}")
    print(f"Position sensitive: {is_sensitive}")
    print(f"Expected to be position sensitive: {expected_sensitive}")
    print(f"Matches expectations: {is_sensitive == expected_sensitive}")
    
    return is_sensitive

def plot_attention_patterns(patterns, title):
    """Plot attention patterns for visualization."""
    patterns = patterns.cpu()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes.flat):
        if i < patterns.shape[1]:  # number of heads
            im = ax.imshow(patterns[0, i])
            ax.set_title(f'Head {i}')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig

def main():
    # Model configurations
    d_model = 64
    n_heads = 8
    seq_len = 16
    
    # Initialize models
    models = {
        'MHA': MHA(d_model, n_heads),
        'MHA_RoPE': Rope_MHA(d_model, n_heads),
        'MHA_Decoupled_RoPE': Decoupled_Rope_MHA(d_model, n_heads),
        'MQA': RopelessMQA(d_model, n_heads),
        'MQA_RoPE': Rope_MQA(d_model, n_heads),
        'MLA': RopelessMLA(d_model, n_heads),
        'MLA_RoPE': MLA(d_model, n_heads)
    }
    
    print("Testing attention patterns...")
    for name, model in models.items():
        print(f"\nAnalyzing {name}...")
        try:
            patterns = get_attention_patterns(model, seq_len, d_model)
            print(f"Attention pattern shape: {patterns.shape}")
            
            # Plot patterns
            fig = plot_attention_patterns(patterns, name)
            plt.savefig(f'attention_patterns_{name.lower()}.png')
            plt.close()
            
            # Additional analysis
            if 'MQA' in name:
                # Check if patterns are shared across heads
                head_similarity = torch.corrcoef(patterns[0].reshape(n_heads, -1))
                print(f"Average head similarity: {head_similarity.mean().item():.4f}")
            elif 'MLA' in name:
                # Check rank of attention patterns
                U, S, V = torch.svd(patterns[0].reshape(n_heads, -1))
                effective_rank = (S > 1e-5).sum().item()
                print(f"Effective rank: {effective_rank}")
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
    
    print("\nTesting KV cache consistency...")
    for name, model in models.items():
        try:
            max_diff = test_kv_cache(model, seq_len, d_model)
            print(f"{name} KV cache max difference: {max_diff:.6f}")
            print(f"KV cache {'consistent' if max_diff < 1e-5 else 'inconsistent'}")
        except Exception as e:
            print(f"Error testing KV cache for {name}: {str(e)}")
    
    print("\nTesting position sensitivity...")
    for name, model in models.items():
        try:
            is_sensitive = test_position_sensitivity(model, seq_len, d_model)
            expected_sensitive = 'RoPE' in name or name == 'MLA'  # MLA uses RoPE by default
            
            print(f"\n{name}:")
            print(f"Position sensitive: {is_sensitive}")
            print(f"Expected to be position sensitive: {expected_sensitive}")
            print(f"Matches expectations: {is_sensitive == expected_sensitive}")
        except Exception as e:
            print(f"Error testing position sensitivity for {name}: {str(e)}")

if __name__ == '__main__':
    main() 