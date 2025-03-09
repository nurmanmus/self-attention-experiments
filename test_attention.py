import torch
import matplotlib.pyplot as plt
from modeling.attention.mha import MHA, Rope_MHA
from modeling.attention.mqa import RopelessMQA, Rope_MQA
from modeling.attention.mla import MLA, RopelessMLA

def get_attention_patterns(model, seq_len=16, d_model=64):
    """Extract attention patterns from a single forward pass."""
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
            Q = x @ module.wq  # Shape: [B, S, D]
            KV = x @ module.wkv  # Shape: [B, S, 2D]
            K, _ = torch.chunk(KV, 2, -1)  # K shape: [B, S, D]
            q_heads = Q.view(Q.shape[0], Q.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
            k_heads = K.view(K.shape[0], K.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
        elif isinstance(module, (MHA, Rope_MHA)):
            # MHA case
            QKV = x @ module.qkv  # Shape: [B, S, 3D]
            qkv = QKV.reshape(QKV.shape[0], QKV.shape[1], 3, module.n_heads, head_dim)
            q, k, v = qkv.unbind(dim=2)  # Each shape: [B, S, H, D/H]
            q_heads = q.transpose(1, 2)  # [B, H, S, D/H]
            k_heads = k.transpose(1, 2)  # [B, H, S, D/H]
        elif isinstance(module, (RopelessMLA, MLA)):
            # MLA case - no changes needed as it's working correctly
            compressed_q = x @ module.W_dq  # [B, S, q_proj_dim]
            Q = compressed_q @ module.W_uq  # [B, S, D]
            compressed_kv = x @ module.W_dkv  # [B, S, kv_proj_dim]
            if hasattr(module, 'W_ukv'):
                KV = compressed_kv @ module.W_ukv  # [B, S, 2D]
                K, _ = torch.chunk(KV, 2, -1)  # [B, S, D]
            else:
                K = module.kv_layernorm(compressed_kv)  # [B, S, D]
            q_heads = Q.view(Q.shape[0], Q.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
            k_heads = K.view(K.shape[0], K.shape[1], module.n_heads, head_dim).transpose(1, 2)  # [B, H, S, D/H]
        else:
            raise ValueError(f"Unknown attention type: {type(module)}")
        
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
    return max_diff

def test_position_sensitivity(model, seq_len=16, d_model=64):
    """Test if RoPE models are position-sensitive while non-RoPE are not."""
    # Create two different sequences with the same content but different positions
    x1 = torch.randn(1, seq_len, d_model)
    x2 = torch.randn(1, seq_len, d_model)
    
    # Make the middle tokens identical in both sequences
    mid_start = seq_len // 4
    mid_end = 3 * seq_len // 4
    x2[:, mid_start:mid_end] = x1[:, mid_start:mid_end]
    
    # Process both sequences
    with torch.no_grad():
        output1, _ = model(x1)
        output2, _ = model(x2)
    
    # Compare outputs for the identical middle section
    # RoPE models should produce different outputs even for identical tokens
    # due to position-dependent encoding
    mid_output_diff = (output1[:, mid_start:mid_end] - output2[:, mid_start:mid_end]).abs().max().item()
    
    # Higher threshold for RoPE variants as they should show position sensitivity
    is_rope = isinstance(model, (Rope_MHA, Rope_MQA, MLA))
    threshold = 0.05 if is_rope else 0.01
    
    return mid_output_diff > threshold

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