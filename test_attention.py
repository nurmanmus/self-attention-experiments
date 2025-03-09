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
        # Get attention weights before softmax
        # For MHA/MQA/MLA, the module itself is the attention module
        q = input[0]  # B, H, S, D
        k = input[1]  # B, H, S, D
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
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
    x = torch.randn(1, seq_len, d_model)
    
    # Original sequence
    with torch.no_grad():
        orig_output, _ = model(x)
    
    # Shifted sequence
    shift = seq_len // 2
    x_shifted = torch.roll(x, shifts=shift, dims=1)
    with torch.no_grad():
        shifted_output, _ = model(x_shifted)
    
    # For RoPE models, outputs should differ
    # For non-RoPE models, outputs should be similar when shifted
    output_diff = (orig_output - torch.roll(shifted_output, shifts=-shift, dims=1)).abs().max().item()
    return output_diff

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
            diff = test_position_sensitivity(model, seq_len, d_model)
            expected_sensitive = 'RoPE' in name
            actual_sensitive = diff > 0.1
            
            print(f"\n{name}:")
            print(f"Position sensitivity: {diff:.6f}")
            print(f"Expected to be position sensitive: {expected_sensitive}")
            print(f"Actually position sensitive: {actual_sensitive}")
            print(f"Matches expectations: {expected_sensitive == actual_sensitive}")
        except Exception as e:
            print(f"Error testing position sensitivity for {name}: {str(e)}")

if __name__ == '__main__':
    main() 