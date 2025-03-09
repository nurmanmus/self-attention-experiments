import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from modeling.gpt import GPTModel
from torch.amp import autocast
import psutil
import os

def measure_memory_usage(model, device, input_size=(1, 512)):
    """Measure peak memory usage of the model during forward and backward pass"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Generate random token indices as input
    x = torch.randint(0, model.vocab_size, input_size).to(device)
    
    # Forward pass
    with autocast('cuda'):
        start_mem = torch.cuda.memory_allocated()
        out, _ = model(x)
        forward_mem = torch.cuda.memory_allocated() - start_mem
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        backward_mem = torch.cuda.memory_allocated() - forward_mem - start_mem
    
    return {
        'forward_memory_mb': forward_mem / 1024**2,
        'backward_memory_mb': backward_mem / 1024**2,
        'total_memory_mb': (forward_mem + backward_mem) / 1024**2
    }

def measure_inference_speed(model, device, input_size=(1, 512), n_runs=100):
    """Measure average inference time"""
    x = torch.randint(0, model.vocab_size, input_size).to(device)
    times = []
    
    # Warmup
    for _ in range(10):
        with torch.no_grad(), autocast('cuda'):
            model(x)
    
    # Measure
    for _ in range(n_runs):
        start_time = time.time()
        with torch.no_grad(), autocast('cuda'):
            model(x)
        times.append(time.time() - start_time)
    
    return {
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000
    }

def analyze_attention_patterns(model, device, input_size=(1, 512)):
    """Analyze attention patterns for each mechanism"""
    x = torch.randint(0, model.vocab_size, input_size).to(device)
    
    # Get attention weights
    model.eval()
    with torch.no_grad(), autocast('cuda'):
        _, cache = model(x)
    
    # Extract attention weights from the last layer
    if hasattr(cache, 'attn_weights'):
        attn_weights = cache.attn_weights[-1]  # Last layer
        
        # Compute statistics
        sparsity = (attn_weights < 0.01).float().mean().item()
        entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(-1).mean().item()
        
        return {
            'sparsity': sparsity,
            'entropy': entropy,
            'max_attention': attn_weights.max().item(),
            'mean_attention': attn_weights.mean().item()
        }
    else:
        return None

def plot_comparison(results, metric, title):
    """Plot comparison of different attention mechanisms"""
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    
    plt.bar(models, values)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./figures/{metric.lower().replace(" ", "_")}_comparison.png')
    plt.close()

def evaluate_attention_mechanisms(sequence_length=512, d_model=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model configurations
    configs = {
        "MHA": {"use_mla": False, "use_mqa": False},
        "MQA": {"use_mla": False, "use_mqa": True},
        "MLA": {"use_mla": True, "use_mqa": False}
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nEvaluating {name}...")
        
        # Initialize model
        model = GPTModel(
            d_model=d_model,
            n_heads=16,
            layers=8,
            vocab_size=10000,
            max_seq_len=1024,
            **config
        ).to(device)
        
        # Load weights if available
        try:
            model.load_state_dict(torch.load(f'./weights/{name.lower()}_model_weights.pt', weights_only=True))
            print(f"Loaded weights for {name}")
        except:
            print(f"No weights found for {name}, using random initialization")
        
        model.eval()
        
        # Run evaluations
        results[name] = {}
        
        # 1. Memory Usage
        print("Measuring memory usage...")
        results[name].update(
            measure_memory_usage(model, device, input_size=(1, sequence_length))
        )
        
        # 2. Inference Speed
        print("Measuring inference speed...")
        results[name].update(
            measure_inference_speed(model, device, input_size=(1, sequence_length))
        )
        
        # 3. Attention Patterns
        print("Analyzing attention patterns...")
        attn_stats = analyze_attention_patterns(model, device, input_size=(1, sequence_length))
        if attn_stats:
            results[name].update(attn_stats)
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Plot comparisons
    metrics = ['total_memory_mb', 'avg_time_ms', 'sparsity', 'entropy']
    titles = ['Memory Usage (MB)', 'Inference Time (ms)', 'Attention Sparsity', 'Attention Entropy']
    
    for metric, title in zip(metrics, titles):
        if all(metric in results[m] for m in results):
            plot_comparison(results, metric, title)
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Run evaluation
    results = evaluate_attention_mechanisms()
    
    # Save results
    with open('attention_evaluation_results.txt', 'w') as f:
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n") 