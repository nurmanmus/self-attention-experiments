import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from modeling.gpt import GPTModel
from torch.amp import autocast
import psutil
import os
from transformers import AutoTokenizer
import seaborn as sns
from datasets import load_dataset, concatenate_datasets
from itertools import groupby
from collections import Counter

def load_test_data(tokenizer, sequence_length=512, num_samples=1000):
    """Load and prepare test data from multiple datasets"""
    datasets = []
    
    # 1. WikiText-103 (larger than WikiText-2)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    datasets.append(wiki)
    
    # 2. C4 dataset (web text)
    c4 = load_dataset("c4", "en", split="validation", streaming=True)
    c4_samples = list(itertools.islice(c4, 1000))  # Get 1000 samples
    c4_dataset = Dataset.from_dict({
        "text": [sample["text"] for sample in c4_samples]
    })
    datasets.append(c4_dataset)
    
    # 3. BookCorpus
    books = load_dataset("bookcorpus", split="train", streaming=True)
    book_samples = list(itertools.islice(books, 1000))  # Get 1000 samples
    books_dataset = Dataset.from_dict({
        "text": [sample["text"] for sample in book_samples]
    })
    datasets.append(books_dataset)
    
    # Combine datasets
    combined = concatenate_datasets(datasets)
    
    # Process text in chunks
    chunks = []
    texts = []
    sources = []  # Track which dataset each sample came from
    
    for idx, text_sample in enumerate(combined["text"]):
        if not text_sample.strip():
            continue
            
        # Tokenize each text sample separately
        tokens = tokenizer(text_sample, 
                         truncation=True,
                         max_length=sequence_length,
                         return_tensors="pt")["input_ids"][0]
        
        if len(tokens) == sequence_length:
            chunks.append(tokens)
            texts.append(text_sample)
            # Determine source based on index
            if idx < len(wiki):
                sources.append("WikiText-103")
            elif idx < len(wiki) + len(c4_samples):
                sources.append("C4")
            else:
                sources.append("BookCorpus")
            
        if len(chunks) >= num_samples:
            break
    
    # If we don't have enough full-length sequences, pad the last ones
    while len(chunks) < num_samples:
        if chunks:
            last_tokens = chunks[-1]
            padded = torch.nn.functional.pad(last_tokens, (0, sequence_length - len(last_tokens)))
            chunks.append(padded)
            texts.append(texts[-1])
            sources.append(sources[-1])
        else:
            chunks.append(torch.zeros(sequence_length, dtype=torch.long))
            texts.append("")
            sources.append("padding")
    
    return torch.stack(chunks[:num_samples]), texts[:num_samples], sources[:num_samples]

def measure_memory_usage(model, device, tokenizer, input_size=(1, 512)):
    """Measure peak memory usage of the model during forward and backward pass"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load real text data
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
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

def measure_inference_speed(model, device, tokenizer, input_size=(1, 512), n_runs=100):
    """Measure average inference time"""
    # Load real text data
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
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

def analyze_attention_patterns(model, device, tokenizer, input_size=(1, 512)):
    """Analyze attention patterns for each mechanism"""
    # Load real text data
    x, texts, sources = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
    # Get attention weights
    model.eval()
    with torch.no_grad(), autocast('cuda'):
        _, cache = model(x)
    
    # Extract attention weights from all layers
    if hasattr(cache, 'attn_weights'):
        results = {}
        
        # Analyze attention patterns across all layers
        all_layers_weights = cache.attn_weights  # List of attention weights for each layer
        
        # 1. Global Statistics
        avg_attn_weights = torch.stack(all_layers_weights).mean(0)  # Average across layers
        results.update({
            'sparsity': (avg_attn_weights < 0.01).float().mean().item(),
            'entropy': -(avg_attn_weights * torch.log(avg_attn_weights + 1e-10)).sum(-1).mean().item(),
            'max_attention': avg_attn_weights.max().item(),
            'mean_attention': avg_attn_weights.mean().item()
        })
        
        # 2. Layer-wise Analysis
        layer_stats = []
        for layer_idx, layer_weights in enumerate(all_layers_weights):
            layer_stats.append({
                'layer': layer_idx,
                'sparsity': (layer_weights < 0.01).float().mean().item(),
                'entropy': -(layer_weights * torch.log(layer_weights + 1e-10)).sum(-1).mean().item(),
                'attention_concentration': (layer_weights.max(-1)[0]).mean().item()
            })
        results['layer_stats'] = layer_stats
        
        # 3. Head-wise Analysis
        head_importances = []
        for layer_weights in all_layers_weights:
            # Reshape to [batch, heads, seq, seq]
            head_weights = layer_weights.view(input_size[0], model.n_heads, input_size[1], input_size[1])
            # Compute head importance based on attention entropy
            head_entropy = -(head_weights * torch.log(head_weights + 1e-10)).sum([-2, -1]).mean(0)
            head_importances.append(head_entropy.cpu().numpy())
        results['head_importances'] = np.stack(head_importances)
        
        # 4. Enhanced Visualizations
        # 4.1 Attention heatmap with text context
        last_layer = all_layers_weights[-1].view(input_size[0], model.n_heads, input_size[1], input_size[1])
        head_entropy = -(last_layer * torch.log(last_layer + 1e-10)).sum([-2, -1]).mean(0)
        most_important_head = head_entropy.argmax().item()
        
        # Create multiple attention heatmaps for different data sources
        for idx, (source_type, group_indices) in enumerate(groupby(range(len(sources)), key=lambda i: sources[i])):
            group_indices = list(group_indices)
            if not group_indices:
                continue
                
            sample_idx = group_indices[0]  # Take first sample from each source
            attn_pattern = last_layer[sample_idx, most_important_head].cpu().numpy()
            
            plt.figure(figsize=(15, 10))
            sns.heatmap(attn_pattern, cmap='viridis')
            plt.title(f'Attention Pattern - {source_type}\n(Layer {len(all_layers_weights)-1}, Head {most_important_head})')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            # Add text sample as subtitle
            plt.figtext(0.5, -0.05, f'Sample text ({source_type}):\n{texts[sample_idx][:100]}...', 
                       wrap=True, horizontalalignment='center', fontsize=8)
            
            plt.savefig(f'./figures/attention_pattern_{source_type.lower()}.png', bbox_inches='tight')
            plt.close()
        
        # 4.2 Enhanced head importance heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(results['head_importances'], cmap='viridis', annot=True, fmt='.2f')
        plt.title('Head Importance Across Layers')
        plt.xlabel('Head Index')
        plt.ylabel('Layer')
        cbar = plt.colorbar()
        cbar.set_label('Importance Score (based on entropy)')
        plt.savefig('./figures/head_importance.png')
        plt.close()
        
        # 4.3 Enhanced layer-wise statistics
        metrics = ['sparsity', 'entropy', 'attention_concentration']
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, metric in enumerate(metrics):
            values = [stats[metric] for stats in layer_stats]
            sns.lineplot(x=range(len(values)), y=values, marker='o', ax=axes[idx])
            axes[idx].set_title(f'Layer-wise {metric.replace("_", " ").title()}')
            axes[idx].set_xlabel('Layer')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].grid(True)
            
            # Add value annotations
            for x, y in enumerate(values):
                axes[idx].annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.suptitle('Layer-wise Attention Statistics', y=1.05)
        plt.tight_layout()
        plt.savefig('./figures/layer_stats.png', bbox_inches='tight')
        plt.close()
        
        # 4.4 Dataset distribution plot
        plt.figure(figsize=(10, 6))
        source_counts = dict(Counter(sources))
        plt.bar(source_counts.keys(), source_counts.values())
        plt.title('Distribution of Samples Across Datasets')
        plt.xlabel('Dataset Source')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./figures/dataset_distribution.png')
        plt.close()
        
        return results
    else:
        return None

def plot_comparison(results, metric, title):
    """Plot comparison of different attention mechanisms"""
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    
    if metric in ['layer_stats', 'head_importances']:
        # Plot layer-wise or head-wise comparisons
        for model_name in models:
            if metric == 'layer_stats':
                values = [stats['entropy'] for stats in results[model_name][metric]]
                plt.plot(range(len(values)), values, label=model_name, marker='o')
            else:
                plt.imshow(results[model_name][metric], aspect='auto')
                plt.colorbar(label='Importance')
        plt.legend()
    else:
        # Plot simple bar comparison
        values = [results[m][metric] for m in models]
        plt.bar(models, values)
    
    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./figures/{metric.lower().replace(" ", "_")}_comparison.png')
    plt.close()

def evaluate_attention_mechanisms(sequence_length=512, d_model=512, vocab_size=50257, num_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize tokenizer (using GPT-2 tokenizer for compatibility)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
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
            vocab_size=vocab_size,
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
            measure_memory_usage(model, device, tokenizer, input_size=(1, sequence_length))
        )
        
        # 2. Inference Speed
        print("Measuring inference speed...")
        results[name].update(
            measure_inference_speed(model, device, tokenizer, input_size=(1, sequence_length))
        )
        
        # 3. Attention Patterns
        print("Analyzing attention patterns...")
        attn_stats = analyze_attention_patterns(model, device, tokenizer, input_size=(1, sequence_length))
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

def display_all_figures(base_path='./figures'):
    """Display all generated figures in the notebook"""
    from IPython.display import display, Image, HTML
    import glob
    
    # Group figures by type
    figure_groups = {
        'Attention Patterns': ['attention_pattern*.png'],
        'Head Analysis': ['head_importance.png'],
        'Layer Statistics': ['layer_stats.png'],
        'Dataset Analysis': ['dataset_distribution.png'],
        'Performance Metrics': ['*_mb_comparison.png', '*_ms_comparison.png'],
        'Attention Metrics': ['sparsity_comparison.png', 'entropy_comparison.png']
    }
    
    for group_name, patterns in figure_groups.items():
        matching_files = []
        for pattern in patterns:
            matching_files.extend(glob.glob(os.path.join(base_path, pattern)))
        
        if matching_files:
            display(HTML(f"<h2>{group_name}</h2>"))
            for file in sorted(matching_files):
                display(Image(filename=file))
                # Display file name and size
                size_kb = os.path.getsize(file) / 1024
                display(HTML(f"<p><i>{os.path.basename(file)} ({size_kb:.1f} KB)</i></p>"))

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Run evaluation
    results = evaluate_attention_mechanisms(num_samples=1000)  # Increased number of samples
    
    # Save results
    with open('attention_evaluation_results.txt', 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        f.write("Dataset Statistics:\n")
        f.write("-----------------\n")
        for source, count in Counter(sources).items():
            f.write(f"{source}: {count} samples\n")
        f.write("\nModel Results:\n")
        f.write("-------------\n")
        for model_name, metrics in results.items():
            f.write(f"\n{model_name}:\n")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")
    
    # Try to display figures if in a notebook environment
    try:
        display_all_figures()
    except NameError:
        print("Figures saved in ./figures directory") 