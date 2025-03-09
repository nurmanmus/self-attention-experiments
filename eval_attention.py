import torch
import torch.nn as nn
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
import itertools
import random
from typing import Tuple, List, Dict
import math

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(attn_type):
    """Initialize model with specified attention type."""
    from modeling.gpt import GPTModel
    
    model = GPTModel(
        d_model=32,  # Reduced model size for testing
        n_heads=8,   # Reduced number of heads
        layers=4,    # Reduced number of layers
        vocab_size=50257,  # GPT-2 vocab size
        max_seq_len=1024,
        use_mla=attn_type == 'mla',
        use_mqa=attn_type == 'mqa'
    )
    
    # Try to load pre-trained weights if available
    try:
        model.load_state_dict(torch.load(f'weights/{attn_type}_model.pt', weights_only=True))
        return model
    except:
        return model

def load_test_data(tokenizer, sequence_length: int = 1024, num_samples: int = 1) -> Tuple[torch.Tensor, List[str]]:
    """Load and prepare test data from multiple sources."""
    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Load datasets
    try:
        # Load WikiText dataset
        wikitext = load_dataset("wikitext", 
                              "wikitext-103-v1", 
                              split="test",
                              trust_remote_code=True)
        
        # Get samples from dataset
        text_samples = []
        for text in wikitext["text"]:
            if text and isinstance(text, str) and len(text.strip()) > 0:
                text_samples.append(("wiki", text.strip()))
                if len(text_samples) >= num_samples:
                    break
        
        # If we don't have enough samples, add some random text
        while len(text_samples) < num_samples:
            text_samples.append(("random", "Random generated text for testing attention patterns."))
        
        # Shuffle and select required number of samples
        random.shuffle(text_samples)
        selected_samples = text_samples[:num_samples]
        
        # Tokenize and prepare tensors
        tokenized_texts = []
        source_texts = []
        
        for source, text in selected_samples:
            # Tokenize with truncation and padding
            tokens = tokenizer(text, 
                             truncation=True,
                             max_length=sequence_length,
                             padding="max_length",
                             return_tensors="pt")
            
            # Get input_ids and ensure it's 2D
            input_ids = tokens["input_ids"]
            if len(input_ids.shape) == 3:  # If shape is (batch, seq_len, hidden)
                input_ids = input_ids.view(-1, sequence_length)
            elif len(input_ids.shape) == 1:  # If shape is (seq_len,)
                input_ids = input_ids.unsqueeze(0)  # Add batch dimension
                
            tokenized_texts.append(input_ids)
            source_texts.append((source, text[:100] + "..." if len(text) > 100 else text))
        
        # Stack tensors
        if tokenized_texts:
            input_tensor = torch.cat(tokenized_texts, dim=0)
            # Ensure final shape is (batch_size, sequence_length)
            if len(input_tensor.shape) > 2:
                input_tensor = input_tensor.view(-1, sequence_length)
        else:
            print("Warning: No valid samples found in the dataset, using random data")
            input_tensor = torch.randint(0, tokenizer.vocab_size, (num_samples, sequence_length))
            
        return input_tensor, source_texts
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Falling back to random data generation")
        # Fallback to simple random data if dataset loading fails
        input_tensor = torch.randint(0, tokenizer.vocab_size, (num_samples, sequence_length))
        return input_tensor, [("random", "Random text") for _ in range(num_samples)]

def measure_memory_usage(model, device, tokenizer, input_size=(1, 512)):
    """Measure memory usage of the model."""
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
    # Ensure input is 2D (batch_size, sequence_length)
    if len(x.shape) > 2:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
    with torch.no_grad():
        outputs = model(x)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = (final_memory - initial_memory) / 1024 / 1024  # Convert to MB
    return memory_used

def measure_inference_speed(model, device, tokenizer, input_size=(1, 512), num_runs=100):
    """Measure inference speed of the model."""
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
    # Ensure input is 2D (batch_size, sequence_length)
    if len(x.shape) > 2:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(x)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def analyze_attention_patterns(model, device, tokenizer, sequence_length=512, num_samples=5):
    """Analyze and visualize attention patterns."""
    model.eval()
    x, source_texts = load_test_data(tokenizer, sequence_length=sequence_length, num_samples=num_samples)
    x = x.to(device)
    
    # Ensure input is 2D (batch_size, sequence_length)
    if len(x.shape) > 2:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
    # Create a hook to capture attention weights
    attention_weights = []
    
    def attention_hook(module, inputs, outputs):
        try:
            # Extract q, k, v from the module's forward pass
            if hasattr(module, 'qkv'):
                # For MHA
                B, S, D = inputs[0].shape
                QKV = inputs[0] @ module.qkv.T
                Q, K, V = torch.chunk(QKV, 3, -1)
                dh = D // module.n_heads
                q_heads = Q.view(B, S, module.n_heads, dh).transpose(1, 2)
                k_heads = K.view(B, S, module.n_heads, dh).transpose(1, 2)
                
            elif hasattr(module, 'wq') and hasattr(module, 'wkv'):
                # For MQA
                B, S, D = inputs[0].shape
                Q = inputs[0] @ module.wq.T
                KV = inputs[0] @ module.wkv
                K, V = torch.chunk(KV, 2, -1)
                dh = D // module.n_heads
                q_heads = Q.view(B, S, module.n_heads, dh).transpose(1, 2)
                k_heads = K.unsqueeze(2).expand(B, -1, module.n_heads, -1).view(B, -1, module.n_heads, dh).transpose(1, 2)
                
            elif hasattr(module, 'W_dq') and hasattr(module, 'W_dkv'):
                # For MLA
                B, S, D = inputs[0].shape
                compressed_q = inputs[0] @ module.W_dq
                compressed_q = module.q_layernorm(compressed_q)
                Q = compressed_q @ module.W_uq
                compressed_kv = inputs[0] @ module.W_dkv
                compressed_kv = module.kv_layernorm(compressed_kv)
                KV = compressed_kv @ module.W_ukv
                K, V = torch.split(KV, module.d_model, dim=-1)
                q_heads = Q.view(B, -1, module.n_heads, module.dh).transpose(1, 2)
                k_heads = K.view(B, -1, module.n_heads, module.dh).transpose(1, 2)
            
            # Compute attention scores
            d_k = q_heads.size(-1)
            scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Get attention weights through softmax
            weights = torch.softmax(scores, dim=-1)
            attention_weights.append(weights.detach().cpu())
            
        except Exception as e:
            print(f"Error in attention hook: {str(e)}")
            print(f"Module type: {type(module)}")
            print(f"Input shapes: {[x.shape for x in inputs]}")
            if isinstance(outputs, tuple):
                print(f"Output shapes: {[x.shape for x in outputs]}")
            else:
                print(f"Output shape: {outputs.shape}")
    
    # Register hooks for each attention layer
    hooks = []
    for name, module in model.named_modules():
        # Only hook the main attention modules, not their sub-components
        if name.endswith('.mha') and isinstance(module, (
            model.layers[0].mha.__class__,  # Use the type of first layer's attention module
        )):
            print(f"Registering hook for {name}: {type(module)}")
            hook = module.register_forward_hook(attention_hook)
            hooks.append(hook)
    
    with torch.no_grad():
        try:
            # Forward pass to get attention weights
            outputs = model(x)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            if not attention_weights:
                print("Warning: No attention weights captured during forward pass")
                print("\nModel architecture:")
                for name, module in model.named_modules():
                    print(f"- {name}: {type(module)}")
                return None
                
            # Verify attention weights format
            print(f"\nNumber of attention layers: {len(attention_weights)}")
            for i, layer_weights in enumerate(attention_weights):
                print(f"Layer {i} attention weights shape: {layer_weights.shape}")
                
        except Exception as e:
            print(f"Error during forward pass: {str(e)}")
            print("\nModel architecture:")
            for name, module in model.named_modules():
                print(f"- {name}: {type(module)}")
            raise e
    
    # Create directory for figures if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Analyze patterns for each sample
    for sample_idx in range(min(num_samples, len(source_texts))):
        source, text = source_texts[sample_idx]
        
        # Plot attention patterns for each layer
        for layer_idx, layer_attention in enumerate(attention_weights):
            # Ensure layer_attention is the right shape (batch, heads, seq, seq)
            if len(layer_attention.shape) == 4:
                layer_weights = layer_attention[sample_idx].cpu()
            else:
                print(f"Warning: Unexpected attention weight shape: {layer_attention.shape}")
                continue
            
            # Average across heads for visualization
            avg_attention = layer_weights.mean(dim=0).numpy()
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(avg_attention, cmap='viridis')
            plt.title(f'Attention Pattern - {source}\nLayer {layer_idx + 1}')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            
            # Add sample text as subtitle
            plt.figtext(0.5, -0.1, f"Sample text: {text}", 
                       wrap=True, horizontalalignment='center', fontsize=8)
            
            # Save figure
            plt.savefig(f'figures/attention_pattern_sample{sample_idx}_layer{layer_idx+1}.png',
                       bbox_inches='tight', dpi=300)
            plt.close()
        
        # Calculate and plot head importance
        head_importance = torch.zeros(len(attention_weights), attention_weights[0].size(1))
        for layer_idx, layer_attention in enumerate(attention_weights):
            # Calculate importance as the mean attention weight for each head
            head_importance[layer_idx] = layer_attention[sample_idx].mean(dim=(1, 2))
        
        # Plot head importance heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(head_importance.numpy(), 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlOrRd')
        plt.title(f'Head Importance - {source}')
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.savefig(f'figures/head_importance_sample{sample_idx}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    return attention_weights

def evaluate_attention_mechanisms(num_samples=5):
    """Evaluate different attention mechanisms."""
    print("Using device:", device)
    
    # Initialize results dictionary
    results = {
        'memory_usage': {},
        'inference_speed': {},
        'attention_patterns': {}
    }
    
    # Test parameters - reduced sizes for testing
    sequence_lengths = [128, 256]  # Reduced from [512, 1024]
    batch_sizes = [1, 2]  # Reduced from [1, 4, 16]
    
    def display_metrics(attn_type, memory_results, speed_results):
        """Display performance metrics for the current attention mechanism."""
        print(f"\n{attn_type.upper()} Performance Metrics:")
        print("\nMemory Usage (MB):")
        print("-" * 50)
        print(f"{'Config':<15} {'Memory (MB)':<10}")
        print("-" * 50)
        for config, memory in memory_results.items():
            print(f"{config:<15} {memory:>10.2f}")
        
        print("\nInference Speed (seconds):")
        print("-" * 50)
        print(f"{'Config':<15} {'Time (s)':<10}")
        print("-" * 50)
        for config, speed in speed_results.items():
            print(f"{config:<15} {speed:>10.5f}")
        print("\n" + "="*50)
    
    # Evaluate each attention mechanism
    for attn_type in ['mha', 'mqa', 'mla']:
        print(f"\nEvaluating {attn_type.upper()}...")
        
        # Initialize model
        model = initialize_model(attn_type)
        if not model:
            print(f"No weights found for {attn_type.upper()}, using random initialization")
        model = model.to(device)
        model.eval()
        
        # Memory usage
        print("Measuring memory usage...")
        memory_results = {}
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                memory_used = measure_memory_usage(model, device, tokenizer, 
                                                input_size=(batch_size, seq_len))
                memory_results[f'b{batch_size}_s{seq_len}'] = memory_used
        results['memory_usage'][attn_type] = memory_results
        
        # Inference speed - reduced number of runs
        print("Measuring inference speed...")
        speed_results = {}
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                avg_time = measure_inference_speed(model, device, tokenizer,
                                                input_size=(batch_size, seq_len),
                                                num_runs=10)  # Reduced from 100
                speed_results[f'b{batch_size}_s{seq_len}'] = avg_time
        results['inference_speed'][attn_type] = speed_results
        
        # Display current metrics
        display_metrics(attn_type, memory_results, speed_results)
        
        # Attention patterns - reduced sequence length and samples
        print("Analyzing attention patterns...")
        attention_weights = analyze_attention_patterns(model, device, tokenizer,
                                                    sequence_length=256,  # Reduced from 1024
                                                    num_samples=2)  # Reduced from 5
        results['attention_patterns'][attn_type] = attention_weights
        
        # Clear GPU memory after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Display final comparative metrics
    print("\nComparative Performance Summary:")
    print("=" * 50)
    
    print("\nMemory Usage Comparison (MB):")
    print("-" * 70)
    print(f"{'Config':<15} {'MHA':>10} {'MQA':>10} {'MLA':>10}")
    print("-" * 70)
    for config in results['memory_usage']['mha'].keys():
        print(f"{config:<15}", end="")
        for attn_type in ['mha', 'mqa', 'mla']:
            print(f"{results['memory_usage'][attn_type][config]:>10.2f}", end="")
        print()
    
    print("\nInference Speed Comparison (seconds):")
    print("-" * 70)
    print(f"{'Config':<15} {'MHA':>10} {'MQA':>10} {'MLA':>10}")
    print("-" * 70)
    for config in results['inference_speed']['mha'].keys():
        print(f"{config:<15}", end="")
        for attn_type in ['mha', 'mqa', 'mla']:
            print(f"{results['inference_speed'][attn_type][config]:>10.5f}", end="")
        print()
    
    return results

def display_all_figures():
    """Display all generated figures in the notebook environment."""
    if not os.path.exists("figures"):
        print("No figures directory found.")
        return
        
    figure_files = sorted(os.listdir("figures"))
    if not figure_files:
        print("No figures found in the figures directory.")
        return
    
    # Import display tools
    try:
        from IPython.display import display, Image
        from matplotlib import pyplot as plt
    except ImportError:
        print("Could not import IPython display modules")
        return
    
    # Group figures by type
    grouped_figures = {}
    for filename in figure_files:
        if filename.endswith('.png'):
            if 'attention_pattern' in filename:
                group = 'Attention Patterns'
            elif 'head_importance' in filename:
                group = 'Head Importance'
            else:
                group = 'Other'
            
            if group not in grouped_figures:
                grouped_figures[group] = []
            grouped_figures[group].append(filename)
    
    # Display figures by group
    for group, files in grouped_figures.items():
        print(f"\n{group}:")
        
        try:
            # Calculate grid dimensions
            n_files = len(files)
            n_cols = min(3, n_files)  # Maximum 3 columns
            n_rows = (n_files + n_cols - 1) // n_cols  # Ceiling division
            
            # Create a figure with subplots
            plt.figure(figsize=(8*n_cols, 6*n_rows))
            
            for idx, filename in enumerate(files, 1):
                file_path = os.path.join("figures", filename)
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"- {filename} ({file_size:.1f} KB)")
                
                # Add subplot
                plt.subplot(n_rows, n_cols, idx)
                img = plt.imread(file_path)
                plt.imshow(img)
                plt.axis('off')
                plt.title(os.path.splitext(filename)[0], fontsize=8, pad=2)
            
            plt.tight_layout(pad=3.0)
            plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error displaying {group}: {str(e)}")
            # Fallback to individual image display
            for filename in files:
                file_path = os.path.join("figures", filename)
                try:
                    img = plt.imread(file_path)
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(os.path.splitext(filename)[0])
                    plt.show()
                    plt.close()
                except Exception as e:
                    print(f"Error displaying {filename}: {str(e)}")
                    continue

if __name__ == "__main__":
    print("Using device:", device)
    print("Loading tokenizer...")
    
    # Initialize tokenizer with padding
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    
    # Run evaluation
    try:
        results = evaluate_attention_mechanisms(num_samples=5)
        
        # Display figures if in notebook environment
        display_all_figures()
    except Exception as e:
        print(f"Error during evaluation: {str(e)}") 