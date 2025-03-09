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

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_model(attn_type):
    """Initialize model with specified attention type."""
    from modeling.gpt import GPTModel
    
    model = GPTModel(
        d_model=512,
        n_heads=16,
        layers=8,
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
            
            tokenized_texts.append(tokens["input_ids"])
            source_texts.append((source, text[:100] + "..." if len(text) > 100 else text))
        
        # Stack tensors
        if tokenized_texts:
            input_tensor = torch.cat(tokenized_texts, dim=0)
        else:
            print("Warning: No valid samples found in the dataset, using random data")
            input_tensor = torch.randint(0, tokenizer.vocab_size, (num_samples, sequence_length))
            return input_tensor, [("random", "Random text") for _ in range(num_samples)]
            
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
    
    with torch.no_grad():
        _ = model(x)
    
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = (final_memory - initial_memory) / 1024 / 1024  # Convert to MB
    return memory_used

def measure_inference_speed(model, device, tokenizer, input_size=(1, 512), num_runs=100):
    """Measure inference speed of the model."""
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def analyze_attention_patterns(model, device, tokenizer, sequence_length=512, num_samples=5):
    """Analyze and visualize attention patterns."""
    model.eval()
    x, source_texts = load_test_data(tokenizer, sequence_length=sequence_length, num_samples=num_samples)
    x = x.to(device)
    
    with torch.no_grad():
        # Get attention weights from the model's forward pass
        outputs = model(x)
        # Assuming the model returns attention weights as part of its output
        if isinstance(outputs, tuple) and len(outputs) > 1:
            attention_weights = outputs[1]  # Second element should be attention weights
        else:
            print("Warning: Model did not return attention weights, skipping attention analysis")
            return None
    
    # Create directory for figures if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Analyze patterns for each sample
    for sample_idx in range(num_samples):
        source, text = source_texts[sample_idx]
        
        # Plot attention patterns for each layer
        for layer_idx, layer_attention in enumerate(attention_weights):
            layer_weights = layer_attention[sample_idx].cpu()
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(layer_weights[0].numpy(), cmap='viridis')
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

def evaluate_attention_mechanisms(num_samples=1000):
    """Evaluate different attention mechanisms."""
    print("Using device:", device)
    
    # Initialize results dictionary
    results = {
        'memory_usage': {},
        'inference_speed': {},
        'attention_patterns': {}
    }
    
    # Test parameters
    sequence_lengths = [512, 1024]
    batch_sizes = [1, 4, 16]
    
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
        
        # Inference speed
        print("Measuring inference speed...")
        speed_results = {}
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                avg_time = measure_inference_speed(model, device, tokenizer,
                                                input_size=(batch_size, seq_len))
                speed_results[f'b{batch_size}_s{seq_len}'] = avg_time
        results['inference_speed'][attn_type] = speed_results
        
        # Attention patterns
        print("Analyzing attention patterns...")
        attention_weights = analyze_attention_patterns(model, device, tokenizer,
                                                    sequence_length=1024,
                                                    num_samples=5)
        results['attention_patterns'][attn_type] = attention_weights
        
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
        for filename in files:
            file_path = os.path.join("figures", filename)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"- {filename} ({file_size:.1f} KB)")
            
            # Display the figure if in a notebook environment
            try:
                from IPython.display import Image, display
                display(Image(filename=file_path))
            except ImportError:
                print(f"  (Figure saved at: {file_path})")

if __name__ == "__main__":
    print("Using device:", device)
    print("Loading tokenizer...")
    
    # Initialize tokenizer with padding
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    
    # Run evaluation
    try:
        results = evaluate_attention_mechanisms(num_samples=1000)
        
        # Display figures if in notebook environment
        display_all_figures()
    except Exception as e:
        print(f"Error during evaluation: {str(e)}") 