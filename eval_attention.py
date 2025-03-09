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
import gc

# Set up device and memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Set memory allocation settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def initialize_model(attn_type):
    """Initialize model with specified attention type."""
    from modeling.gpt import GPTModel
    
    model = GPTModel(
        d_model=512,  # More realistic model size
        n_heads=8,    # Standard number of heads
        layers=12,    # Standard number of layers
        vocab_size=50257,  # GPT-2 vocab size
        max_seq_len=2048,  # Increased max sequence length
        use_mla=attn_type == 'mla',
        use_mqa=attn_type == 'mqa'
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Try to load pre-trained weights if available
    try:
        model.load_state_dict(torch.load(f'weights/{attn_type}_model.pt', weights_only=True))
        return model
    except:
        return model

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def load_test_data(tokenizer, sequence_length: int = 2048, num_samples: int = 1) -> Tuple[torch.Tensor, List[str]]:
    """Load and prepare test data from multiple sources."""
    try:
        # Load multiple datasets for more diverse testing
        datasets = []
        
        # Load WikiText-103
        wikitext = load_dataset("wikitext", 
                              "wikitext-103-v1", 
                              split="test",
                              trust_remote_code=True)
        datasets.append(("wiki", wikitext))
        
        # Load C4 dataset subset
        try:
            c4 = load_dataset("c4", "en", split="validation", streaming=True)
            c4_samples = list(itertools.islice(c4, 1000))  # Get 1000 samples
            datasets.append(("c4", c4_samples))
        except Exception as e:
            print(f"Error loading C4 dataset: {str(e)}")
        
        # Combine samples from all datasets
        text_samples = []
        for source, dataset in datasets:
            if isinstance(dataset, list):
                # For C4 dataset
                for item in dataset:
                    if len(text_samples) >= num_samples:
                        break
                    if item and 'text' in item and len(item['text'].strip()) > 100:
                        text_samples.append((source, item['text'].strip()))
            else:
                # For WikiText dataset
                for text in dataset["text"]:
                    if len(text_samples) >= num_samples:
                        break
                    if text and isinstance(text, str) and len(text.strip()) > 100:
                        text_samples.append((source, text.strip()))
        
        # If we still don't have enough samples, add some random text
        while len(text_samples) < num_samples:
            text_samples.append(("random", "Random generated text for testing performance metrics." * 100))
        
        # Shuffle and select required number of samples
        random.shuffle(text_samples)
        selected_samples = text_samples[:num_samples]
        
        # Tokenize and prepare tensors with longer sequences
        tokenized_texts = []
        source_texts = []
        
        for source, text in selected_samples:
            # Tokenize with truncation and padding
            tokens = tokenizer(text, 
                             truncation=True,
                             max_length=sequence_length,
                             padding="max_length",
                             return_tensors="pt")
            
            input_ids = tokens["input_ids"]
            if len(input_ids.shape) == 3:
                input_ids = input_ids.view(-1, sequence_length)
            elif len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)
                
            tokenized_texts.append(input_ids)
            source_texts.append((source, text[:100] + "..." if len(text) > 100 else text))
        
        # Stack tensors
        if tokenized_texts:
            input_tensor = torch.cat(tokenized_texts, dim=0)
            if len(input_tensor.shape) > 2:
                input_tensor = input_tensor.view(-1, sequence_length)
        else:
            print("Warning: No valid samples found in the datasets, using random data")
            input_tensor = torch.randint(0, tokenizer.vocab_size, (num_samples, sequence_length))
            
        return input_tensor, source_texts
        
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        print("Falling back to random data generation")
        input_tensor = torch.randint(0, tokenizer.vocab_size, (num_samples, sequence_length))
        return input_tensor, [("random", "Random text") for _ in range(num_samples)]

def measure_memory_usage(model, device, tokenizer, input_size=(1, 512)):
    """Measure memory usage of the model."""
    clear_gpu_memory()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        with torch.cuda.amp.autocast():  # Use mixed precision
            x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
            x = x.to(device)
            
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
        clear_gpu_memory()
        return memory_used
    except RuntimeError as e:
        print(f"Memory measurement failed: {str(e)}")
        clear_gpu_memory()
        return float('inf')

def measure_inference_speed(model, device, tokenizer, input_size=(1, 512), num_runs=100):
    """Measure inference speed of the model."""
    clear_gpu_memory()
    
    try:
        with torch.cuda.amp.autocast():  # Use mixed precision
            x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
            x = x.to(device)
            
            if len(x.shape) > 2:
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
            
            # Warmup with fewer runs
            with torch.no_grad():
                for _ in range(5):
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
            
            clear_gpu_memory()
            
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
        clear_gpu_memory()
        return avg_time
    except RuntimeError as e:
        print(f"Speed measurement failed: {str(e)}")
        clear_gpu_memory()
        return float('inf')

def measure_kqv_cache_performance(model, device, tokenizer, input_size=(1, 512), num_runs=10):
    """Measure performance metrics for KQV cache computation."""
    x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
    x = x.to(device)
    
    # Ensure input is 2D (batch_size, sequence_length)
    if len(x.shape) > 2:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
    
    # Dictionary to store performance metrics
    metrics = {
        'memory': 0,
        'time': 0,
        'cache_size': 0
    }
    
    def measure_kqv_hook(module, inputs, outputs):
        try:
            # Measure memory before computation
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Start timing
            start_time = time.time()
            
            # Compute KQV based on attention type
            if hasattr(module, 'qkv'):
                # For MHA
                B, S, D = inputs[0].shape
                QKV = inputs[0] @ module.qkv.T
                Q, K, V = torch.chunk(QKV, 3, -1)
                cache_size = K.numel() + V.numel()
                
            elif hasattr(module, 'wq') and hasattr(module, 'wkv'):
                # For MQA
                B, S, D = inputs[0].shape
                Q = inputs[0] @ module.wq.T
                KV = inputs[0] @ module.wkv
                K, V = torch.chunk(KV, 2, -1)
                cache_size = K.numel() + V.numel()
                
            elif hasattr(module, 'W_dq') and hasattr(module, 'W_dkv'):
                # For MLA
                B, S, D = inputs[0].shape
                compressed_kv = inputs[0] @ module.W_dkv
                compressed_kv = module.kv_layernorm(compressed_kv)
                KV = compressed_kv @ module.W_ukv
                K, V = torch.split(KV, module.d_model, dim=-1)
                cache_size = K.numel() + V.numel()
            
            # End timing
            end_time = time.time()
            
            # Measure memory after computation
            end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Update metrics
            metrics['memory'] += (end_mem - start_mem) / 1024 / 1024  # Convert to MB
            metrics['time'] += end_time - start_time
            metrics['cache_size'] = cache_size * 4 / 1024 / 1024  # Convert to MB (assuming float32)
            
        except Exception as e:
            print(f"Error in KQV measurement hook: {str(e)}")
    
    # Register hooks for each attention layer
    hooks = []
    for name, module in model.named_modules():
        if name.endswith('.mha') and isinstance(module, (
            model.layers[0].mha.__class__,
        )):
            hook = module.register_forward_hook(measure_kqv_hook)
            hooks.append(hook)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            outputs = model(x)
    
    # Reset metrics after warmup
    metrics = {
        'memory': 0,
        'time': 0,
        'cache_size': 0
    }
    
    # Measure over multiple runs
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average the metrics
    num_layers = len(model.layers)
    metrics['memory'] /= (num_runs * num_layers)
    metrics['time'] /= (num_runs * num_layers)
    
    return metrics

def evaluate_attention_mechanisms(num_samples=32):
    """Evaluate different attention mechanisms."""
    print("Using device:", device)
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.2f} GB")
    
    # Initialize results dictionary
    results = {
        'memory_usage': {},
        'inference_speed': {},
        'kqv_cache_metrics': {}
    }
    
    # Adjusted test parameters for A100
    sequence_lengths = [512, 1024, 2048]
    batch_sizes = [1, 4, 16]  # Reduced batch sizes
    
    def display_metrics(attn_type, memory_results, speed_results, kqv_metrics):
        """Display performance metrics for the current attention mechanism."""
        print(f"\n{attn_type.upper()} Performance Metrics:")
        
        print("\nMemory Usage (MB):")
        print("-" * 60)
        print(f"{'Config':<20} {'Memory (MB)':<15}")
        print("-" * 60)
        for config, memory in memory_results.items():
            print(f"{config:<20} {memory:>15.2f}")
        
        print("\nInference Speed (seconds):")
        print("-" * 60)
        print(f"{'Config':<20} {'Time (s)':<15} {'Tokens/sec':>12}")
        print("-" * 60)
        for config, speed in speed_results.items():
            batch_size = int(config.split('b')[1].split('_')[0])
            seq_len = int(config.split('s')[1])
            tokens_per_sec = (batch_size * seq_len) / speed
            print(f"{config:<20} {speed:>15.5f} {tokens_per_sec:>12.0f}")
        
        print("\nKQV Cache Metrics:")
        print("-" * 60)
        print(f"{'Config':<20} {'Memory (MB)':<12} {'Time (ms)':<10} {'Cache (MB)':<10}")
        print("-" * 60)
        for config, metrics in kqv_metrics.items():
            print(f"{config:<20}", end="")
            print(f"{metrics['memory']:>12.2f}", end="")
            print(f"{metrics['time']*1000:>10.2f}", end="")  # Convert to ms
            print(f"{metrics['cache_size']:>10.2f}")
        
        print("\n" + "="*60)
    
    # Evaluate each attention mechanism
    for attn_type in ['mha', 'mqa', 'mla']:
        print(f"\nEvaluating {attn_type.upper()}...")
        
        try:
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
                    try:
                        memory_used = measure_memory_usage(model, device, tokenizer, 
                                                        input_size=(batch_size, seq_len))
                        memory_results[f'b{batch_size}_s{seq_len}'] = memory_used
                        clear_gpu_memory()
                    except RuntimeError as e:
                        print(f"Skipping config b{batch_size}_s{seq_len} due to memory constraints")
                        memory_results[f'b{batch_size}_s{seq_len}'] = float('inf')
            results['memory_usage'][attn_type] = memory_results
            
            # Inference speed
            print("Measuring inference speed...")
            speed_results = {}
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    if memory_results[f'b{batch_size}_s{seq_len}'] != float('inf'):
                        try:
                            avg_time = measure_inference_speed(model, device, tokenizer,
                                                            input_size=(batch_size, seq_len),
                                                            num_runs=10)
                            speed_results[f'b{batch_size}_s{seq_len}'] = avg_time
                            clear_gpu_memory()
                        except RuntimeError as e:
                            print(f"Skipping speed measurement for b{batch_size}_s{seq_len}")
                            speed_results[f'b{batch_size}_s{seq_len}'] = float('inf')
                    else:
                        speed_results[f'b{batch_size}_s{seq_len}'] = float('inf')
            results['inference_speed'][attn_type] = speed_results
            
            # KQV cache performance
            print("Measuring KQV cache performance...")
            kqv_metrics = {}
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    if memory_results[f'b{batch_size}_s{seq_len}'] != float('inf'):
                        try:
                            metrics = measure_kqv_cache_performance(model, device, tokenizer,
                                                                 input_size=(batch_size, seq_len))
                            kqv_metrics[f'b{batch_size}_s{seq_len}'] = metrics
                            clear_gpu_memory()
                        except RuntimeError as e:
                            print(f"Skipping KQV measurement for b{batch_size}_s{seq_len}")
                            kqv_metrics[f'b{batch_size}_s{seq_len}'] = {
                                'memory': float('inf'),
                                'time': float('inf'),
                                'cache_size': float('inf')
                            }
                    else:
                        kqv_metrics[f'b{batch_size}_s{seq_len}'] = {
                            'memory': float('inf'),
                            'time': float('inf'),
                            'cache_size': float('inf')
                        }
            results['kqv_cache_metrics'][attn_type] = kqv_metrics
            
            # Display current metrics
            display_metrics(attn_type, memory_results, speed_results, kqv_metrics)
            
        except Exception as e:
            print(f"Error evaluating {attn_type}: {str(e)}")
        finally:
            # Clear GPU memory
            clear_gpu_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    print("Using device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.2f} GB")
    print("Loading tokenizer...")
    
    # Initialize tokenizer with padding
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Run evaluation
    try:
        results = evaluate_attention_mechanisms(num_samples=32)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        clear_gpu_memory() 