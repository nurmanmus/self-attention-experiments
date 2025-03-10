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
import torch.utils.checkpoint as checkpoint
import functools

# Set up device and memory management
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # Set memory allocation settings
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
    torch.backends.cudnn.benchmark = True
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for supported modules."""
    def create_custom_forward(module):
        orig_forward = module.forward
        
        @functools.wraps(orig_forward)
        def custom_forward(*args, **kwargs):
            return module._forward(*args, **kwargs)
        
        def _checkpointed_forward(*args, **kwargs):
            return checkpoint.checkpoint(custom_forward, *args, use_reentrant=False, **kwargs)
        
        module.forward = _checkpointed_forward
    
    # Apply checkpointing to transformer layers
    for module in model.layers:
        if hasattr(module, 'forward'):
            module._forward = module.forward
            create_custom_forward(module)
    
    return model

def initialize_model(attn_type):
    """Initialize model with specified attention type."""
    from modeling.gpt import GPTModel
    
    print(f"\nInitializing {attn_type.upper()} model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Parse attention type and variant
    use_mla = attn_type.startswith('mla')
    use_mqa = attn_type.startswith('mqa')
    use_rope = 'rope' in attn_type
    cache_compress = not ('uncompressed' in attn_type)
    
    print(f"Attention configuration:")
    print(f"  use_mla: {use_mla}")
    print(f"  use_mqa: {use_mqa}")
    print(f"  use_rope: {use_rope}")
    print(f"  cache_compress: {cache_compress}")
    
    # Fixed model dimensions for consistency
    model_config = {
        'd_model': 512,      # Model dimension
        'n_heads': 8,        # Number of attention heads
        'layers': 12,        # Number of transformer layers
        'vocab_size': 50257, # GPT-2 vocab size
        'max_seq_len': 2048, # Maximum sequence length
        'use_mla': use_mla,
        'use_mqa': use_mqa,
        'use_rope': use_rope,
        'cache_compress': cache_compress
    }
    
    # Add latent_dim only for MLA
    if use_mla:
        model_config['latent_dim'] = 64
    
    print(f"\nModel configuration:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")
    
    try:
        print("\nCreating model instance...")
        model = GPTModel(**model_config)
        print("Model created successfully")
        
        print("\nModel structure:")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize weights deterministically
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Use the same initialization for all linear layers
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        print("\nInitializing weights...")
        model.apply(init_weights)
        print("Weights initialized")
        
        # Enable memory efficient features
        print("\nEnabling gradient checkpointing...")
        model = enable_gradient_checkpointing(model)
        print("Gradient checkpointing enabled")
        
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

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
        with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
            x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
            x = x.to(device)
            
            if len(x.shape) > 2:
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
            
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=True):
                outputs = model(x)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                torch.cuda.synchronize()  # Ensure all operations are completed
        
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
        with torch.cuda.amp.autocast(enabled=True):  # Use mixed precision
            x, _ = load_test_data(tokenizer, sequence_length=input_size[1], num_samples=input_size[0])
            x = x.to(device)
            
            if len(x.shape) > 2:
                batch_size = x.size(0)
                x = x.view(batch_size, -1)
            
            # Warmup with fewer runs
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=True):
                for _ in range(5):
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    torch.cuda.synchronize()
            
            clear_gpu_memory()
            
            # Measure time
            start_time = time.time()
            with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=True):
                for _ in range(num_runs):
                    outputs = model(x)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    torch.cuda.synchronize()
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
                
                # Check if this is RoPE or Ropeless variant
                if hasattr(module, 'qk_rope_dim'):
                    # RoPE MLA
                    compressed_kv = inputs[0] @ module.W_dkv
                    KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                         [module.kv_proj_dim, module.qk_rope_dim],
                                                         dim=-1)
                    KV_for_lora = module.kv_layernorm(KV_for_lora)
                    KV = KV_for_lora @ module.W_ukv
                    KV = KV.view(B, -1, module.n_heads, module.dh+module.qk_nope_dim).transpose(1,2)
                    K, V = torch.split(KV, [module.qk_nope_dim, module.dh], dim=-1)
                    
                    # Include RoPE part in cache size calculation
                    K_for_rope = K_for_rope.view(B, -1, 1, module.qk_rope_dim).transpose(1,2)
                    K_for_rope = K_for_rope.repeat(1, module.n_heads, 1, 1)
                    cache_size = (K.numel() + K_for_rope.numel() + V.numel())
                else:
                    # Ropeless MLA
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
        if any(name.endswith(suffix) for suffix in ['.mha', '.mqa', '.mla']):
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

def compute_model_outputs(model, input_ids, device):
    """Compute model outputs with proper error handling and device management."""
    try:
        print(f"Input shape: {input_ids.shape}")
        print(f"Device: {device}")
        print(f"Model device: {next(model.parameters()).device}")
        
        model.eval()
        input_ids = input_ids.to(device)
        print(f"Inputs moved to device: {input_ids.device}")
        
        with torch.no_grad():
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    print("Computing model outputs...")
                    outputs = model(input_ids)
                    print(f"Raw output type: {type(outputs)}")
                    
                    if isinstance(outputs, tuple):
                        print(f"Output tuple length: {len(outputs)}")
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    print(f"Logits shape: {logits.shape}")
                    
                    # Ensure logits are the right shape (batch_size, seq_len, vocab_size)
                    if len(logits.shape) != 3:
                        raise ValueError(f"Expected logits shape (batch_size, seq_len, vocab_size), got {logits.shape}")
                    
                    # Get predictions
                    print("Computing probabilities...")
                    # Move to CPU first to avoid potential GPU memory issues
                    logits_cpu = logits.cpu().to(torch.float32)
                    probs = torch.softmax(logits_cpu, dim=-1)
                    print(f"Probabilities shape: {probs.shape}")
                    
                    print("Computing predictions...")
                    predictions = torch.argmax(probs, dim=-1)
                    print(f"Predictions shape: {predictions.shape}")
                    
                    # Ensure all outputs have consistent shapes
                    batch_size, seq_len, vocab_size = logits_cpu.shape
                    
                    # Verify and reshape if needed
                    if probs.shape != (batch_size, seq_len, vocab_size):
                        print(f"Warning: Reshaping probs from {probs.shape} to {(batch_size, seq_len, vocab_size)}")
                        probs = probs.view(batch_size, seq_len, vocab_size)
                    
                    if predictions.shape != (batch_size, seq_len):
                        print(f"Warning: Reshaping predictions from {predictions.shape} to {(batch_size, seq_len)}")
                        predictions = predictions.view(batch_size, seq_len)
                    
                    outputs_dict = {
                        'logits': logits_cpu,
                        'probs': probs,
                        'predictions': predictions
                    }
                    
                    # Final shape verification
                    print("\nVerifying output shapes:")
                    print(f"  logits: {outputs_dict['logits'].shape}")
                    print(f"  probs: {outputs_dict['probs'].shape}")
                    print(f"  predictions: {outputs_dict['predictions'].shape}")
                    
                    return outputs_dict
                    
            except RuntimeError as e:
                print(f"Runtime error during model computation: {str(e)}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")
                print(f"CUDA memory cached: {torch.cuda.memory_reserved()/1024/1024/1024:.2f} GB")
                return None
            except Exception as e:
                print(f"Unexpected error during model computation: {str(e)}")
                return None
    except Exception as e:
        print(f"Error in compute_model_outputs: {str(e)}")
        return None

def compare_outputs(outputs1, outputs2, name1, name2, tolerance=1e-3):
    """Compare outputs between two models and return detailed metrics."""
    if outputs1 is None or outputs2 is None:
        return {
            'match': False,
            'error': 'One or both outputs are None'
        }
    
    try:
        # Get shapes
        shapes1 = {k: v.shape for k, v in outputs1.items()}
        shapes2 = {k: v.shape for k, v in outputs2.items()}
        
        print(f"\nShape comparison for {name1} vs {name2}:")
        for key in shapes1.keys():
            print(f"  {key}: {shapes1[key]} vs {shapes2.get(key, 'missing')}")
        
        # Verify shapes match for each component
        for key in ['logits', 'probs', 'predictions']:
            if key not in outputs1 or key not in outputs2:
                return {
                    'match': False,
                    'error': f"Missing {key} in one of the outputs"
                }
            if outputs1[key].shape != outputs2[key].shape:
                return {
                    'match': False,
                    'error': f"Shape mismatch for {key}: {outputs1[key].shape} vs {outputs2[key].shape}"
                }
        
        results = {}
        
        # Compare logits
        logits_diff = torch.abs(outputs1['logits'] - outputs2['logits'])
        results['max_logits_diff'] = logits_diff.max().item()
        results['mean_logits_diff'] = logits_diff.mean().item()
        results['std_logits_diff'] = logits_diff.std().item()
        
        # Compare probabilities
        probs_diff = torch.abs(outputs1['probs'] - outputs2['probs'])
        results['max_probs_diff'] = probs_diff.max().item()
        results['mean_probs_diff'] = probs_diff.mean().item()
        results['std_probs_diff'] = probs_diff.std().item()
        
        # Compare top-k predictions
        k = 5
        top_k1 = torch.topk(outputs1['probs'], k, dim=-1).indices
        top_k2 = torch.topk(outputs2['probs'], k, dim=-1).indices
        
        # Compute overlap for each position
        top_k_overlap = torch.zeros_like(top_k1[:, :, 0], dtype=torch.float32)
        for i in range(k):
            for j in range(k):
                top_k_overlap += (top_k1[:, :, i] == top_k2[:, :, j]).float()
        
        results['top_k_match_rate'] = (top_k_overlap > 0).float().mean().item()
        
        # Exact prediction match rate
        pred_match = (outputs1['predictions'] == outputs2['predictions']).float()
        results['prediction_match_rate'] = pred_match.mean().item()
        
        # Overall match criteria
        results['match'] = (
            results['max_logits_diff'] < tolerance and
            results['max_probs_diff'] < tolerance and
            results['prediction_match_rate'] > 0.9  # Allow for some minor differences
        )
        
        # Print detailed comparison results
        print(f"\nDetailed comparison of {name1} vs {name2}:")
        print(f"Logits differences:")
        print(f"  Max: {results['max_logits_diff']:.6f}")
        print(f"  Mean: {results['mean_logits_diff']:.6f}")
        print(f"  Std: {results['std_logits_diff']:.6f}")
        
        print(f"\nProbability differences:")
        print(f"  Max: {results['max_probs_diff']:.6f}")
        print(f"  Mean: {results['mean_probs_diff']:.6f}")
        print(f"  Std: {results['std_probs_diff']:.6f}")
        
        print(f"\nPrediction metrics:")
        print(f"  Top-{k} overlap rate: {results['top_k_match_rate']*100:.2f}%")
        print(f"  Exact match rate: {results['prediction_match_rate']*100:.2f}%")
        print(f"  Overall match: {'Yes' if results['match'] else 'No'}")
        
        return results
        
    except Exception as e:
        import traceback
        print(f"Error during comparison:")
        print(traceback.format_exc())
        return {
            'match': False,
            'error': f"Comparison error: {str(e)}"
        }

def validate_attention_mechanisms(models, test_inputs, device, tolerance=1e-4):
    """Validate that different attention mechanisms produce consistent outputs."""
    print("\nValidating attention mechanism outputs...")
    print(f"Test inputs shape: {test_inputs.shape}")
    print(f"Device: {device}")
    print(f"Using tolerance: {tolerance}")
    
    outputs = {}
    comparisons = {}
    
    # Compute outputs for each model
    for name, model in models.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing {name.upper()}...")
            print(f"Model device: {next(model.parameters()).device}")
            
            outputs[name] = compute_model_outputs(model, test_inputs, device)
            
            if outputs[name] is None:
                print(f"Warning: Failed to compute outputs for {name}")
            else:
                print(f"Successfully computed outputs for {name}")
                print(f"Output shapes:")
                print(f"  logits: {outputs[name]['logits'].shape}")
                print(f"  probs: {outputs[name]['probs'].shape}")
                print(f"  predictions: {outputs[name]['predictions'].shape}")
        except Exception as e:
            print(f"Error computing outputs for {name}: {str(e)}")
            outputs[name] = None
    
    # Compare outputs between all pairs of models
    print("\nComparing model outputs...")
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            comparison_key = f"{name1}_vs_{name2}"
            
            print(f"\n{'-'*50}")
            print(f"Comparing {name1.upper()} vs {name2.upper()}...")
            
            try:
                if outputs[name1] is None or outputs[name2] is None:
                    print(f"Skipping comparison due to missing outputs")
                    comparisons[comparison_key] = {'match': False, 'error': 'Missing outputs'}
                    continue
                
                print("Both outputs available, performing comparison...")
                comparisons[comparison_key] = compare_outputs(
                    outputs[name1], outputs[name2], name1, name2, tolerance
                )
                
                if 'error' in comparisons[comparison_key]:
                    print(f"Comparison error: {comparisons[comparison_key]['error']}")
                else:
                    print(f"Comparison completed: Match = {comparisons[comparison_key]['match']}")
                    
            except Exception as e:
                print(f"Error during comparison: {str(e)}")
                comparisons[comparison_key] = {'match': False, 'error': str(e)}
    
    # Overall validation result
    print("\nAnalyzing validation results...")
    valid_comparisons = [comp for comp in comparisons.values() if 'error' not in comp]
    
    if valid_comparisons:
        all_match = all(comp['match'] for comp in valid_comparisons)
        print("\nOverall Validation Result:")
        print(f"Total comparisons: {len(comparisons)}")
        print(f"Valid comparisons: {len(valid_comparisons)}")
        print(f"All mechanisms produce consistent outputs: {'Yes' if all_match else 'No'}")
        
        if not all_match:
            print("\nInconsistent comparisons:")
            for key, comp in comparisons.items():
                if 'match' in comp and not comp['match']:
                    print(f"- {key}")
    else:
        print("\nWarning: No valid comparisons were made")
        print("Comparison errors:")
        for key, comp in comparisons.items():
            if 'error' in comp:
                print(f"- {key}: {comp['error']}")
    
    return comparisons

def evaluate_attention_mechanisms(num_samples=32):
    """Evaluate different attention mechanisms."""
    print("Using device:", device)
    if torch.cuda.is_available():
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory/1024/1024/1024:.2f} GB")
        print(f"Current GPU Memory Usage: {torch.cuda.memory_allocated()/1024/1024/1024:.2f} GB")
    
    # Initialize results dictionary
    results = {
        'memory_usage': {},
        'inference_speed': {},
        'kqv_cache_metrics': {},
        'validation_results': {}
    }
    
    # Initialize models for RoPE attention mechanisms only
    models = {}
    attention_types = [
        'mha_rope',         # Multi-Head Attention with RoPE
        'mqa_rope',         # Multi-Query Attention with RoPE
        'mla_rope',         # Multi-Level Attention with RoPE
    ]
    
    for attn_type in attention_types:
        try:
            print(f"\nInitializing {attn_type.upper()} model...")
            models[attn_type] = initialize_model(attn_type)
            models[attn_type].to(device)
            models[attn_type].eval()
        except Exception as e:
            print(f"Error initializing {attn_type} model: {str(e)}")
            models[attn_type] = None
    
    # Generate test data for validation
    print("\nPreparing validation data...")
    try:
        # Use smaller sequence length and batch size for validation
        validation_inputs, _ = load_test_data(tokenizer, sequence_length=128, num_samples=2)
    except Exception as e:
        print(f"Error loading validation data: {str(e)}")
        return results
    
    # Perform validation only for successfully initialized models
    valid_models = {k: v for k, v in models.items() if v is not None}
    if valid_models:
        results['validation_results'] = validate_attention_mechanisms(
            valid_models, validation_inputs, device, tolerance=1e-4
        )
    else:
        print("No valid models to validate")
        return results
    
    # Adjusted test parameters
    sequence_lengths = [128, 256, 512]  # Reduced sequence lengths
    batch_sizes = [1, 2, 4]  # Reduced batch sizes
    
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
    for attn_type in attention_types:
        print(f"\nEvaluating {attn_type.upper()}...")
        
        try:
            # Memory usage
            print("Measuring memory usage...")
            memory_results = {}
            for seq_len in sequence_lengths:
                for batch_size in batch_sizes:
                    try:
                        memory_used = measure_memory_usage(models[attn_type], device, tokenizer, 
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
                            avg_time = measure_inference_speed(models[attn_type], device, tokenizer,
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
                            metrics = measure_kqv_cache_performance(models[attn_type], device, tokenizer,
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