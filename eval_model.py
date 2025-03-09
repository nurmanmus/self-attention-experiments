import torch
import numpy as np
from modeling.gpt import GPTModel

def load_model(model_path, device, use_mla=False, use_mqa=False, use_rope=True):
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000,
                     max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa,
                     use_rope=use_rope, use_decoupled=True)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def calculate_perplexity(model, data, batch_size, device):
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(data) - batch_size, batch_size):            
            if i % 1000 == 0:
                print(f"{i} / {len(data)}")
            batch = data[i:i + batch_size]
            inputs = torch.tensor(batch[:, :-1]).to(device)
            targets = torch.tensor(batch[:, 1:]).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                                         targets.view(-1), reduction='sum')
                
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def evaluate_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "./data/packed_data.npy"
    batch_size = 16

    # Model configurations to evaluate
    configs = [
        {"name": "MHA", "path": "./weights/mha_model_weights.pt", "use_mla": False, "use_mqa": False},
        {"name": "MQA", "path": "./weights/mqa_model_weights.pt", "use_mla": False, "use_mqa": True},
        {"name": "MLA", "path": "./weights/mla_model_weights.pt", "use_mla": True, "use_mqa": False}
    ]

    # Load the evaluation data
    print("Loading evaluation data...")
    with open(data_path, 'rb') as f:
        data = np.load(f)

    results = {}
    
    # Evaluate each model
    for config in configs:
        print(f"\nEvaluating {config['name']} model...")
        try:
            model = load_model(
                config['path'], 
                device,
                use_mla=config['use_mla'],
                use_mqa=config['use_mqa']
            )
            
            perplexity = calculate_perplexity(model, data, batch_size, device)
            results[config['name']] = perplexity
            print(f"{config['name']} Perplexity: {perplexity:.2f}")
            
        except FileNotFoundError:
            print(f"Model weights not found for {config['name']}. Make sure to train the model first.")
            continue
        except Exception as e:
            print(f"Error evaluating {config['name']}: {str(e)}")
            continue

    # Print comparison
    if results:
        print("\nModel Comparison:")
        print("-" * 40)
        print("Model Type | Perplexity")
        print("-" * 40)
        for model_name, perplexity in results.items():
            print(f"{model_name:9} | {perplexity:.2f}")
        print("-" * 40)
    else:
        print("\nNo models were successfully evaluated. Please train the models first.")

if __name__ == "__main__":
    evaluate_models()
