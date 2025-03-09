import torch
import numpy as np
from modeling.gpt import GPTModel
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
import time

def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

def plot_loss_curve(x, y, model_type):
    plt.figure()
    plt.plot(x, y)
    plt.title(f"LLM Training Loss with {model_type}")
    plt.xlabel("tokens")
    plt.ylabel("cross entropy loss")
    plt.savefig(f"./figures/{model_type.lower().replace(' ', '_')}_training_curve.png")
    plt.close()

def train_model(model_type="MHA"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_type} on {device}")
    
    # Configure model based on type
    use_mla = model_type == "MLA"
    use_mqa = model_type == "MQA"
    use_rope = True  # Using RoPE for all models
    
    # Full-sized model since we have GPU
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000,
                     max_seq_len=1024, use_mla=use_mla, use_mqa=use_mqa,
                     use_rope=use_rope, use_decoupled=True)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_type} model has {param_count} parameters.")

    model = model.to(device)

    # gradient scaling for mixed precision training
    scaler = GradScaler()
    acc_steps = 4  # Gradient accumulation steps

    batch_size = 12  # Larger batch size for GPU
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.999))
    
    # Full training steps
    total_steps = 1e5  # Reduced from original but still substantial
    scheduler = cosine_with_warmup_lr_scheduler(opt, total_steps, int(total_steps * 0.01))
    loss_fn = torch.nn.CrossEntropyLoss()

    # loading the data
    print("Loading data...")
    with open('./data/packed_data.npy', 'rb') as f:
        data = np.load(f)
    
    dataset = TensorDataset(torch.from_numpy(data[:, :-1]), torch.from_numpy(data[:, 1:]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)

    #logging
    total_tokens = 0
    train_losses_y = []
    train_losses_x = []

    print(f"Starting training for {model_type}...")
    start_time = time.time()
    
    for epoch in range(3):  # Train for 3 epochs
        for i, (dat, targ) in enumerate(dataloader):
            dat = dat.to(device, non_blocking=True)
            targ = targ.to(device, non_blocking=True)
            
            if i % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch}, Batch {i}/{len(dataloader)}, Time elapsed: {elapsed:.2f}s")

            # Mixed precision training
            with torch.cuda.amp.autocast():
                out, _ = model(dat)
                out = out.permute(0, 2, 1)
                loss = loss_fn(out, targ)
                loss = loss / acc_steps

            scaler.scale(loss).backward()

            if (i + 1) % acc_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                opt.zero_grad(set_to_none=True)

            total_tokens += dat.numel()

            if (i + 1) % (10 * acc_steps) == 0:
                train_losses_x.append(total_tokens)
                train_losses_y.append(loss.item() * acc_steps)  # Adjust for accumulation
                print(f"{model_type} - Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item() * acc_steps:.4f}")
                plot_loss_curve(train_losses_x, train_losses_y, model_type)

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss.item(),
        }, f"./weights/{model_type.lower()}_checkpoint_epoch_{epoch}.pt")

    # save final model weights
    torch.save(model.state_dict(), f"./weights/{model_type.lower()}_model_weights.pt")
    print(f"Finished training {model_type} model")
    print(f"Total training time: {time.time() - start_time:.2f}s")

def train():
    # Train all three variants
    for model_type in ["MHA", "MQA", "MLA"]:
        print(f"\nTraining {model_type}...")
        train_model(model_type)
        print(f"Completed {model_type} training\n")

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create necessary directories
    import os
    os.makedirs("weights", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    
    train()
