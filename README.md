# Multi-head Latent Attention (MLA) Experiments

This repository contains implementations and experiments comparing different attention mechanisms in transformer models:
- Multi-head Attention (MHA) - The standard attention mechanism
- Multi-Query Attention (MQA) - Shared key and value heads
- Multi-head Latent Attention (MLA) - Using low-rank factorization

## Setup and Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mla-experiments.git
cd mla-experiments

# Install requirements
pip install torch transformers datasets matplotlib numpy tqdm
```

## Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Enable GPU runtime:
   - Runtime > Change runtime type > Hardware accelerator > GPU

4. Clone and set up the repository:
```python
!git clone https://github.com/YOUR_USERNAME/mla-experiments.git
%cd mla-experiments
!pip install torch transformers datasets matplotlib numpy tqdm
```

5. Prepare the data:
```python
%cd data
!python download_data.py
!python hftokenizer.py
!python construct_dataset.py
%cd ..
```

6. Train the models:
```python
!python train_model.py
```

## Project Structure

- `modeling/attention/` - Implementation of attention mechanisms
  - `mha.py` - Multi-head Attention
  - `mqa.py` - Multi-Query Attention
  - `mla.py` - Multi-head Latent Attention
- `data/` - Data preparation scripts
- `train_model.py` - Training script
- `eval_model.py` - Evaluation script

## Training Details

The training process:
1. Downloads and processes the WikiText dataset
2. Creates a custom tokenizer
3. Trains three model variants (MHA, MQA, MLA)
4. Saves model weights and training curves
5. Evaluates models using perplexity

Models are trained with:
- 512 hidden dimensions
- 16 attention heads
- 8 transformer layers
- Mixed precision training
- Gradient accumulation
- Cosine learning rate schedule with warmup

## Results

Training curves and model weights are saved in:
- `figures/` - Training loss curves
- `weights/` - Model checkpoints and final weights