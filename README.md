# mygpt

A small GPT-style language model (Lumiere) built from scratch in PyTorch.

## Setup

```bash
pip install torch tiktoken matplotlib
```

## Usage

```bash
python train.py          # train on the_verdict.txt
python plot_losses.py    # plot training/validation loss
```

Checkpoints are saved locally to `model_checkpoints/` (not tracked in git).
