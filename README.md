# Karpathy NN Zero to Hero

Notes and exercises while following [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI05v9XCA/G0wIecFHzbMW) series.

This repo captures progress through the lectures, implemented step by step in Jupyter notebooks.

## Contents

### `micrograd/`

- `micrograd_from_scratch_yay.ipynb` — building autograd and a tiny neural net from scratch:
  - `Value` class with forward ops (`+`, `*`, `/`, `**`, `tanh`, `exp`) and reverse-mode autograd
  - Computation graph visualization with Graphviz
  - Manual and topological-sort backprop
  - Single-neuron and MLP examples with training loop setup
- `micrograd_cold_rebuild.ipynb` — cold rebuild of micrograd without looking back at the original notes:
  - Reimplemented `Value`, `Neuron`, `Layer`, and `MLP` from memory
  - Unit tests for ops, mixed Value/scalar operands, layers, and the full MLP
  - End-to-end training on the lecture toy dataset to verify forward + backward together

### `makemore/`

- `names.txt` — baby-name training data used by both notebooks
- `build_makemore_yay.ipynb` — lecture walkthrough of a character-level bigram language model:
  - Counting bigrams, converting counts to probabilities (with smoothing), and sampling names
  - Negative log-likelihood as the training objective
  - The same model as a one-layer neural net: one-hot inputs, softmax, NLL + L2, gradient descent
- `makemore_cold_rebuild.ipynb` — cold rebuild of the bigram model without looking back at the original notes:
  - Recounted bigrams, sampled from the count table, and scored NLL
  - Retrained the neural-net version (softmax + L2) and sampled names from it

### `makemore part 2/`

- `names.txt` — same baby-name training data as `makemore/`
- `building_makemore_mlp_yay.ipynb` — lecture walkthrough of a character-level MLP language model (Bengio et al.):
  - Context window of 3 previous characters mapped through a learned embedding table
  - Hidden tanh layer + softmax over the next character, trained with negative log-likelihood
  - Train / val / test splits, learning-rate search, and sampling names from the trained model
- `makemore_mlp_cold_rebuild.ipynb` — cold rebuild of the MLP without looking back at the original notes:
  - Rebuilt the 3-character context dataset with train / val / test splits
  - Retrained the embedding + tanh + softmax model (cross-entropy, minibatch SGD, learning-rate decay)
  - Evaluated train and validation loss
- `A Neural Probabilistic Language Model.pdf` — Bengio et al. 2003 paper this lecture implements

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For Graphviz diagram rendering (micrograd notebooks), install the system Graphviz binary as well:

```bash
brew install graphviz   # macOS
```

Then open a notebook under `micrograd/`, `makemore/`, or `makemore part 2/` in Jupyter or VS Code/Cursor and select the project `venv` kernel.

## Reference

- Course repo: [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
- Lecture 1: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Lecture 2: [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- Lecture 3: [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
- Original micrograd repo: [karpathy/micrograd](https://github.com/karpathy/micrograd)
- Original makemore repo: [karpathy/makemore](https://github.com/karpathy/makemore)
