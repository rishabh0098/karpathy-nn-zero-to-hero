# Karpathy NN Zero to Hero — Micrograd from Scratch

Notes and exercises while following [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI05v9XCA/G0wIecFHzbMW) series.

This repo captures progress after completing the **first lecture** (*Micrograd: Explaining Backpropagation*), implemented step by step in Jupyter notebooks.

## Contents

### `lecture-1/`

- `micrograd_from_scratch_yay.ipynb` — building autograd and a tiny neural net from scratch:
  - `Value` class with forward ops (`+`, `*`, `/`, `**`, `tanh`, `exp`) and reverse-mode autograd
  - Computation graph visualization with Graphviz
  - Manual and topological-sort backprop
  - Single-neuron and MLP examples with training loop setup
- `micrograd_cold_rebuild.ipynb` — cold rebuild of micrograd without looking back at the original notes:
  - Reimplemented `Value`, `Neuron`, `Layer`, and `MLP` from memory
  - Unit tests for ops, mixed Value/scalar operands, layers, and the full MLP
  - End-to-end training on the lecture toy dataset to verify forward + backward together

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For Graphviz diagram rendering, install the system Graphviz binary as well:

```bash
brew install graphviz   # macOS
```

Then open a notebook under `lecture-1/` in Jupyter or VS Code/Cursor and select the project `venv` kernel.

## Reference

- Lecture: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Original micrograd repo: [karpathy/micrograd](https://github.com/karpathy/micrograd)
