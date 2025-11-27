### V-Torch

## A torch library for V

V-Torch is a deep learning library for the [V programming language](vlang.io).

## Roadmap & Implementation Plan

The development is prioritized as follows to reach a working prototype:

1. **Tensor Module** (CPU only)
    - [ ] Data storage (`[]f32`)
    - [ ] Shape & Strides (`compute_strides`)
    - [ ] Basic Arithmetic (Add, Sub, Mul, Div, Matmul)
    - [ ] Broadcasting support

2. **Autograd Module**
    - [ ] Computation Graph (DAG)
    - [ ] Backward engine (Recursive/Topological sort)
    - [ ] `Function` interface for custom ops

3. **NN Module**
    - [ ] `Module` interface
    - [ ] Linear Layers
    - [ ] Activations (ReLU, Sigmoid, Tanh)
    - [ ] Loss Functions (MSE, CrossEntropy)

4. **Optimizer Module**
    - [ ] Optimizer interface
    - [ ] SGD
    - [ ] Adam

5. **Verification & Examples**
    - [ ] XOR Example
    - [ ] MNIST Training Script

## Project Structure

The planned directory structure for the library:

```
V-lib/
├─ tensor/       # Core Tensor struct + arithmetic + broadcasting
├─ autograd/     # Backward engine, DAG, Function interface
├─ nn/           # Module interface, Linear, Conv, activations, loss
├─ optim/        # SGD, Adam, Optimizer interface
├─ examples/     # Training scripts (MNIST, XOR)
├─ tests/        # _test.v files for each module
└─ main.v        # Entry point for demos
```

## Goals

- **Pure V**: Focus on correctness and API ergonomics in pure V first.
- **Modularity**: Each module should be self-contained with its own tests.
- **Performance**: Start with correctness, then optimize (e.g., single allocation for intermediates).
