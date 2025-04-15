# TorchOpt

A modular optimization library based on PyTorch for implementing and experimenting with various optimization algorithms.

## Motivation

As an ordinary college student learning optimization algorithms, I found that existing libraries for optimization and benchmark testing were often fragmented, lacked compatibility, and made it difficult to compare different algorithms effectively. Additionally, parallelizing these algorithms was not straightforward.

During my studies, I wanted to create a project that would help me better understand these algorithms while also providing a practical tool for experimentation. By leveraging PyTorch's powerful tensor operations and parallel computing capabilities, I created this unified framework to aid my learning process.

The goal of TorchOpt is to:

- Provide a modular design with base classes that can easily integrate various test functions and optimization algorithms.
- Enable seamless GPU acceleration for faster computation.
- Simplify the process of comparing and experimenting with different optimization techniques.
- Serve as a learning tool for students like me who are interested in optimization algorithms.

## Overview

TorchOpt is a lightweight, extensible framework for optimization algorithms built on PyTorch. It provides:

- A collection of classic benchmark functions for testing optimization algorithms
- Easy-to-extend base classes for creating custom optimization algorithms
- GPU acceleration support through PyTorch's device management
- Built-in visualization and tracking of optimization progress

## Installation

```bash
pip install torchopt  # Coming soon!
```

Or install from source:

```bash
git clone https://github.com/why1249/torchopt.git
cd torchopt
pip install -e .
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy
- tqdm

## Quick Start

```python
from algorithm import pso
from benchmarks import classic
import torch

# Define test function
dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
test_function = classic.Rastrigin(dim=dim, bounds=bounds)

# Set PSO parameters
parameters = {
    'c1': 1.5,  # Cognitive parameter
    'c2': 1.5,  # Social parameter
    'w': 0.5,   # Inertia weight
}

# Initialize optimizer
pso_instance = pso.PSO(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters=parameters,
    print_interval=10,
)

# Run optimization
best_solution, best_fitness = pso_instance.optimize()

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Available Benchmark Functions

TorchOpt includes several classic benchmark functions:

- **Sphere**: A simple unimodal function
- **Rastrigin**: A highly multimodal function with many local minima
- **Rosenbrock**: A function with a narrow valley from local optimum to global optimum
- **Griewank**: A multimodal function with interdependence among variables
- **Ackley**: A function with many local minima but one global minimum
- **Schwefel**: A complex function with the global minimum far from the center

More benchmark functions will be added in future releases.

## Available Optimization Algorithms

Currently implemented algorithms:

- **PSO (Particle Swarm Optimization)**: A population-based stochastic optimization technique

Additional optimization algorithms are planned for future updates.

## Extending TorchOpt

### Creating a New Test Function

Extend the `TestFunction` base class:

```python
from core.base import TestFunction
import torch

class MyCustomFunction(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Implement your function here
        return torch.sum(x ** 2, dim=-1)  # Example: Sphere function
```

### Creating a New Optimizer

Extend the `Optimizer` base class:

```python
from core.base import Optimizer, TestFunction
import torch

class MyOptimizer(Optimizer):
    def __init__(self, test_function: TestFunction, 
                 population_size: int, max_iter: int, 
                 parameters: dict, print_interval: int = 10, 
                 device: str = 'cpu'):
        super().__init__(test_function, population_size, max_iter, 
                         parameters, print_interval, device)
    
    def initialize(self):
        # Initialize your algorithm
        pass
    
    def update(self):
        # Update your algorithm for one iteration
        pass
        
    def _get_best(self):
        # Return the best solution and fitness
        pass
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.
