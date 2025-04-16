# Algorithms

TorchOptLib provides implementations of various optimization algorithms and supports custom optimizer creation.

## Implemented Algorithms

Currently implemented algorithms in the library include:

- **PSO (Particle Swarm Optimization)**: A population-based stochastic optimization technique
- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**: An evolutionary algorithm for difficult non-linear non-convex optimization problems

Each algorithm can be called through a simple interface and supports GPU acceleration.

## Algorithm Usage Example

```python
from torchoptlib.algorithm import pso
from torchoptlib.benchmarks import classic
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
```

## Custom Optimizers

You can create your own optimization algorithm by extending the `Optimizer` base class:

```python
from torchoptlib.core.base import Optimizer, TestFunction
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
        # For example: generate initial population, set initial states, etc.
        pass
    
    def update(self):
        # Implement update logic for a single iteration
        # This is the core part of the algorithm
        pass
        
    def _get_best(self):
        # Return the best solution and fitness
        # For example: return self.global_best_position, self.global_best_fitness
        pass
```

When creating a custom optimizer, you must implement these three methods:

- `initialize()`: Initialize all variables and states needed by the algorithm
- `update()`: Contain the logic for a single iteration of the algorithm
- `_get_best()`: Return the current best solution and its corresponding fitness value

All these methods should leverage PyTorch's tensor operations to support GPU acceleration.

## Parameter Configuration

Most optimization algorithms require specific parameters to control their behavior. When initializing an optimizer, you can pass these parameters through the `parameters` dictionary:

```python
parameters = {
    'param1': value1,
    'param2': value2,
    # ...more parameters
}

optimizer = MyOptimizer(
    test_function=function,
    population_size=50,
    max_iter=200,
    parameters=parameters
)
```

Refer to the documentation of each algorithm for their specific parameters.
