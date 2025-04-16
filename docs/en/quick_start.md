# Quick Start Guide

This guide will help you get started with TorchOptLib quickly by walking through a simple example.

## Basic Usage Example

The following example demonstrates how to use the Particle Swarm Optimization (PSO) algorithm to optimize the Rastrigin function:

```python
from torchoptlib.benchmarks import classic
from torchoptlib.algorithm import pso
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

## Step-by-Step Breakdown

1. **Import necessary modules**: Import the benchmark function and optimization algorithm from their respective modules.

2. **Define the test function**: Create a benchmark function with appropriate dimension and bounds.

3. **Set algorithm parameters**: Configure the parameters specific to the optimization algorithm.

4. **Initialize the optimizer**: Create an instance of the optimizer with the test function and parameters.

5. **Run the optimization**: Call the `optimize()` method to start the optimization process.

6. **Retrieve results**: Get the best solution and its fitness value.

## Using GPU Acceleration

To use GPU acceleration, simply specify the device when creating the optimizer:

```python
pso_instance = pso.PSO(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters=parameters,
    print_interval=10,
    device='cuda'  # Use GPU if available
)
```

Make sure you have a CUDA-compatible GPU and the CUDA toolkit installed.

## Next Steps

After getting familiar with the basic usage, you can:

- Explore different [benchmark functions](benchmarks.md)
- Try other [optimization algorithms](algorithms.md)
- Learn how to [extend TorchOptLib](extending.md) with your own functions and algorithms
