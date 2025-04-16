# Benchmark Functions

TorchOptLib includes several classic benchmark functions commonly used to evaluate optimization algorithms. These functions have known characteristics that help assess an algorithm's performance in different scenarios.

## Available Functions

### Sphere Function

A simple unimodal function that is continuous, convex, and separable. It's often used as a baseline for testing optimization algorithms.

```python
from torchoptlib.benchmarks import classic
import torch

dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
sphere = classic.Sphere(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = \sum_{i=1}^{n} x_i^2$

### Rastrigin Function

A highly multimodal function with many local minima arranged in a regular pattern. It's a challenging function for many optimization algorithms.

```python
rastrigin = classic.Rastrigin(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = 10n + \sum_{i=1}^{n} [x_i^2 - 10\cos(2\pi x_i)]$

### Rosenbrock Function

A function with a narrow valley from local optimum to global optimum. Finding the valley is easy, but converging to the global optimum is difficult.

```python
rosenbrock = classic.Rosenbrock(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$

### Griewank Function

A multimodal function with interdependence among variables. It has many local minima due to the cosine term.

```python
griewank = classic.Griewank(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = 1 + \frac{1}{4000}\sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos(\frac{x_i}{\sqrt{i}})$

### Ackley Function

A function with many local minima but one global minimum. It combines exponential terms with cosine modulation.

```python
ackley = classic.Ackley(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = -20\exp(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}) - \exp(\frac{1}{n}\sum_{i=1}^{n} \cos(2\pi x_i)) + 20 + e$

### Schwefel Function

A complex function with the global minimum far from the center and close to the bounds. It presents challenges due to its deep local minima.

```python
schwefel = classic.Schwefel(dim=dim, bounds=bounds)
```

Mathematical formula: $f(x) = 418.9829n - \sum_{i=1}^{n} x_i\sin(\sqrt{|x_i|})$

## Creating Custom Benchmark Functions

You can create your own benchmark functions by extending the `TestFunction` base class:

```python
from torchoptlib.core.base import TestFunction
import torch

class MyCustomFunction(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Implement your function here
        return torch.sum(x ** 2, dim=-1)  # Example: Sphere function
```

The `evaluate` method should accept a tensor of shape `(batch_size, dim)` and return a tensor of shape `(batch_size,)` containing the function values.
