# Extending TorchOptLib

TorchOptLib is designed to be easily extensible. You can create your own test functions and optimization algorithms to suit your specific needs.

## Creating Custom Test Functions

To create a custom test function, you need to extend the `TestFunction` base class and implement the required methods:

```python
from torchoptlib.core.base import TestFunction
import torch

class MyCustomFunction(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
        # Initialize any additional attributes here
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # Implement your function here
        # x has shape (batch_size, dim)
        # Return tensor of shape (batch_size,)
        return torch.sum(x ** 2, dim=-1)  # Example: Sphere function
```

### Key Components:

1. **Constructor**: Initialize your function with dimension and bounds
2. **evaluate()**: The core method that computes the function value for given points
   - Input: Tensor of shape `(batch_size, dim)`
   - Output: Tensor of shape `(batch_size,)`

### Example: Custom Variant of Rastrigin Function

```python
class CustomRastrigin(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor], A: float = 10.0):
        super().__init__(dim, bounds)
        self.A = A
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        n = self.dim
        return self.A * n + torch.sum(x**2 - self.A * torch.cos(2 * torch.pi * x), dim=-1)
```

## Creating Custom Optimization Algorithms

To create a custom optimizer, extend the `Optimizer` base class and implement the required methods:

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
        # Initialize the algorithm state
        # Example: Generate initial population
        min_b, max_b = self.test_function.bounds
        self.positions = torch.rand((self.population_size, self.test_function.dim),
                                   device=self.device) * (max_b - min_b) + min_b
        # ... initialize other state variables
    
    def update(self):
        # Implement one iteration of your algorithm
        # This method will be called for each iteration
        # Example: Update positions based on your algorithm's logic
        pass
        
    def _get_best(self):
        # Return the best solution and fitness
        # Example: Find the individual with minimum fitness
        best_idx = torch.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx]
```

### Key Components:

1. **Constructor**: Initialize your optimizer with problem parameters
2. **initialize()**: Set up initial state of the algorithm
3. **update()**: Implement one iteration of the optimization process
4. **_get_best()**: Return the current best solution and its fitness value

### Example: Simplified Simulated Annealing

```python
class SimulatedAnnealing(Optimizer):
    def __init__(self, test_function: TestFunction,
                 population_size: int = 1,  # SA typically uses just one solution
                 max_iter: int = 1000,
                 parameters: dict = {'temp_start': 1.0, 'temp_end': 0.01, 'cooling_rate': 0.95},
                 print_interval: int = 10,
                 device: str = 'cpu'):
        super().__init__(test_function, population_size, max_iter, parameters, print_interval, device)
    
    def initialize(self):
        min_b, max_b = self.test_function.bounds
        # Initialize single position
        self.position = torch.rand((self.test_function.dim,), device=self.device) * (max_b - min_b) + min_b
        self.fitness = self._evaluate(self.position.unsqueeze(0)).squeeze()
        
        self.best_position = self.position.clone()
        self.best_fitness = self.fitness.clone()
        
        # Initialize temperature
        self.temp = self.parameters['temp_start']
        
    def update(self):
        min_b, max_b = self.test_function.bounds
        
        # Generate new candidate solution with small perturbation
        step_size = 0.1 * (max_b - min_b)
        new_position = self.position + (torch.rand_like(self.position) * 2 - 1) * step_size
        new_position = torch.clamp(new_position, min_b, max_b)
        
        # Evaluate new solution
        new_fitness = self._evaluate(new_position.unsqueeze(0)).squeeze()
        
        # Decide whether to accept the new solution
        delta = new_fitness - self.fitness
        if delta < 0 or torch.rand(1).item() < torch.exp(-delta / self.temp):
            self.position = new_position
            self.fitness = new_fitness
            
            # Update best solution if needed
            if new_fitness < self.best_fitness:
                self.best_position = new_position.clone()
                self.best_fitness = new_fitness.clone()
        
        # Cool down temperature
        self.temp = max(self.temp * self.parameters['cooling_rate'], self.parameters['temp_end'])
    
    def _get_best(self):
        return self.best_position, self.best_fitness
```

## GPU Acceleration

TorchOptLib leverages PyTorch's GPU capabilities to accelerate optimization algorithms. Using GPU acceleration is straightforward:

### Basic GPU Usage

```python
# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize your optimizer with the device parameter
optimizer = MyOptimizer(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters={'param1': 1.0, 'param2': 2.0},
    device=device  # This tells the optimizer to use GPU if available
)

# Run optimization - it will automatically use the GPU
best_solution, best_fitness = optimizer.optimize()
```

### Creating GPU-Compatible Extensions

When implementing custom functions or optimizers, ensure they work with GPU by:

1. Creating tensors on the specified device:

```python
def initialize(self):
    min_b, max_b = self.test_function.bounds
    # Create tensors directly on the specified device
    self.population = torch.rand((self.population_size, self.test_function.dim), 
                                device=self.device) * (max_b - min_b) + min_b
```

2. Moving existing tensors to the right device:

```python
# If you have existing tensors
my_tensor = my_tensor.to(self.device)
```

That's all you need for basic GPU acceleration! For more advanced GPU techniques such as multi-GPU training, mixed precision training, or custom CUDA kernels, please refer to the PyTorch official documentation:

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [Distributed Training](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)

## Testing Your Extensions

Once you've created your custom components, you can test them with the rest of the library:

```python
from torchoptlib.benchmarks import classic
import torch

# Use your custom function
dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
test_function = MyCustomFunction(dim=dim, bounds=bounds)

# Use your custom optimizer
myopt = MyOptimizer(
    test_function=test_function,
    population_size=30,
    max_iter=100,
    parameters={'param1': 1.0, 'param2': 2.0},
)

# Run optimization
best_solution, best_fitness = myopt.optimize()
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")
```

## Best Practices

1. **Vectorization**: Use PyTorch's tensor operations instead of loops for better performance
2. **GPU Support**: Ensure your code works with both CPU and GPU tensors
3. **Error Handling**: Add appropriate validation for parameters and inputs
4. **Documentation**: Add docstrings to explain your implementation and parameters
5. **Testing**: Create tests to verify your implementation works correctly
