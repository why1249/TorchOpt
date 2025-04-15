from core.base import TestFunction
import torch

class Sphere(TestFunction):
    """Sphere function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x ** 2, dim=-1)
    
class Rastrigin(TestFunction):
    """Rastrigin function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        A = 10
        return A * self.dim + torch.sum(x ** 2 - A * torch.cos(2 * torch.pi * x), dim=-1)
    
class Rosenbrock(TestFunction):
    """Rosenbrock function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1 - x[:, :-1]) ** 2, dim=-1)
    
class Griewank(TestFunction):
    """Griewank function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        sum_term = torch.sum(x ** 2, dim=-1) / 4000
        prod_term = torch.prod(torch.cos(x / torch.sqrt(torch.arange(1, self.dim + 1).to(self.device))), dim=-1)
        return sum_term - prod_term + 1
    
class Ackley(TestFunction):
    """Ackley function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        sum_term = torch.sum(x ** 2, dim=-1)
        cos_term = torch.sum(torch.cos(2 * torch.pi * x), dim=-1)
        return -20 * torch.exp(-0.2 * torch.sqrt(sum_term / self.dim)) - torch.exp(cos_term / self.dim) + 20 + torch.e
    
class Schwefel(TestFunction):
    """Schwefel function"""
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return 418.9829 * self.dim - torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))), dim=-1)