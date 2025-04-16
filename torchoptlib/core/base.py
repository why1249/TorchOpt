import torch
from abc import ABC, abstractmethod
import time
class TestFunction(ABC):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            dim: the dimension of the function
            bounds: (min_bounds, max_bounds) the bounds of the function
        """
        self.dim = dim
        self.min_bounds, self.max_bounds = bounds
        self._validate_bounds()
        
    def _validate_bounds(self):
        assert self.min_bounds.shape == (self.dim,)
        assert self.max_bounds.shape == (self.dim,)
        assert torch.all(self.min_bounds < self.max_bounds)
    
    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """the function to be optimized"""
        pass
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.evaluate(x)
    
    @property
    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.min_bounds, self.max_bounds)
    
    @property
    def name(self) -> str:
        return self.__class__.__name__

class Optimizer(ABC):
    def __init__(self, 
                 test_function: TestFunction,
                 population_size: int,
                 max_iter: int,
                 parameters: dict,
                 print_interval: int = 10,
                 device: str = 'cpu',):
        """
        Args:
            test_function: the function to be optimized
            population_size: the number of individuals in the population
            max_iter: the maximum number of iterations
            parameters: the parameters of the algorithm
            print_interval: the interval of printing the best fitness
            device: the device to run the algorithm on
        """
        self.test_function = test_function
        self.population_size = population_size
        self.max_iter = max_iter
        self.parameters = parameters
        self.device = device
        self.history = []
        self.print_interval = print_interval
        
    @abstractmethod
    def initialize(self):
        """initialize the population and parameters"""
        pass
    
    @abstractmethod
    def update(self):
        """update the population and parameters"""
        pass
    
    def optimize(self) -> tuple[torch.Tensor, float]:
        """Optimize the function"""
        start_time = time.time()
        self.initialize()
        
        for iter in range(self.max_iter):
            self.update()
            best_solution, best_fitness = self._get_best()
            self.history.append(best_fitness.item())
            
            if (iter + 1) % self.print_interval == 0 or iter == self.max_iter - 1:
                elapsed_time = time.time() - start_time
                print(f"Iteration {iter + 1}/{self.max_iter}: Best fitness = {best_fitness.item()}, Elapsed time = {elapsed_time:.2f}s")
        
        print(f"Optimization finished in {time.time() - start_time:.2f}s")
        return best_solution, best_fitness
    
    def _get_best(self) -> tuple[torch.Tensor, float]:
        """"Get the best solution and fitness"""
        pass
    
    @property
    def algorithm(self) -> str:
        return self.__class__.__name__