import torch
from abc import ABC, abstractmethod

class TestFunction(ABC):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        """
        Args:
            dim: 问题维度
            bounds: (min_bounds, max_bounds) 每个维度的上下界
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
        """评估函数，支持批量计算"""
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
            test_function: 测试函数实例
            population_size: 种群大小
            max_iter: 最大迭代次数
            parameters: 算法特定参数
            n_workers: 并行工作数
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
        """初始化种群"""
        pass
    
    @abstractmethod
    def update(self):
        """单次迭代更新"""
        pass
    
    def optimize(self) -> tuple[torch.Tensor, float]:
        """执行完整优化过程"""
        self.initialize()
        
        for iter in range(self.max_iter):
            self.update()
            best_solution, best_fitness = self._get_best()
            self.history.append(best_fitness.item())
            
            if (iter + 1) % self.print_interval == 0 or iter == self.max_iter - 1:
                print(f"Iteration {iter + 1}/{self.max_iter}: Best fitness = {best_fitness.item()}")
                
        return best_solution, best_fitness
    
    def _get_best(self) -> tuple[torch.Tensor, float]:
        """获取当前最优解和适应度"""
        pass
    
    @property
    def algorithm(self) -> str:
        return self.__class__.__name__