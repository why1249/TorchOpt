from core.base import *
import torch

class PSO(Optimizer):
    def __init__(self, test_function: TestFunction,
                population_size: int=50,
                max_iter: int=1000,
                parameters: dict={'w': 0.5, 'c1': 1.5, 'c2': 1.5},
                print_interval=10,
                device='cpu'):
        super().__init__(test_function, population_size, max_iter, parameters, print_interval, device)

        
    def initialize(self):
        min_b, max_b = self.test_function.bounds
        self.positions = torch.rand((self.population_size, self.test_function.dim), dtype=torch.float64,
                               device=self.device) * (max_b - min_b) + min_b
        self.velocities = torch.zeros_like(self.positions, dtype=torch.float64, device=self.device)
        self.personal_best = self.positions.clone()
        self.personal_best_fitness = self._evaluate(self.positions)
        best = self._get_best()
        self.global_best = best[0].clone()
        self.global_best_fitness = best[1].clone()
        
    def update(self):
        w, c1, c2 = self.parameters['w'], self.parameters['c1'], self.parameters['c2']
        r1, r2 = torch.rand(2, *self.positions.shape)
        
        # 更新速度
        self.velocities = (w * self.velocities +
                           c1 * r1 * (self.personal_best - self.positions) +
                           c2 * r2 * (self.global_best - self.positions))
        
        # 更新位置
        self.positions += self.velocities
        
        # 边界处理
        min_b, max_b = self.test_function.bounds
        self.positions = torch.clamp(self.positions, min_b, max_b)
        
        # 评估
        fitness = self._evaluate(self.positions)
        
        # 更新个体最优
        update_mask = fitness < self.personal_best_fitness
        self.personal_best[update_mask] = self.positions[update_mask]
        self.personal_best_fitness[update_mask] = fitness[update_mask]

        # 更新全局最优
        best = self._get_best()
        if best[1] < self.global_best_fitness:
            self.global_best = best[0].clone()
            self.global_best_fitness = best[1].clone()
        
    def _get_best(self):
        best_idx = torch.argmin(self.personal_best_fitness)
        return self.personal_best[best_idx], self.personal_best_fitness[best_idx]
    
    def _evaluate(self, positions):
        return self.test_function(positions)