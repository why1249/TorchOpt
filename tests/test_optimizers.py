import unittest
import torch
import numpy as np
from torchopt.benchmarks.classic import Sphere, Rastrigin
from torchopt.algorithm.pso import PSO
from torchopt.algorithm.cma_es import CMAES

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        # 创建简单函数和较小的维度用于快速测试
        self.dim = 2
        self.bounds = (torch.tensor([-5.0] * self.dim), torch.tensor([5.0] * self.dim))
        self.sphere = Sphere(dim=self.dim, bounds=self.bounds)
        self.rastrigin = Rastrigin(dim=self.dim, bounds=self.bounds)
        
        # 设置较小的迭代次数，以加快测试速度
        self.max_iter = 20
        self.population_size = 20
        self.print_interval = 100
        
    def test_pso_initialization(self):
        # 测试PSO初始化
        pso = PSO(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'w': 0.5, 'c1': 1.5, 'c2': 1.5},
            print_interval=self.print_interval
        )
        
        pso.initialize()
        
        # 检查初始化后的属性
        self.assertEqual(pso.positions.shape, (self.population_size, self.dim))
        self.assertEqual(pso.velocities.shape, (self.population_size, self.dim))
        self.assertEqual(pso.personal_best.shape, (self.population_size, self.dim))
        self.assertEqual(pso.personal_best_fitness.shape, (self.population_size,))
        self.assertEqual(pso.global_best.shape, (self.dim,))
        
        # 检查粒子是否在界限内
        min_b, max_b = self.bounds
        self.assertTrue(torch.all(pso.positions >= min_b))
        self.assertTrue(torch.all(pso.positions <= max_b))
    
    def test_pso_update(self):
        # 测试PSO更新
        pso = PSO(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'w': 0.5, 'c1': 1.5, 'c2': 1.5},
            print_interval=self.print_interval
        )
        
        pso.initialize()
        old_best_fitness = pso.global_best_fitness.clone()
        
        # 执行一次更新
        pso.update()
        
        # 检查是否更新了粒子位置
        self.assertEqual(pso.positions.shape, (self.population_size, self.dim))
        self.assertEqual(pso.velocities.shape, (self.population_size, self.dim))
        
        # 检查粒子是否仍在界限内
        min_b, max_b = self.bounds
        self.assertTrue(torch.all(pso.positions >= min_b))
        self.assertTrue(torch.all(pso.positions <= max_b))
        
        # 检查全局最优是否有改进（或至少不变差）
        self.assertTrue(pso.global_best_fitness <= old_best_fitness)
    
    def test_pso_optimize(self):
        # 测试PSO优化过程
        pso = PSO(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'w': 0.5, 'c1': 1.5, 'c2': 1.5},
            print_interval=self.print_interval
        )
        
        solution, fitness = pso.optimize()
        
        # 检查最终解决方案
        self.assertEqual(solution.shape, (self.dim,))
        self.assertTrue(isinstance(fitness.item(), float))
        
        # 检查历史记录
        self.assertEqual(len(pso.history), self.max_iter)
        
        # 对于球函数，最终结果应该接近零（但由于迭代次数少，可能不会非常接近）
        if self.sphere == pso.test_function:
            self.assertLess(fitness.item(), pso.history[0])
    
    def test_cmaes_initialization(self):
        # 测试CMAES初始化
        cmaes = CMAES(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'sigma': 0.3, 'mu_factor': 0.5},
            print_interval=self.print_interval
        )
        
        cmaes.initialize()
        
        # 检查初始化后的属性
        self.assertEqual(cmaes.mean.shape, (self.dim,))
        self.assertEqual(cmaes.pc.shape, (self.dim,))
        self.assertEqual(cmaes.ps.shape, (self.dim,))
        self.assertEqual(cmaes.C.shape, (self.dim, self.dim))
        self.assertEqual(cmaes.B.shape, (self.dim, self.dim))
        self.assertEqual(cmaes.D.shape, (self.dim,))
        
        # 检查协方差矩阵是否是正定的
        eigenvalues = torch.linalg.eigvalsh(cmaes.C)
        self.assertTrue(torch.all(eigenvalues > 0))
    
    def test_cmaes_update(self):
        # 测试CMAES更新
        cmaes = CMAES(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'sigma': 0.3, 'mu_factor': 0.5},
            print_interval=self.print_interval
        )
        
        cmaes.initialize()
        
        # 执行一次更新
        cmaes.update()
        
        # 检查解决方案
        self.assertEqual(cmaes.solutions.shape, (self.population_size, self.dim))
        self.assertEqual(cmaes.fitness.shape, (self.population_size,))
        
        # 检查解决方案是否在界限内
        min_b, max_b = self.bounds
        self.assertTrue(torch.all(cmaes.solutions >= min_b))
        self.assertTrue(torch.all(cmaes.solutions <= max_b))
        
        # 检查协方差矩阵是否仍然是对称的
        diff = torch.abs(cmaes.C - cmaes.C.T)
        self.assertTrue(torch.all(diff < 1e-6))
    
    def test_cmaes_optimize(self):
        # 测试CMAES优化过程
        cmaes = CMAES(
            test_function=self.sphere,
            population_size=self.population_size,
            max_iter=self.max_iter,
            parameters={'sigma': 0.3, 'mu_factor': 0.5},
            print_interval=self.print_interval
        )
        
        solution, fitness = cmaes.optimize()
        
        # 检查最终解决方案
        self.assertEqual(solution.shape, (self.dim,))
        self.assertTrue(isinstance(fitness.item(), float))
        
        # 检查历史记录
        self.assertEqual(len(cmaes.history), self.max_iter)
        
        # 最终结果应该比初始结果更好
        self.assertLess(fitness.item(), cmaes.history[0])

if __name__ == '__main__':
    unittest.main()
