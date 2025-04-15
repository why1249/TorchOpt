import unittest
import torch
import numpy as np
from torchoptlib.benchmarks.classic import Sphere, Rastrigin, Rosenbrock, Griewank, Ackley, Schwefel

class TestBenchmarks(unittest.TestCase):
    def setUp(self):
        self.dim = 10
        self.bounds = (torch.tensor([-10.0] * self.dim), torch.tensor([10.0] * self.dim))
        self.device = torch.device('cpu')
        
        # 创建测试点
        self.zero_point = torch.zeros((1, self.dim), dtype=torch.float64)
        self.random_point = torch.rand((1, self.dim), dtype=torch.float64) * 20 - 10
        self.batch_points = torch.rand((5, self.dim), dtype=torch.float64) * 20 - 10
    
    def test_sphere(self):
        func = Sphere(dim=self.dim, bounds=self.bounds)
        
        # 原点的值应为0
        result = func(self.zero_point)
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
        
        # 手动计算一个点的值进行比较
        point = self.random_point[0]
        expected = torch.sum(point ** 2).item()
        actual = func(self.random_point).item()
        self.assertAlmostEqual(actual, expected, places=6)
    
    def test_rastrigin(self):
        func = Rastrigin(dim=self.dim, bounds=self.bounds)
        
        # 原点的值应为0
        result = func(self.zero_point)
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
        
        # 手动计算一个点的值进行比较
        point = self.random_point[0]
        expected = 10 * self.dim + torch.sum(point ** 2 - 10 * torch.cos(2 * torch.pi * point)).item()
        actual = func(self.random_point).item()
        self.assertAlmostEqual(actual, expected, places=6)
    
    def test_rosenbrock(self):
        func = Rosenbrock(dim=self.dim, bounds=self.bounds)
        
        # 全1向量的值应为0
        ones = torch.ones((1, self.dim), dtype=torch.float64)
        result = func(ones)
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
    
    def test_griewank(self):
        func = Griewank(dim=self.dim, bounds=self.bounds)
        
        # 原点的值应为0
        result = func(self.zero_point)
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
    
    def test_ackley(self):
        func = Ackley(dim=self.dim, bounds=self.bounds)
        
        # 原点的值应为0
        result = func(self.zero_point)
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
    
    def test_schwefel(self):
        func = Schwefel(dim=self.dim, bounds=self.bounds)
        
        # 批处理测试
        results = func(self.batch_points)
        self.assertEqual(results.shape, torch.Size([5]))
        
        # 全部为420.9687的向量应该接近全局最优
        x_opt = torch.full((1, self.dim), 420.9687, dtype=torch.float64)
        result = func(x_opt)
        self.assertLess(result.item(), 1e-3)

if __name__ == '__main__':
    unittest.main()
