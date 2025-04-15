import unittest
import torch
from torchopt.core.base import TestFunction

class MockTestFunction(TestFunction):
    """A mock implementation of TestFunction for testing"""
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x, dim=-1)

class TestCore(unittest.TestCase):
    def test_test_function_base(self):
        # 检查对错误边界的验证
        dim = 5
        
        # 正确的边界情况
        bounds = (torch.tensor([-10.0] * dim), torch.tensor([10.0] * dim))
        func = MockTestFunction(dim=dim, bounds=bounds)
        self.assertEqual(func.dim, dim)
        self.assertTrue(torch.all(func.min_bounds == bounds[0]))
        self.assertTrue(torch.all(func.max_bounds == bounds[1]))
        
        # 测试函数调用
        x = torch.ones((3, dim))
        result = func(x)
        self.assertEqual(result.shape, torch.Size([3]))
        self.assertTrue(torch.all(result == dim))
        
        # 测试属性
        self.assertEqual(func.name, "MockTestFunction")
        min_b, max_b = func.bounds
        self.assertTrue(torch.all(min_b == bounds[0]))
        self.assertTrue(torch.all(max_b == bounds[1]))
        
        # 错误的边界维度
        with self.assertRaises(AssertionError):
            wrong_bounds = (torch.tensor([-10.0] * (dim-1)), torch.tensor([10.0] * dim))
            MockTestFunction(dim=dim, bounds=wrong_bounds)
        
        # 无效的边界值（最小值大于最大值）
        with self.assertRaises(AssertionError):
            wrong_bounds = (torch.tensor([10.0] * dim), torch.tensor([-10.0] * dim))
            MockTestFunction(dim=dim, bounds=wrong_bounds)

if __name__ == '__main__':
    unittest.main()
