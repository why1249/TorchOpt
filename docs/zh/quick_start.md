# 快速入门指南

本指南将通过一个简单的示例帮助您快速上手 TorchOptLib。

## 基本使用示例

以下示例演示了如何使用粒子群优化（PSO）算法优化 Rastrigin 函数：

```python
from torchoptlib.benchmarks import classic
from torchoptlib.algorithm import pso
import torch

# 定义测试函数
dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
test_function = classic.Rastrigin(dim=dim, bounds=bounds)

# 设置 PSO 参数
parameters = {
    'c1': 1.5,  # 认知参数
    'c2': 1.5,  # 社会参数
    'w': 0.5,   # 惯性权重
}

# 初始化优化器
pso_instance = pso.PSO(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters=parameters,
    print_interval=10,
)

# 运行优化
best_solution, best_fitness = pso_instance.optimize()

print(f"最佳解决方案: {best_solution}")
print(f"最佳适应度: {best_fitness}")
```

## 逐步解析

1. **导入必要模块**：从各自的模块导入基准函数和优化算法。

2. **定义测试函数**：创建具有适当维度和边界的基准函数。

3. **设置算法参数**：配置特定于优化算法的参数。

4. **初始化优化器**：使用测试函数和参数创建优化器实例。

5. **运行优化**：调用 `optimize()` 方法开始优化过程。

6. **获取结果**：获取最佳解决方案及其适应度值。

## 使用 GPU 加速

要使用 GPU 加速，只需在创建优化器时指定设备：

```python
pso_instance = pso.PSO(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters=parameters,
    print_interval=10,
    device='cuda'  # 如果可用，使用 GPU
)
```

确保您有兼容 CUDA 的 GPU 和已安装的 CUDA 工具包。

## 后续步骤

在熟悉基本用法后，您可以：

- 探索不同的[基准测试函数](benchmarks.md)
- 尝试其他[优化算法](algorithms.md)
- 学习如何使用您自己的函数和算法[扩展 TorchOptLib](extending.md)
