# 算法

TorchOptLib 提供了多种优化算法的实现，并支持用户自定义新的优化器。

## 已实现的算法

当前库中已实现的优化算法包括：

- **PSO (粒子群优化算法)**: 一种基于群体的随机优化技术
- **CMA-ES (协方差矩阵自适应进化策略)**: 一种适用于困难非线性非凸优化问题的进化算法

每个算法都可以通过简单的接口调用，并支持GPU加速。

## 算法使用示例

```python
from torchoptlib.algorithm import pso
from torchoptlib.benchmarks import classic
import torch

# 定义测试函数
dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
test_function = classic.Rastrigin(dim=dim, bounds=bounds)

# 设置PSO参数
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
```

## 自定义优化器

您可以通过继承 `Optimizer` 基类来创建自己的优化算法：

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
        # 初始化您的算法
        # 例如：生成初始种群、设置初始状态等
        pass
    
    def update(self):
        # 实现单次迭代的更新逻辑
        # 这是算法的核心部分
        pass
        
    def _get_best(self):
        # 返回最佳解和适应度
        # 例如：return self.global_best_position, self.global_best_fitness
        pass
```

创建自定义优化器时，必须实现以下三个方法：

- `initialize()`: 初始化算法所需的所有变量和状态
- `update()`: 包含算法的单次迭代逻辑
- `_get_best()`: 返回当前最佳解和对应的适应度值

所有这些方法都应该利用PyTorch的张量操作以支持GPU加速。

## 参数配置

大多数优化算法需要特定的参数来控制其行为。在初始化优化器时，可以通过`parameters`字典传递这些参数：

```python
parameters = {
    'param1': value1,
    'param2': value2,
    # ...更多参数
}

optimizer = MyOptimizer(
    test_function=function,
    population_size=50,
    max_iter=200,
    parameters=parameters
)
```

请参考各个算法的文档以了解其特定参数。
