# 扩展 TorchOptLib

TorchOptLib 被设计为易于扩展的库。您可以创建自己的测试函数和优化算法以满足特定需求。

## 创建自定义测试函数

要创建自定义测试函数，您需要继承 `TestFunction` 基类并实现所需的方法：

```python
from torchoptlib.core.base import TestFunction
import torch

class MyCustomFunction(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
        # 在这里初始化任何额外的属性
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # 在这里实现您的函数
        # x 的形状为 (batch_size, dim)
        # 返回形状为 (batch_size,) 的张量
        return torch.sum(x ** 2, dim=-1)  # 示例：球函数
```

### 关键组件：

1. **构造函数**：使用维度和边界初始化您的函数
2. **evaluate()**：计算给定点的函数值的核心方法
   - 输入：形状为 `(batch_size, dim)` 的张量
   - 输出：形状为 `(batch_size,)` 的张量

### 示例：Rastrigin 函数的自定义变体

```python
class CustomRastrigin(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor], A: float = 10.0):
        super().__init__(dim, bounds)
        self.A = A
        
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        n = self.dim
        return self.A * n + torch.sum(x**2 - self.A * torch.cos(2 * torch.pi * x), dim=-1)
```

## 创建自定义优化算法

要创建自定义优化器，扩展 `Optimizer` 基类并实现所需的方法：

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
        # 初始化算法状态
        # 示例：生成初始种群
        min_b, max_b = self.test_function.bounds
        self.positions = torch.rand((self.population_size, self.test_function.dim),
                                   device=self.device) * (max_b - min_b) + min_b
        # ... 初始化其他状态变量
    
    def update(self):
        # 实现一次算法迭代
        # 此方法将在每次迭代中被调用
        # 示例：根据算法逻辑更新位置
        pass
        
    def _get_best(self):
        # 返回最佳解决方案和适应度
        # 示例：找到具有最小适应度的个体
        best_idx = torch.argmin(self.fitness)
        return self.positions[best_idx], self.fitness[best_idx]
```

### 关键组件：

1. **构造函数**：使用问题参数初始化优化器
2. **initialize()**：设置算法的初始状态
3. **update()**：实现优化过程的一次迭代
4. **_get_best()**：返回当前最佳解决方案及其适应度值

### 示例：简化的模拟退火算法

```python
class SimulatedAnnealing(Optimizer):
    def __init__(self, test_function: TestFunction,
                 population_size: int = 1,  # SA 通常只使用一个解决方案
                 max_iter: int = 1000,
                 parameters: dict = {'temp_start': 1.0, 'temp_end': 0.01, 'cooling_rate': 0.95},
                 print_interval: int = 10,
                 device: str = 'cpu'):
        super().__init__(test_function, population_size, max_iter, parameters, print_interval, device)
    
    def initialize(self):
        min_b, max_b = self.test_function.bounds
        # 初始化单个位置
        self.position = torch.rand((self.test_function.dim,), device=self.device) * (max_b - min_b) + min_b
        self.fitness = self._evaluate(self.position.unsqueeze(0)).squeeze()
        
        self.best_position = self.position.clone()
        self.best_fitness = self.fitness.clone()
        
        # 初始化温度
        self.temp = self.parameters['temp_start']
        
    def update(self):
        min_b, max_b = self.test_function.bounds
        
        # 生成具有小扰动的新候选解
        step_size = 0.1 * (max_b - min_b)
        new_position = self.position + (torch.rand_like(self.position) * 2 - 1) * step_size
        new_position = torch.clamp(new_position, min_b, max_b)
        
        # 评估新解
        new_fitness = self._evaluate(new_position.unsqueeze(0)).squeeze()
        
        # 决定是否接受新解
        delta = new_fitness - self.fitness
        if delta < 0 or torch.rand(1).item() < torch.exp(-delta / self.temp):
            self.position = new_position
            self.fitness = new_fitness
            
            # 如有必要，更新最佳解
            if new_fitness < self.best_fitness:
                self.best_position = new_position.clone()
                self.best_fitness = new_fitness.clone()
        
        # 降低温度
        self.temp = max(self.temp * self.parameters['cooling_rate'], self.parameters['temp_end'])
    
    def _get_best(self):
        return self.best_position, self.best_fitness
```

## 测试您的扩展

创建自定义组件后，您可以使用库的其余部分进行测试：

```python
from torchoptlib.benchmarks import classic
import torch

# 使用您的自定义函数
dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
test_function = MyCustomFunction(dim=dim, bounds=bounds)

# 使用您的自定义优化器
myopt = MyOptimizer(
    test_function=test_function,
    population_size=30,
    max_iter=100,
    parameters={'param1': 1.0, 'param2': 2.0},
)

# 运行优化
best_solution, best_fitness = myopt.optimize()
print(f"最佳解决方案: {best_solution}")
print(f"最佳适应度: {best_fitness}")
```

## GPU 加速

TorchOptLib 利用 PyTorch 的 GPU 功能来加速优化算法。使用 GPU 加速非常简单：

### 基本 GPU 使用

```python
# 检查 CUDA 是否可用
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用 device 参数初始化优化器
optimizer = MyOptimizer(
    test_function=test_function,
    population_size=50,
    max_iter=100,
    parameters={'param1': 1.0, 'param2': 2.0},
    device=device  # 这告诉优化器在可用时使用 GPU
)

# 运行优化 - 它会自动使用 GPU
best_solution, best_fitness = optimizer.optimize()
```

### 创建 GPU 兼容的扩展

在实现自定义函数或优化器时，确保它们与 GPU 兼容：

1. 在指定设备上创建张量：

```python
def initialize(self):
    min_b, max_b = self.test_function.bounds
    # 直接在指定设备上创建张量
    self.population = torch.rand((self.population_size, self.test_function.dim), 
                                device=self.device) * (max_b - min_b) + min_b
```

2. 将现有张量移动到正确的设备：

```python
# 如果您有现有的张量
my_tensor = my_tensor.to(self.device)
```

这就是基本 GPU 加速所需的全部内容！对于更高级的 GPU 技术，如多 GPU 训练、混合精度训练或自定义 CUDA 内核，请参考 PyTorch 官方文档：

- [PyTorch CUDA 语义](https://pytorch.org/docs/stable/notes/cuda.html)
- [分布式训练](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [自动混合精度](https://pytorch.org/docs/stable/amp.html)

## 最佳实践

1. **向量化**：使用 PyTorch 的张量操作而非循环，以获得更好的性能
2. **GPU 支持**：确保您的代码同时适用于 CPU 和 GPU 张量
3. **错误处理**：为参数和输入添加适当的验证
4. **文档**：添加文档字符串以解释您的实现和参数
5. **测试**：创建测试以验证您的实现是否正确工作
