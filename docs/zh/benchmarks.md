# 基准测试函数

TorchOptLib 包含几个经典的基准测试函数，这些函数通常用于评估优化算法。这些函数具有已知的特性，有助于评估算法在不同场景下的性能。

## 可用函数

### Sphere 函数（球函数）

一个简单的单峰函数，连续、凸且可分离。它通常作为测试优化算法的基准。

```python
from torchoptlib.benchmarks import classic
import torch

dim = 10
bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
sphere = classic.Sphere(dim=dim, bounds=bounds)
```

数学公式：$f(x) = \sum_{i=1}^{n} x_i^2$

### Rastrigin 函数

一个高度多峰的函数，具有许多按规则排列的局部最小值。对于许多优化算法来说，这是一个具有挑战性的函数。

```python
rastrigin = classic.Rastrigin(dim=dim, bounds=bounds)
```

数学公式：$f(x) = 10n + \sum_{i=1}^{n} [x_i^2 - 10\cos(2\pi x_i)]$

### Rosenbrock 函数（香蕉函数）

一个从局部最优到全局最优有狭窄谷的函数。找到山谷很容易，但收敛到全局最优困难。

```python
rosenbrock = classic.Rosenbrock(dim=dim, bounds=bounds)
```

数学公式：$f(x) = \sum_{i=1}^{n-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]$

### Griewank 函数

一个变量间相互依赖的多峰函数。由于余弦项，它有许多局部最小值。

```python
griewank = classic.Griewank(dim=dim, bounds=bounds)
```

数学公式：$f(x) = 1 + \frac{1}{4000}\sum_{i=1}^{n} x_i^2 - \prod_{i=1}^{n} \cos(\frac{x_i}{\sqrt{i}})$

### Ackley 函数

一个有许多局部最小值但只有一个全局最小值的函数。它结合了指数项和余弦调制。

```python
ackley = classic.Ackley(dim=dim, bounds=bounds)
```

数学公式：$f(x) = -20\exp(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}) - \exp(\frac{1}{n}\sum_{i=1}^{n} \cos(2\pi x_i)) + 20 + e$

### Schwefel 函数

一个全局最小值远离中心并靠近边界的复杂函数。由于其深度局部最小值，它具有挑战性。

```python
schwefel = classic.Schwefel(dim=dim, bounds=bounds)
```

数学公式：$f(x) = 418.9829n - \sum_{i=1}^{n} x_i\sin(\sqrt{|x_i|})$

## 创建自定义基准函数

您可以通过扩展 `TestFunction` 基类来创建自己的基准函数：

```python
from torchoptlib.core.base import TestFunction
import torch

class MyCustomFunction(TestFunction):
    def __init__(self, dim: int, bounds: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dim, bounds)
    
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        # 在此实现您的函数
        return torch.sum(x ** 2, dim=-1)  # 示例：Sphere 函数
```

`evaluate` 方法应接受形状为 `(batch_size, dim)` 的张量，并返回形状为 `(batch_size,)` 的张量，其中包含函数值。
