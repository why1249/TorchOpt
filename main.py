from torchopt.benchmarks import classic
from torchopt.algorithm import pso
import torchopt
import torch
if __name__ == "__main__":
    # 测试函数
    dim = 10
    print(torchopt.__file__)
    bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
    test_function = classic.Rastrigin(dim=10, bounds=bounds)
    
    # PSO参数
    parameters = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.5,
    }
    
    # PSO实例化
    pso_instance = pso.PSO(
        test_function=test_function,
        population_size=50,
        max_iter=100,
        parameters=parameters,
        print_interval=10,
    )
    
    # 执行优化
    best_solution, best_fitness = pso_instance.optimize()
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")