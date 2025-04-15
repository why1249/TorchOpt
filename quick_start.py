from torchoptlib.benchmarks import classic
from torchoptlib.algorithm import pso, cma_es
import torchoptlib
import torch
print(torch.__version__)
if __name__ == "__main__":
    # Test function
    dim = 100
    print(torchoptlib.__file__)
    bounds = (torch.tensor([-5.12] * dim), torch.tensor([5.12] * dim))
    test_function = classic.Rastrigin(dim=dim, bounds=bounds)
    
    # PSO
    parameters = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.5,
    }
    
    pso_instance = pso.PSO(
        test_function=test_function,
        population_size=50,
        max_iter=1000,
        parameters=parameters,
        print_interval=100,
    )
    
    best_solution, best_fitness = pso_instance.optimize()
    
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    # CMA-ES
    parameters = {
        'sigma': 0.3,
        'mu_factor': 0.5,
    }

    cma_es_instance = cma_es.CMAES(
        test_function=test_function,
        population_size=50,
        max_iter=1000,
        parameters=parameters,
        print_interval=100,
    )

    best_solution, best_fitness = cma_es_instance.optimize()

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")