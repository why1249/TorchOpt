from torchoptlib.core.base import *
import torch
import numpy as np

class DE(Optimizer):
    """
    Differential Evolution (DE) algorithm implementation.
    
    DE is a stochastic population-based optimization algorithm that is simple yet powerful
    for global optimization over continuous spaces.
    
    Parameters:
        test_function (TestFunction): The function to be optimized.
        population_size (int): The number of individuals in the population.
        max_iter (int): The maximum number of iterations.
        parameters (dict): Parameters of the DE algorithm:
            - F (float): Differential weight (scaling factor), typically in [0.5, 1.0].
            - CR (float): Crossover probability, typically in [0.8, 1.0].
            - strategy (str): DE variant to use ('rand/1/bin', 'best/1/bin', 'rand/2/bin', etc.).
        print_interval (int): The interval of printing the best fitness.
        device (str): The device to run the algorithm on ('cpu' or 'cuda').
    """
    def __init__(self, test_function: TestFunction,
                population_size: int=50,
                max_iter: int=1000,
                parameters: dict={'F': 0.8, 'CR': 0.9, 'strategy': 'rand/1/bin'},
                print_interval: int=10,
                device: str='cpu'):
        super().__init__(test_function, population_size, max_iter, parameters, print_interval, device)
        
        # Validate and set default parameters
        self.F = self.parameters.get('F', 0.8)
        self.CR = self.parameters.get('CR', 0.9)
        self.strategy = self.parameters.get('strategy', 'rand/1/bin')
        
        # Parse strategy
        strategy_parts = self.strategy.split('/')
        if len(strategy_parts) != 3:
            raise ValueError(f"Invalid strategy: {self.strategy}. Format should be 'x/y/z'.")
            
        self.base_vector_selection = strategy_parts[0]  # 'rand' or 'best'
        self.n_difference_vectors = int(strategy_parts[1])  # Typically 1 or 2
        self.crossover_type = strategy_parts[2]  # 'bin' (binomial) or 'exp' (exponential)
        
        if self.crossover_type not in ['bin', 'exp']:
            raise ValueError(f"Unsupported crossover type: {self.crossover_type}")
        
    def initialize(self):
        # Initialize population randomly within bounds
        min_b, max_b = self.test_function.bounds
        self.population = torch.rand((self.population_size, self.test_function.dim), 
                                    dtype=torch.float64, device=self.device) * (max_b - min_b) + min_b
        
        # Evaluate initial population
        self.fitness = self._evaluate(self.population)
        
        # Keep track of best solution
        best_idx = torch.argmin(self.fitness)
        self.best_solution = self.population[best_idx].clone()
        self.best_fitness = self.fitness[best_idx].clone()
        
    def update(self):
        # Create trial population
        trial_pop = torch.zeros_like(self.population)
        
        # For each individual in the population
        for i in range(self.population_size):
            # Select base vector
            if self.base_vector_selection == 'rand':
                # Random individual from population
                r0 = torch.randint(0, self.population_size, (1,)).item()
                while r0 == i:
                    r0 = torch.randint(0, self.population_size, (1,)).item()
                base = self.population[r0]
            else:  # 'best'
                # Current best individual
                base = self.best_solution
            
            # Generate difference vectors
            diff_vectors = []
            for _ in range(self.n_difference_vectors):
                # Select two random distinct individuals
                r1, r2 = torch.randint(0, self.population_size, (2,)).tolist()
                while r1 == r2 or r1 == i or r2 == i:
                    r1, r2 = torch.randint(0, self.population_size, (2,)).tolist()
                
                # Create difference vector
                diff_vectors.append(self.population[r1] - self.population[r2])
            
            # Create mutant vector by adding weighted differences to base vector
            mutant = base.clone()
            for diff in diff_vectors:
                mutant += self.F * diff
            
            # Apply crossover (recombination)
            if self.crossover_type == 'bin':
                # Binomial crossover
                j_rand = torch.randint(0, self.test_function.dim, (1,)).item()
                cross_points = torch.rand(self.test_function.dim, device=self.device) < self.CR
                cross_points[j_rand] = True  # Ensure at least one parameter is changed
                
                trial = torch.where(cross_points, mutant, self.population[i])
            else:  # 'exp'
                # Exponential crossover
                trial = self.population[i].clone()
                j = torch.randint(0, self.test_function.dim, (1,)).item()
                L = 0
                
                while L < self.test_function.dim and torch.rand(1).item() < self.CR:
                    trial[j] = mutant[j]
                    j = (j + 1) % self.test_function.dim
                    L += 1
            
            # Apply bounds
            min_b, max_b = self.test_function.bounds
            trial = torch.clamp(trial, min_b, max_b)
            
            trial_pop[i] = trial
        
        # Evaluate trial population
        trial_fitness = self._evaluate(trial_pop)
        
        # Selection - keep better solutions
        improvements = trial_fitness < self.fitness
        self.population[improvements] = trial_pop[improvements]
        self.fitness[improvements] = trial_fitness[improvements]
        
        # Update best solution
        best_idx = torch.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_solution = self.population[best_idx].clone()
            self.best_fitness = self.fitness[best_idx].clone()
            
    def _get_best(self):
        return self.best_solution, self.best_fitness
        
    def _evaluate(self, positions):
        return self.test_function(positions)
    