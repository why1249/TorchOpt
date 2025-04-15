from torchopt.core.base import *
import torch
import numpy as np

class CMAES(Optimizer):
    def __init__(self, test_function: TestFunction,
                 population_size: int = 50,
                 max_iter: int = 1000,
                 parameters: dict = {'sigma': 0.3, 'mu_factor': 0.5},
                 print_interval: int = 10,
                 device: str = 'cpu'):
        """
        CMA-ES optimizer implementation
        
        Args:
            test_function: The function to be optimized
            population_size: Population size (lambda)
            max_iter: Maximum number of iterations
            parameters: Dictionary with parameters:
                - sigma: Initial step size
                - mu_factor: Proportion of population to use as parents (0 to 1)
            print_interval: Interval for printing progress
            device: Device to run computations on
        """
        super().__init__(test_function, population_size, max_iter, parameters, print_interval, device)
        
    def initialize(self):
        """Initialize CMA-ES parameters and population"""
        # Problem dimension
        self.n = self.test_function.dim
        
        # Strategy parameter setting
        self.sigma = self.parameters.get('sigma', 0.3)  # Step size
        
        # Number of parents/points for recombination
        self.mu = int(self.population_size * self.parameters.get('mu_factor', 0.5))
        
        # Calculate weights
        self.weights = torch.log(torch.tensor(self.mu + 0.5, device=self.device, dtype=torch.float64)) - torch.log(torch.arange(1, self.mu + 1, device=self.device, dtype=torch.float64))
        self.weights = self.weights / self.weights.sum()
        self.mueff = (self.weights.sum() ** 2) / (self.weights ** 2).sum()
        
        # Adaptation parameters
        self.cc = 4.0 / (self.n + 4.0)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2.0 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1.0 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        
        # Initialize dynamic state variables
        min_bounds, max_bounds = self.test_function.bounds
        mean_value = (min_bounds + max_bounds) / 2
        self.mean = mean_value.to(self.device, dtype=torch.float64)  # Initial mean
        self.pc = torch.zeros(self.n, device=self.device, dtype=torch.float64)  # Evolution path for C
        self.ps = torch.zeros(self.n, device=self.device, dtype=torch.float64)  # Evolution path for sigma
        self.C = torch.eye(self.n, device=self.device, dtype=torch.float64)  # Covariance matrix
        self.B = torch.eye(self.n, device=self.device, dtype=torch.float64)  # B defines coordinate system
        self.D = torch.ones(self.n, device=self.device, dtype=torch.float64)  # D contains scaling factors
        
        # Generate initial population
        self.solutions = None
        self.fitness = None
        
        # Best solution tracking
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def update(self):
        """Update one generation of CMA-ES"""
        # 1. Generate and evaluate population
        # Sample from multivariate normal distribution
        z = torch.randn(self.population_size, self.n, device=self.device, dtype=torch.float64)
        
        # Apply transformation: y = B * D * z
        y = z @ (self.B * self.D).T
        
        # Apply scaling and shift: x = mean + sigma * y
        self.solutions = self.mean + self.sigma * y
        
        # Enforce bounds
        min_bounds, max_bounds = self.test_function.bounds
        self.solutions = torch.max(torch.min(self.solutions, max_bounds), min_bounds)
        
        # Evaluate fitness
        self.fitness = self._evaluate(self.solutions)
        
        # 2. Sort by fitness and compute weighted mean
        sorted_indices = torch.argsort(self.fitness)
        best_solutions = self.solutions[sorted_indices[:self.mu]]
        
        # 3. Update mean
        old_mean = self.mean.clone()
        self.mean = torch.matmul(self.weights, best_solutions)
        
        # 4. Update evolution paths
        y_w = (self.mean - old_mean) / self.sigma
        
        # Update ps (path for sigma)
        cs_term = (self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + torch.sqrt(cs_term.clone().detach()) * (self.B @ (torch.inverse(torch.diag(self.D)) @ (self.B.T @ y_w)))
        
        # Update pc (path for C)
        hsig = torch.norm(self.ps) / torch.sqrt(1 - (1 - self.cs) ** (2 * (self.max_iter + 1))) < 1.4 + 2 / (self.n + 1)
        cc_term = (self.cc * (2 - self.cc) * self.mueff)
        self.pc = (1 - self.cc) * self.pc + hsig * torch.sqrt(cc_term.clone().detach()) * y_w
        
        # 5. Update covariance matrix
        # Rank-one update
        rank_one = self.pc.unsqueeze(1) @ self.pc.unsqueeze(0)
        
        # Rank-mu update
        weights_io = torch.cat([self.weights, torch.zeros(self.population_size - self.mu, device=self.device, dtype=torch.float64)])
        y_k = (self.solutions - old_mean) / self.sigma
        rank_mu = torch.zeros_like(self.C)
        for k in range(self.population_size):
            w_k = weights_io[k]
            if w_k > 0:
                rank_mu += w_k * (y_k[sorted_indices[k]].unsqueeze(1) @ y_k[sorted_indices[k]].unsqueeze(0))
        
        # Combine updates
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu
        
        # 6. Update step size sigma
        norm_ps = torch.norm(self.ps)
        sqrt_n = torch.sqrt(torch.tensor(self.n, device=self.device, dtype=torch.float64).clone().detach())
        self.sigma *= torch.exp((self.cs / self.damps) * (norm_ps / sqrt_n - 1))
        
        # 7. Compute eigenvalues and eigenvectors of C
        # We use torch.linalg.eigh as it's more suitable for symmetric matrices
        self.D, self.B = torch.linalg.eigh(self.C)
        self.D = torch.sqrt(torch.clamp(self.D, min=1e-8))
        
        # 8. Update best solution if necessary
        current_best_idx = sorted_indices[0]
        if self.fitness[current_best_idx] < self.best_fitness:
            self.best_solution = self.solutions[current_best_idx].clone()
            self.best_fitness = self.fitness[current_best_idx].item()
    
    def _get_best(self):
        """Return best solution and fitness from current population"""
        if self.solutions is None or self.fitness is None:
            min_bounds, max_bounds = self.test_function.bounds
            return (min_bounds + max_bounds) / 2, float('inf')
        
        best_idx = torch.argmin(self.fitness)
        return self.solutions[best_idx], self.fitness[best_idx]
    
    def _evaluate(self, positions):
        """Evaluate population solutions"""
        return self.test_function(positions)
