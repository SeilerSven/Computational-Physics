import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable
import random

class GeneticAlgorithm:
   # Genetic algorithm for maximization
    
    def __init__(self, population_size: int = 100, generations: int = 200, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elite_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
    def create_individual(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        # Create individual inbetween bounds
        return np.array([random.uniform(low, high) for low, high in bounds])
    
    def create_population(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        # Create random population
        return np.array([self.create_individual(bounds) for _ in range(self.population_size)])
    
    def tournament_selection(self, population: np.ndarray, fitness: np.ndarray, 
                           tournament_size: int = 3) -> np.ndarray:
        # Tournament selection
        selected_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[selected_indices]
        winner_idx = selected_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Crossover
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def mutate(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        # Gaussian mutation
        mutated = individual.copy()
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation with 10% der bandwidth as standard deviation
                sigma = (bounds[i][1] - bounds[i][0]) * 0.1
                mutated[i] += np.random.normal(0, sigma)
                # Boundaries
                mutated[i] = np.clip(mutated[i], bounds[i][0], bounds[i][1])
        return mutated
    
    def optimize(self, fitness_function: Callable, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float, List[float]]:
        # Optimization loop
        population = self.create_population(bounds)
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Fitness calculation
            fitness = np.array([fitness_function(individual) for individual in population])
            
            # Save best solution
            best_idx = np.argmax(fitness)
            best_individual = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            best_fitness_history.append(best_fitness)
            
            # Elitism
            # Save best individuals
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            elite = population[elite_indices].copy()
            
            # Create new population
            new_population = []
            
            # Append elites
            for individual in elite:
                new_population.append(individual)
            
            # Fill population by selection, crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # Set population size 
            population = np.array(new_population[:self.population_size])
            
            # Show progress
            if generation % 50 == 0:
                print(f"Generation {generation}: Best Fitness = {best_fitness:.6f}")
        
        # Final evaluation
        final_fitness = np.array([fitness_function(individual) for individual in population])
        best_idx = np.argmax(final_fitness)
        
        return population[best_idx], final_fitness[best_idx], best_fitness_history

# Define problems

def problem_p1(x: np.ndarray) -> float:
    # Problem 1: Concentric waves
    # f(x,y) = cos²(n*π*r) * exp(-r²/σ²)
    # with r² = (x-0.5)² + (y-0.5)²
    # Parameters: n=9, σ²=0.15
    n = 9
    sigma_squared = 0.15
    
    x_coord, y_coord = x[0], x[1]
    r_squared = (x_coord - 0.5)**2 + (y_coord - 0.5)**2
    r = np.sqrt(r_squared)
    
    cos_term = np.cos(n * np.pi * r)**2
    exp_term = np.exp(-r_squared / sigma_squared)
    
    return cos_term * exp_term

def problem_p2(x: np.ndarray) -> float:
    # Problem 2: Zwei 2D-Gauß distribution
    # f(x,y) = 0.8 * exp(-r1²/(0.3)²) + 0.879008 * exp(-r2²/(0.03)²)
    # with r1² = (x-0.5)² + (y-0.5)²
    # with r2² = (x-0.6)² + (y-0.1)²
    x_coord, y_coord = x[0], x[1]
    
    # First gaussian distribution
    r1_squared = (x_coord - 0.5)**2 + (y_coord - 0.5)**2
    gauss1 = 0.8 * np.exp(-r1_squared / (0.3**2))
    
    # second gaussian distribution
    r2_squared = (x_coord - 0.6)**2 + (y_coord - 0.1)**2
    gauss2 = 0.879008 * np.exp(-r2_squared / (0.03**2))
    
    return gauss1 + gauss2

def visualize_function(func: Callable, title: str, bounds: List[Tuple[float, float]]):
    # Visualize function as contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    return X, Y, Z

def main():
    # Main function
    
    # Define search area
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    # Adjust genetic algorithm
    ga = GeneticAlgorithm(
        population_size=150,
        generations=300,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=10
    )
    
    print("=" * 60)
    print("GENNETIC ALGORITHM - Optimization of P1 and P2")
    print("=" * 60)
    
    # P1
    print("\n Problem P1: Concentric waves")
    print("-" * 40)
    
    best_p1, fitness_p1, history_p1 = ga.optimize(problem_p1, bounds)
    
    print(f"\n Solution P1:")
    print(f"   Best parameter: x = {best_p1[0]:.6f}, y = {best_p1[1]:.6f}")
    print(f"   Maximum function value: {fitness_p1:.6f}")
    
    # P2
    print("\n Problem P2: 2D-gaussian distributions")
    print("-" * 40)
    
    best_p2, fitness_p2, history_p2 = ga.optimize(problem_p2, bounds)
    
    print(f"\n Solution P2:")
    print(f"   Optimal parameter: x = {best_p2[0]:.6f}, y = {best_p2[1]:.6f}")
    print(f"   Maximum function value: {fitness_p2:.6f}")
    
    # Show convergence
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history_p1, 'b-', linewidth=2)
    plt.title('Convergence P1 (Concentric waves)')
    plt.xlabel('Generation')
    plt.ylabel('Best fitness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history_p2, 'r-', linewidth=2)
    plt.title('Convergence P2 (Gaussian distributions)')
    plt.xlabel('Generation')
    plt.ylabel('Best fitness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(history_p1, 'b-', label='P1', linewidth=2)
    plt.plot(history_p2, 'r-', label='P2', linewidth=2)
    plt.title('Comparison of convergence')
    plt.xlabel('Generation')
    plt.ylabel('Best fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualization
    print("\n Visualization...")
    
    # Visualize P1
    X1, Y1, Z1 = visualize_function(problem_p1, 'Problem P1: Concentric waves', bounds)
    plt.plot(best_p1[0], best_p1[1], 'r*', markersize=15, label=f'Maximum: ({best_p1[0]:.3f}, {best_p1[1]:.3f})')
    plt.legend()
    plt.show()
    
    # Visualize P2
    X2, Y2, Z2 = visualize_function(problem_p2, 'Problem P2: 2D-gaussian distributions', bounds)
    plt.plot(best_p2[0], best_p2[1], 'r*', markersize=15, label=f'Maximum: ({best_p2[0]:.3f}, {best_p2[1]:.3f})')
    plt.legend()
    plt.show()
    
    # Results
    print("\n Results:")
    print("-" * 40)
    print("Problem P1 (Concentric waves):")
    print(f"  - The maximum is at x={best_p1[0]:.6f}, y={best_p1[1]:.6f}")
    print(f"  - Function value: {fitness_p1:.6f}")
    
    print("\nProblem P2 (Gaussian distributions):")
    print(f"  - The maximum is at x={best_p2[0]:.6f}, y={best_p2[1]:.6f}")
    print(f"  - Function value: {fitness_p2:.6f}")
    
    # Verify with grid search
    print("\n Verify with grid search...")
    
    def grid_search(func, bounds, resolution=1000):
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        max_val = -np.inf
        max_pos = None
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                val = func(np.array([xi, yj]))
                if val > max_val:
                    max_val = val
                    max_pos = (xi, yj)
        
        return max_pos, max_val
    
    # Verification for P1
    grid_pos_p1, grid_val_p1 = grid_search(problem_p1, bounds, 200)
    print(f"\nGrid search P1: x={grid_pos_p1[0]:.6f}, y={grid_pos_p1[1]:.6f}, f={grid_val_p1:.6f}")
    print(f"GA-solution P1:   x={best_p1[0]:.6f}, y={best_p1[1]:.6f}, f={fitness_p1:.6f}")
    print(f"Delta P1:  Δx={abs(grid_pos_p1[0]-best_p1[0]):.6f}, Δy={abs(grid_pos_p1[1]-best_p1[1]):.6f}")
    
    # Verification for P2
    grid_pos_p2, grid_val_p2 = grid_search(problem_p2, bounds, 200)
    print(f"\nGrid search P2: x={grid_pos_p2[0]:.6f}, y={grid_pos_p2[1]:.6f}, f={grid_val_p2:.6f}")
    print(f"GA-solution P2:   x={best_p2[0]:.6f}, y={best_p2[1]:.6f}, f={fitness_p2:.6f}")
    print(f"Delta P2:  Δx={abs(grid_pos_p2[0]-best_p2[0]):.6f}, Δy={abs(grid_pos_p2[1]-best_p2[1]):.6f}")

if __name__ == "__main__":
    main()