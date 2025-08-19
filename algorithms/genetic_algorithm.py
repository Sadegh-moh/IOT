"""
Genetic Algorithm Implementation for Task Scheduling
Baseline algorithm for comparison with MBPT
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from models.task import Task, TaskStatus
from models.edge_server import EdgeServer
import config


@dataclass
class GAResult:
    """Result of Genetic Algorithm execution"""
    task_assignments: Dict[str, List[Task]]
    total_profit: float
    total_penalty: float
    net_profit: float
    makespan: float
    energy_consumption: float
    deadline_miss_ratio: float
    reliability: float
    execution_time: float
    evolution_history: List[float]


class GeneticAlgorithm:
    """
    Genetic Algorithm for Task Scheduling
    
    This algorithm uses evolutionary principles to find optimal task assignments
    through selection, crossover, and mutation operations.
    """
    
    def __init__(self, config_params: dict = None):
        """
        Initialize Genetic Algorithm
        
        Args:
            config_params: Configuration parameters for the algorithm
        """
        self.config = config_params or config.GA_CONFIG
        self.population_size = self.config.get('population_size', 100)
        self.generations = self.config.get('generations', 200)
        self.crossover_rate = self.config.get('crossover_rate', 0.8)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.elite_size = self.config.get('elite_size', 10)
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
    
    def optimize_schedule(self, 
                         tasks: List[Task], 
                         edge_servers: List[EdgeServer],
                         current_time: float) -> GAResult:
        """
        Optimize task scheduling using Genetic Algorithm
        
        Args:
            tasks: List of tasks to schedule
            edge_servers: List of available edge servers
            current_time: Current simulation time
            
        Returns:
            GAResult containing optimized schedule and metrics
        """
        import time
        start_time = time.time()
        
        if not tasks or not edge_servers:
            return self._create_empty_result()
        
        # Initialize population
        population = self._initialize_population(tasks, edge_servers)
        
        # Evolution history
        evolution_history = []
        best_fitness = 0.0
        
        # Main evolution loop
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitness_scores = [self._calculate_fitness(individual, tasks, edge_servers, current_time) 
                            for individual in population]
            
            # Track best fitness
            generation_best = max(fitness_scores)
            evolution_history.append(generation_best)
            
            if generation_best > best_fitness:
                best_fitness = generation_best
            
            # Selection
            selected = self._selection(population, fitness_scores)
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate new individuals through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutation(child, len(edge_servers))
                
                new_population.append(child)
            
            population = new_population
        
        # Find best solution
        final_fitness = [self._calculate_fitness(individual, tasks, edge_servers, current_time) 
                        for individual in population]
        best_individual_idx = np.argmax(final_fitness)
        best_solution = population[best_individual_idx]
        
        # Convert solution to task assignments
        task_assignments = self._convert_solution_to_assignments(
            best_solution, tasks, edge_servers
        )
        
        # Calculate metrics
        total_profit, total_penalty = self._calculate_profit_metrics(tasks, current_time)
        makespan = self._calculate_makespan(task_assignments, edge_servers)
        energy_consumption = self._calculate_energy_consumption(edge_servers)
        deadline_miss_ratio = self._calculate_deadline_miss_ratio(tasks, current_time)
        reliability = self._calculate_reliability(edge_servers)
        
        execution_time = time.time() - start_time
        
        # Create result
        result = GAResult(
            task_assignments=task_assignments,
            total_profit=total_profit,
            total_penalty=total_penalty,
            net_profit=total_profit - total_penalty,
            makespan=makespan,
            energy_consumption=energy_consumption,
            deadline_miss_ratio=deadline_miss_ratio,
            reliability=reliability,
            execution_time=execution_time,
            evolution_history=evolution_history
        )
        
        # Update performance tracking
        self._update_performance_tracking(result)
        
        return result
    
    def _initialize_population(self, tasks: List[Task], edge_servers: List[EdgeServer]) -> List[np.ndarray]:
        """
        Initialize population with random individuals
        
        Args:
            tasks: List of tasks
            edge_servers: List of edge servers
            
        Returns:
            List of individuals (each is an array of server assignments)
        """
        population = []
        
        for _ in range(self.population_size):
            # Random assignment of tasks to servers
            individual = np.random.randint(0, len(edge_servers), size=len(tasks))
            population.append(individual)
        
        return population
    
    def _selection(self, population: List[np.ndarray], fitness_scores: List[float]) -> List[np.ndarray]:
        """
        Select individuals for reproduction using tournament selection
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for each individual
            
        Returns:
            List of selected individuals
        """
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Tournament selection
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Child individual
        """
        # Single-point crossover
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child = np.concatenate([
            parent1[:crossover_point],
            parent2[crossover_point:]
        ])
        
        return child
    
    def _mutation(self, individual: np.ndarray, num_servers: int) -> np.ndarray:
        """
        Perform mutation on an individual
        
        Args:
            individual: Individual to mutate
            num_servers: Number of available servers
            
        Returns:
            Mutated individual
        """
        mutated = individual.copy()
        
        # Random mutation of some genes
        for i in range(len(mutated)):
            if random.random() < 0.1:  # 10% mutation probability per gene
                mutated[i] = random.randint(0, num_servers - 1)
        
        return mutated
    
    def _calculate_fitness(self, 
                          individual: np.ndarray, 
                          tasks: List[Task], 
                          edge_servers: List[EdgeServer],
                          current_time: float) -> float:
        """
        Calculate fitness of an individual
        
        Args:
            individual: Individual representing task assignments
            tasks: List of tasks
            edge_servers: List of edge servers
            current_time: Current simulation time
            
        Returns:
            Fitness value (higher is better)
        """
        if len(individual) != len(tasks):
            return 0.0
        
        # Convert individual to task assignments
        task_assignments = self._convert_solution_to_assignments(
            individual, tasks, edge_servers
        )
        
        # Calculate objective function components
        profit_score = self._calculate_profit_score(tasks, current_time)
        deadline_score = self._calculate_deadline_score(tasks, current_time)
        energy_score = self._calculate_energy_score(edge_servers)
        load_balance_score = self._calculate_load_balance_score(task_assignments, edge_servers)
        
        # Combined fitness (weighted sum)
        fitness = (0.4 * profit_score + 
                  0.3 * deadline_score + 
                  0.2 * energy_score + 
                  0.1 * load_balance_score)
        
        return fitness
    
    def _calculate_profit_score(self, tasks: List[Task], current_time: float) -> float:
        """Calculate profit-based score"""
        if not tasks:
            return 0.0
        
        total_profit = sum(task.calculate_profit_loss(current_time) for task in tasks)
        max_profit = sum(task.profit for task in tasks)
        
        if max_profit > 0:
            return total_profit / max_profit
        return 0.0
    
    def _calculate_deadline_score(self, tasks: List[Task], current_time: float) -> float:
        """Calculate deadline-based score"""
        if not tasks:
            return 0.0
        
        missed_deadlines = sum(1 for task in tasks if task.is_deadline_missed(current_time))
        return 1.0 - (missed_deadlines / len(tasks))
    
    def _calculate_energy_score(self, edge_servers: List[EdgeServer]) -> float:
        """Calculate energy efficiency score"""
        if not edge_servers:
            return 0.0
        
        avg_efficiency = np.mean([server.energy_efficiency for server in edge_servers])
        return avg_efficiency
    
    def _calculate_load_balance_score(self, 
                                    task_assignments: Dict[str, List[Task]], 
                                    edge_servers: List[EdgeServer]) -> float:
        """Calculate load balancing score"""
        if not edge_servers:
            return 0.0
        
        server_loads = []
        for server in edge_servers:
            if server.server_id in task_assignments:
                load = len(task_assignments[server.server_id]) / server.cpu_cores
                server_loads.append(load)
            else:
                server_loads.append(0.0)
        
        if not server_loads:
            return 0.0
        
        # Calculate load balance (lower variance is better)
        variance = np.var(server_loads)
        return 1.0 / (1.0 + variance)
    
    def _convert_solution_to_assignments(self, 
                                       individual: np.ndarray, 
                                       tasks: List[Task], 
                                       edge_servers: List[EdgeServer]) -> Dict[str, List[Task]]:
        """
        Convert individual solution to task assignments
        
        Args:
            individual: Individual representing task assignments
            tasks: List of tasks
            edge_servers: List of edge servers
            
        Returns:
            Dictionary mapping server IDs to assigned tasks
        """
        task_assignments = {server.server_id: [] for server in edge_servers}
        
        for i, task in enumerate(tasks):
            if i < len(individual):
                server_idx = individual[i]
                if 0 <= server_idx < len(edge_servers):
                    server = edge_servers[server_idx]
                    task_assignments[server.server_id].append(task)
        
        return task_assignments
    
    def _calculate_profit_metrics(self, tasks: List[Task], current_time: float) -> Tuple[float, float]:
        """Calculate total profit and penalty"""
        total_profit = 0.0
        total_penalty = 0.0
        
        for task in tasks:
            profit_loss = task.calculate_profit_loss(current_time)
            if profit_loss > 0:
                total_profit += profit_loss
            else:
                total_penalty += abs(profit_loss)
        
        return total_profit, total_penalty
    
    def _calculate_makespan(self, 
                           task_assignments: Dict[str, List[Task]], 
                           edge_servers: List[EdgeServer]) -> float:
        """Calculate makespan"""
        server_makespans = []
        
        for server in edge_servers:
            if server.server_id in task_assignments:
                tasks = task_assignments[server.server_id]
                if tasks:
                    total_execution_time = sum(task.execution_time for task in tasks)
                    server_makespan = total_execution_time / max(server.cpu_cores, 1)
                    server_makespans.append(server_makespan)
        
        return max(server_makespans) if server_makespans else 0.0
    
    def _calculate_energy_consumption(self, edge_servers: List[EdgeServer]) -> float:
        """Calculate total energy consumption"""
        return sum(server.total_energy_consumed for server in edge_servers)
    
    def _calculate_deadline_miss_ratio(self, tasks: List[Task], current_time: float) -> float:
        """Calculate deadline miss ratio"""
        if not tasks:
            return 0.0
        
        missed_deadlines = sum(1 for task in tasks if task.is_deadline_missed(current_time))
        return missed_deadlines / len(tasks)
    
    def _calculate_reliability(self, edge_servers: List[EdgeServer]) -> float:
        """Calculate system reliability"""
        if not edge_servers:
            return 0.0
        
        total_availability = sum(server.get_availability() for server in edge_servers)
        avg_availability = total_availability / len(edge_servers)
        
        fault_tolerance_factor = 0.999
        reliability = avg_availability * fault_tolerance_factor
        
        return min(1.0, reliability)
    
    def _create_empty_result(self) -> GAResult:
        """Create empty result when no tasks or servers"""
        return GAResult(
            task_assignments={},
            total_profit=0.0,
            total_penalty=0.0,
            net_profit=0.0,
            makespan=0.0,
            energy_consumption=0.0,
            deadline_miss_ratio=0.0,
            reliability=0.0,
            execution_time=0.0,
            evolution_history=[]
        )
    
    def _update_performance_tracking(self, result: GAResult):
        """Update performance tracking metrics"""
        self.execution_history.append({
            'timestamp': len(self.execution_history),
            'net_profit': result.net_profit,
            'makespan': result.makespan,
            'energy_consumption': result.energy_consumption,
            'deadline_miss_ratio': result.deadline_miss_ratio,
            'reliability': result.reliability,
            'execution_time': result.execution_time
        })
    
    def get_performance_summary(self) -> dict:
        """Get performance summary of the algorithm"""
        return self.performance_metrics.copy()
    
    def __str__(self) -> str:
        """String representation of the Genetic Algorithm"""
        return (f"GeneticAlgorithm(population={self.population_size}, "
                f"generations={self.generations}, "
                f"crossover_rate={self.crossover_rate}, "
                f"mutation_rate={self.mutation_rate})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the Genetic Algorithm"""
        return self.__str__()
