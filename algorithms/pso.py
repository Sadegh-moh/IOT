"""
PSO (Particle Swarm Optimization) Algorithm Implementation
Baseline algorithm for task scheduling comparison
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from models.task import Task, TaskStatus
from models.edge_server import EdgeServer
import config


@dataclass
class PSOResult:
    """Result of PSO algorithm execution"""
    task_assignments: Dict[str, List[Task]]
    total_profit: float
    total_penalty: float
    net_profit: float
    makespan: float
    energy_consumption: float
    deadline_miss_ratio: float
    reliability: float
    execution_time: float
    convergence_history: List[float]


class PSOAlgorithm:
    """
    Particle Swarm Optimization Algorithm for Task Scheduling
    
    This algorithm uses swarm intelligence to find optimal task assignments
    by considering multiple objectives: profit, deadline, and energy efficiency.
    """
    
    def __init__(self, config_params: dict = None):
        """
        Initialize PSO algorithm
        
        Args:
            config_params: Configuration parameters for the algorithm
        """
        self.config = config_params or config.PSO_CONFIG
        self.particles = self.config.get('particles', 50)
        self.iterations = self.config.get('iterations', 100)
        self.cognitive_weight = self.config.get('cognitive_weight', 2.0)
        self.social_weight = self.config.get('social_weight', 2.0)
        self.inertia_weight = self.config.get('inertia_weight', 0.7)
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
    
    def optimize_schedule(self, 
                         tasks: List[Task], 
                         edge_servers: List[EdgeServer],
                         current_time: float) -> PSOResult:
        """
        Optimize task scheduling using PSO algorithm
        
        Args:
            tasks: List of tasks to schedule
            edge_servers: List of available edge servers
            current_time: Current simulation time
            
        Returns:
            PSOResult containing optimized schedule and metrics
        """
        import time
        start_time = time.time()
        
        if not tasks or not edge_servers:
            return self._create_empty_result()
        
        # Initialize particles
        particles = self._initialize_particles(tasks, edge_servers)
        velocities = self._initialize_velocities(len(tasks), len(edge_servers))
        
        # Initialize best positions and fitness
        personal_best = particles.copy()
        personal_best_fitness = [self._calculate_fitness(p, tasks, edge_servers, current_time) 
                               for p in particles]
        
        global_best_idx = np.argmax(personal_best_fitness)
        global_best = particles[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        convergence_history = [global_best_fitness]
        
        # Main PSO loop
        for iteration in range(self.iterations):
            # Update particles
            for i in range(self.particles):
                # Update velocity
                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], personal_best[i], global_best
                )
                
                # Update position
                particles[i] = self._update_position(particles[i], velocities[i])
                
                # Ensure position is valid
                particles[i] = self._ensure_valid_position(particles[i], len(edge_servers))
                
                # Calculate fitness
                fitness = self._calculate_fitness(particles[i], tasks, edge_servers, current_time)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
            
            convergence_history.append(global_best_fitness)
            
            # Adaptive inertia weight
            self.inertia_weight *= 0.99
        
        # Convert best solution to task assignments
        task_assignments = self._convert_solution_to_assignments(
            global_best, tasks, edge_servers
        )
        
        # Calculate metrics
        total_profit, total_penalty = self._calculate_profit_metrics(tasks, current_time)
        makespan = self._calculate_makespan(task_assignments, edge_servers)
        energy_consumption = self._calculate_energy_consumption(edge_servers)
        deadline_miss_ratio = self._calculate_deadline_miss_ratio(tasks, current_time)
        reliability = self._calculate_reliability(edge_servers)
        
        execution_time = time.time() - start_time
        
        # Create result
        result = PSOResult(
            task_assignments=task_assignments,
            total_profit=total_profit,
            total_penalty=total_penalty,
            net_profit=total_profit - total_penalty,
            makespan=makespan,
            energy_consumption=energy_consumption,
            deadline_miss_ratio=deadline_miss_ratio,
            reliability=reliability,
            execution_time=execution_time,
            convergence_history=convergence_history
        )
        
        # Update performance tracking
        self._update_performance_tracking(result)
        
        return result
    
    def _initialize_particles(self, tasks: List[Task], edge_servers: List[EdgeServer]) -> List[np.ndarray]:
        """
        Initialize particle positions
        
        Args:
            tasks: List of tasks
            edge_servers: List of edge servers
            
        Returns:
            List of particle position arrays
        """
        particles = []
        
        for _ in range(self.particles):
            # Random assignment of tasks to servers
            particle = np.random.randint(0, len(edge_servers), size=len(tasks))
            particles.append(particle)
        
        return particles
    
    def _initialize_velocities(self, num_tasks: int, num_servers: int) -> List[np.ndarray]:
        """
        Initialize particle velocities
        
        Args:
            num_tasks: Number of tasks
            num_servers: Number of edge servers
            
        Returns:
            List of velocity arrays
        """
        velocities = []
        
        for _ in range(self.particles):
            # Random velocities in range [-num_servers/2, num_servers/2]
            velocity = np.random.uniform(-num_servers/2, num_servers/2, size=num_tasks)
            velocities.append(velocity)
        
        return velocities
    
    def _update_velocity(self, 
                        velocity: np.ndarray, 
                        position: np.ndarray, 
                        personal_best: np.ndarray, 
                        global_best: np.ndarray) -> np.ndarray:
        """
        Update particle velocity
        
        Args:
            velocity: Current velocity
            position: Current position
            personal_best: Personal best position
            global_best: Global best position
            
        Returns:
            Updated velocity
        """
        # Cognitive component
        cognitive = self.cognitive_weight * np.random.random() * (personal_best - position)
        
        # Social component
        social = self.social_weight * np.random.random() * (global_best - position)
        
        # Update velocity
        new_velocity = (self.inertia_weight * velocity + cognitive + social)
        
        # Limit velocity
        max_velocity = len(global_best) / 2
        new_velocity = np.clip(new_velocity, -max_velocity, max_velocity)
        
        return new_velocity
    
    def _update_position(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Update particle position
        
        Args:
            position: Current position
            velocity: Current velocity
            
        Returns:
            Updated position
        """
        new_position = position + velocity
        
        # Round to nearest integer
        new_position = np.round(new_position).astype(int)
        
        return new_position
    
    def _ensure_valid_position(self, position: np.ndarray, num_servers: int) -> np.ndarray:
        """
        Ensure position is within valid bounds
        
        Args:
            position: Particle position
            num_servers: Number of available servers
            
        Returns:
            Valid position
        """
        # Ensure all values are within [0, num_servers-1]
        position = np.clip(position, 0, num_servers - 1)
        
        return position
    
    def _calculate_fitness(self, 
                          particle: np.ndarray, 
                          tasks: List[Task], 
                          edge_servers: List[EdgeServer],
                          current_time: float) -> float:
        """
        Calculate fitness of a particle (solution)
        
        Args:
            particle: Particle position representing task assignments
            tasks: List of tasks
            edge_servers: List of edge servers
            current_time: Current simulation time
            
        Returns:
            Fitness value (higher is better)
        """
        if len(particle) != len(tasks):
            return 0.0
        
        # Convert particle to task assignments
        task_assignments = self._convert_solution_to_assignments(
            particle, tasks, edge_servers
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
                                       particle: np.ndarray, 
                                       tasks: List[Task], 
                                       edge_servers: List[EdgeServer]) -> Dict[str, List[Task]]:
        """
        Convert particle solution to task assignments
        
        Args:
            particle: Particle position
            tasks: List of tasks
            edge_servers: List of edge servers
            
        Returns:
            Dictionary mapping server IDs to assigned tasks
        """
        task_assignments = {server.server_id: [] for server in edge_servers}
        
        for i, task in enumerate(tasks):
            if i < len(particle):
                server_idx = particle[i]
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
    
    def _create_empty_result(self) -> PSOResult:
        """Create empty result when no tasks or servers"""
        return PSOResult(
            task_assignments={},
            total_profit=0.0,
            total_penalty=0.0,
            net_profit=0.0,
            makespan=0.0,
            energy_consumption=0.0,
            deadline_miss_ratio=0.0,
            reliability=0.0,
            execution_time=0.0,
            convergence_history=[]
        )
    
    def _update_performance_tracking(self, result: PSOResult):
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
        """String representation of the PSO algorithm"""
        return (f"PSOAlgorithm(particles={self.particles}, "
                f"iterations={self.iterations}, "
                f"cognitive_w={self.cognitive_weight}, "
                f"social_w={self.social_weight})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the PSO algorithm"""
        return self.__str__()
