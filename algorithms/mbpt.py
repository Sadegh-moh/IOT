"""
MBPT (Maximum Benefit Per Unit Time) Algorithm Implementation
Optimizes task scheduling based on economic value and execution time
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from models.task import Task, TaskStatus
from models.edge_server import EdgeServer
import config


@dataclass
class MBPTResult:
    """Result of MBPT algorithm execution"""
    task_assignments: Dict[str, List[Task]]
    total_profit: float
    total_penalty: float
    net_profit: float
    makespan: float
    energy_consumption: float
    deadline_miss_ratio: float
    reliability: float
    execution_time: float


class MBPTAlgorithm:
    """
    Maximum Benefit Per Unit Time Algorithm
    
    This algorithm prioritizes tasks based on the ratio of profit to execution time,
    ensuring that high-value tasks are executed earlier to maximize system profit.
    """
    
    def __init__(self, config_params: dict = None):
        """
        Initialize MBPT algorithm
        
        Args:
            config_params: Configuration parameters for the algorithm
        """
        self.config = config_params or config.MBPT_CONFIG
        self.profit_weight = self.config.get('profit_weight', 0.6)
        self.deadline_weight = self.config.get('deadline_weight', 0.3)
        self.utilization_weight = self.config.get('utilization_weight', 0.1)
        
        # Performance tracking
        self.execution_history = []
        self.performance_metrics = {}
    
    def calculate_task_priority(self, task: Task, current_time: float) -> float:
        """
        Calculate task priority using MBPT formula
        
        Args:
            task: Task to calculate priority for
            current_time: Current simulation time
            
        Returns:
            Calculated priority value
        """
        # Base MBPT priority: profit per unit time
        if task.execution_time > 0:
            mbpt_priority = task.profit / task.execution_time
        else:
            mbpt_priority = 0.0
        
        # Deadline urgency factor
        remaining_time = task.deadline - current_time
        if remaining_time > 0:
            deadline_factor = 1.0 / (remaining_time + 1)  # +1 to avoid division by zero
        else:
            deadline_factor = 1.0  # High priority for overdue tasks
        
        # Utilization efficiency factor
        utilization_factor = task.utilization
        
        # Combined priority with weights
        priority = (
            self.profit_weight * mbpt_priority +
            self.deadline_weight * deadline_factor +
            self.utilization_weight * utilization_factor
        )
        
        return priority
    
    def optimize_schedule(self, 
                         tasks: List[Task], 
                         edge_servers: List[EdgeServer],
                         current_time: float) -> MBPTResult:
        """
        Optimize task scheduling using MBPT algorithm
        
        Args:
            tasks: List of tasks to schedule
            edge_servers: List of available edge servers
            current_time: Current simulation time
            
        Returns:
            MBPTResult containing optimized schedule and metrics
        """
        import time
        start_time = time.time()
        
        # Calculate priorities for all tasks
        for task in tasks:
            task.priority = self.calculate_task_priority(task, current_time)
        
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Initialize assignments
        task_assignments = {server.server_id: [] for server in edge_servers}
        
        # Sort servers by performance score
        sorted_servers = sorted(edge_servers, 
                              key=lambda s: s.get_performance_score(), 
                              reverse=True)
        
        # Assign tasks to servers
        unassigned_tasks = []
        
        for task in sorted_tasks:
            assigned = False
            
            # Try to assign to best available server
            for server in sorted_servers:
                if server.can_accept_task(task.utilization):
                    task_assignments[server.server_id].append(task)
                    server.assign_task(task.task_id, task.utilization)
                    assigned = True
                    break
            
            if not assigned:
                unassigned_tasks.append(task)
        
        # Calculate metrics
        total_profit, total_penalty = self._calculate_profit_metrics(tasks, current_time)
        makespan = self._calculate_makespan(task_assignments, edge_servers)
        energy_consumption = self._calculate_energy_consumption(edge_servers)
        deadline_miss_ratio = self._calculate_deadline_miss_ratio(tasks, current_time)
        reliability = self._calculate_reliability(edge_servers)
        
        execution_time = time.time() - start_time
        
        # Create result
        result = MBPTResult(
            task_assignments=task_assignments,
            total_profit=total_profit,
            total_penalty=total_penalty,
            net_profit=total_profit - total_penalty,
            makespan=makespan,
            energy_consumption=energy_consumption,
            deadline_miss_ratio=deadline_miss_ratio,
            reliability=reliability,
            execution_time=execution_time
        )
        
        # Update performance tracking
        self._update_performance_tracking(result)
        
        return result
    
    def _calculate_profit_metrics(self, tasks: List[Task], current_time: float) -> Tuple[float, float]:
        """
        Calculate total profit and penalty for all tasks
        
        Args:
            tasks: List of tasks
            current_time: Current simulation time
            
        Returns:
            Tuple of (total_profit, total_penalty)
        """
        total_profit = 0.0
        total_penalty = 0.0
        
        for task in tasks:
            if task.status == TaskStatus.COMPLETED:
                if task.completion_time and task.completion_time <= task.deadline:
                    total_profit += task.profit
                else:
                    # Penalty for missing deadline
                    delay = task.completion_time - task.deadline
                    penalty = task.penalty * (delay / task.deadline) if task.deadline > 0 else task.penalty
                    total_penalty += penalty
            elif task.status == TaskStatus.MISSED_DEADLINE:
                total_penalty += task.penalty
            elif task.status == TaskStatus.FAILED:
                total_penalty += task.penalty
        
        return total_profit, total_penalty
    
    def _calculate_makespan(self, 
                           task_assignments: Dict[str, List[Task]], 
                           edge_servers: List[EdgeServer]) -> float:
        """
        Calculate makespan (total execution time)
        
        Args:
            task_assignments: Task assignments by server
            edge_servers: List of edge servers
            
        Returns:
            Makespan value
        """
        server_makespans = []
        
        for server in edge_servers:
            if server.server_id in task_assignments:
                tasks = task_assignments[server.server_id]
                if tasks:
                    # Calculate server makespan based on task execution times and server capacity
                    total_execution_time = sum(task.execution_time for task in tasks)
                    server_makespan = total_execution_time / max(server.cpu_cores, 1)
                    server_makespans.append(server_makespan)
        
        return max(server_makespans) if server_makespans else 0.0
    
    def _calculate_energy_consumption(self, edge_servers: List[EdgeServer]) -> float:
        """
        Calculate total energy consumption
        
        Args:
            edge_servers: List of edge servers
            
        Returns:
            Total energy consumption
        """
        return sum(server.total_energy_consumed for server in edge_servers)
    
    def _calculate_deadline_miss_ratio(self, tasks: List[Task], current_time: float) -> float:
        """
        Calculate deadline miss ratio
        
        Args:
            tasks: List of tasks
            current_time: Current simulation time
            
        Returns:
            Deadline miss ratio (0.0-1.0)
        """
        if not tasks:
            return 0.0
        
        missed_deadlines = 0
        for task in tasks:
            if task.is_deadline_missed(current_time):
                missed_deadlines += 1
        
        return missed_deadlines / len(tasks)
    
    def _calculate_reliability(self, edge_servers: List[EdgeServer]) -> float:
        """
        Calculate system reliability based on server availability
        
        Args:
            edge_servers: List of edge servers
            
        Returns:
            System reliability (0.0-1.0)
        """
        if not edge_servers:
            return 0.0
        
        # Calculate average server availability
        total_availability = sum(server.get_availability() for server in edge_servers)
        avg_availability = total_availability / len(edge_servers)
        
        # Apply fault tolerance factor
        fault_tolerance_factor = 0.999  # Target reliability from config
        reliability = avg_availability * fault_tolerance_factor
        
        return min(1.0, reliability)
    
    def _update_performance_tracking(self, result: MBPTResult):
        """
        Update performance tracking metrics
        
        Args:
            result: MBPT algorithm result
        """
        self.execution_history.append({
            'timestamp': len(self.execution_history),
            'net_profit': result.net_profit,
            'makespan': result.makespan,
            'energy_consumption': result.energy_consumption,
            'deadline_miss_ratio': result.deadline_miss_ratio,
            'reliability': result.reliability,
            'execution_time': result.execution_time
        })
        
        # Update performance metrics
        if self.execution_history:
            recent_results = self.execution_history[-10:]  # Last 10 executions
            
            self.performance_metrics = {
                'avg_net_profit': np.mean([r['net_profit'] for r in recent_results]),
                'avg_makespan': np.mean([r['makespan'] for r in recent_results]),
                'avg_energy_consumption': np.mean([r['energy_consumption'] for r in recent_results]),
                'avg_deadline_miss_ratio': np.mean([r['deadline_miss_ratio'] for r in recent_results]),
                'avg_reliability': np.mean([r['reliability'] for r in recent_results]),
                'avg_execution_time': np.mean([r['execution_time'] for r in recent_results]),
                'total_executions': len(self.execution_history)
            }
    
    def get_performance_summary(self) -> dict:
        """
        Get performance summary of the algorithm
        
        Returns:
            Dictionary with performance metrics
        """
        return self.performance_metrics.copy()
    
    def __str__(self) -> str:
        """String representation of the MBPT algorithm"""
        return (f"MBPTAlgorithm(profit_w={self.profit_weight:.2f}, "
                f"deadline_w={self.deadline_weight:.2f}, "
                f"utilization_w={self.utilization_weight:.2f})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the MBPT algorithm"""
        return self.__str__()
