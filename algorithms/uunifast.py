"""
UUNIFAST Algorithm Implementation
Generates realistic task sets for real-time scheduling simulation
"""

import random
import numpy as np
from typing import List, Dict, Any
from models.task import TaskType
import config


class UUNIFASTGenerator:
    """
    UUNIFAST (UniFast) algorithm for generating realistic task sets
    
    This algorithm generates task sets with specified total utilization
    and realistic distribution of individual task utilizations.
    """
    
    def __init__(self, config_params: dict = None):
        """
        Initialize UUNIFAST generator
        
        Args:
            config_params: Configuration parameters
        """
        self.config = config_params or config.UUNIFAST_CONFIG
    
    def generate_tasks(self, num_tasks: int, config_params: dict = None) -> List[Dict[str, Any]]:
        """
        Generate a set of tasks using UUNIFAST algorithm
        
        Args:
            num_tasks: Number of tasks to generate
            config_params: Configuration parameters
            
        Returns:
            List of task parameter dictionaries
        """
        config_params = config_params or self.config
        
        # Get configuration
        total_utilization = config_params.get('total_utilization', 0.8)
        hard_tasks_ratio = config_params.get('hard_tasks_ratio', 0.3)
        soft_tasks_ratio = config_params.get('soft_tasks_ratio', 0.7)
        
        # Calculate number of each task type
        num_hard_tasks = int(num_tasks * hard_tasks_ratio)
        num_soft_tasks = num_tasks - num_hard_tasks
        
        # Generate utilizations using UUNIFAST
        utilizations = self._generate_utilizations(num_tasks, total_utilization)
        
        # Generate task parameters
        tasks = []
        
        for i in range(num_tasks):
            # Determine task type
            if i < num_hard_tasks:
                task_type = TaskType.HARD_REAL_TIME
            else:
                task_type = TaskType.SOFT_REAL_TIME
            
            # Generate task parameters
            task_params = self._generate_task_parameters(
                i, utilizations[i], task_type, config_params
            )
            
            tasks.append(task_params)
        
        return tasks
    
    def _generate_utilizations(self, num_tasks: int, total_utilization: float) -> List[float]:
        """
        Generate individual task utilizations using UUNIFAST
        
        Args:
            num_tasks: Number of tasks
            total_utilization: Total system utilization
            
        Returns:
            List of individual task utilizations
        """
        utilizations = []
        remaining_utilization = total_utilization
        
        for i in range(num_tasks - 1):
            # Generate utilization for current task
            # UUNIFAST formula: u_i = U * (1 - r^(1/(n-i)))^(1/(n-i))
            r = random.random()
            n_minus_i = num_tasks - i
            
            if n_minus_i > 0:
                u_i = remaining_utilization * (1 - r**(1/n_minus_i))
                utilizations.append(u_i)
                remaining_utilization -= u_i
            else:
                utilizations.append(0.0)
        
        # Last task gets remaining utilization
        utilizations.append(max(0.0, remaining_utilization))
        
        # Ensure all utilizations are within valid range
        utilizations = [max(0.01, min(0.9, u)) for u in utilizations]
        
        return utilizations
    
    def _generate_task_parameters(self, 
                                 task_id: int, 
                                 utilization: float, 
                                 task_type: TaskType,
                                 config_params: dict) -> Dict[str, Any]:
        """
        Generate parameters for a single task
        
        Args:
            task_id: Task identifier
            utilization: Task utilization
            task_type: Type of task
            config_params: Configuration parameters
            
        Returns:
            Dictionary with task parameters
        """
        # Get configuration ranges
        exec_range = config.TASK_CONFIG['execution_time_range']
        deadline_range = config.TASK_CONFIG['deadline_range']
        period_range = config.TASK_CONFIG['period_range']
        profit_range = config.TASK_CONFIG['profit_range']
        penalty_range = config.TASK_CONFIG['penalty_range']
        
        # Generate execution time based on utilization
        # For hard real-time tasks, use smaller execution times
        if task_type == TaskType.HARD_REAL_TIME:
            execution_time = random.uniform(exec_range[0], exec_range[1] * 0.7)
        else:
            execution_time = random.uniform(exec_range[0], exec_range[1])
        
        # Generate period (must be >= execution_time)
        min_period = max(execution_time * 2, period_range[0])
        period = random.uniform(min_period, period_range[1])
        
        # Generate deadline based on task type
        if task_type == TaskType.HARD_REAL_TIME:
            # Hard real-time: deadline <= period
            deadline = random.uniform(execution_time, period)
        else:
            # Soft real-time: deadline can be > period
            deadline = random.uniform(period * 0.8, period * 1.5)
        
        # Generate profit and penalty
        profit = random.uniform(profit_range[0], profit_range[1])
        penalty = random.uniform(penalty_range[0], penalty_range[1])
        
        # Ensure penalty is proportional to profit
        penalty = min(penalty, profit * 0.8)
        
        return {
            'task_id': f"uunifast_task_{task_id}",
            'task_type': task_type,
            'execution_time': execution_time,
            'deadline': deadline,
            'period': period,
            'utilization': utilization,
            'profit': profit,
            'penalty': penalty
        }
    
    def generate_heterogeneous_tasks(self, 
                                   num_tasks: int, 
                                   server_capacities: List[int]) -> List[Dict[str, Any]]:
        """
        Generate tasks considering server heterogeneity
        
        Args:
            num_tasks: Number of tasks to generate
            server_capacities: List of server CPU core counts
            
        Returns:
            List of task parameter dictionaries
        """
        # Calculate total system capacity
        total_capacity = sum(server_capacities)
        
        # Generate tasks with utilization proportional to capacity
        tasks = []
        
        for i in range(num_tasks):
            # Assign task to a server based on capacity
            server_idx = i % len(server_capacities)
            server_capacity = server_capacities[server_idx]
            
            # Generate task parameters
            task_params = self._generate_task_parameters(
                i, 
                random.uniform(0.1, 0.3),  # Utilization per core
                random.choice([TaskType.HARD_REAL_TIME, TaskType.SOFT_REAL_TIME]),
                self.config
            )
            
            # Adjust execution time based on server capacity
            task_params['execution_time'] *= (8 / server_capacity)  # Normalize to 8-core server
            task_params['assigned_server_capacity'] = server_capacity
            
            tasks.append(task_params)
        
        return tasks
    
    def validate_task_set(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate generated task set
        
        Args:
            tasks: List of task parameters
            
        Returns:
            Validation results
        """
        if not tasks:
            return {'valid': False, 'errors': ['No tasks generated']}
        
        errors = []
        warnings = []
        
        total_utilization = 0.0
        hard_tasks = 0
        soft_tasks = 0
        
        for i, task in enumerate(tasks):
            # Check execution time
            if task['execution_time'] <= 0:
                errors.append(f"Task {i}: Invalid execution time {task['execution_time']}")
            
            # Check deadline
            if task['deadline'] <= task['execution_time']:
                errors.append(f"Task {i}: Deadline {task['deadline']} <= execution time {task['execution_time']}")
            
            # Check period
            if task['period'] < task['execution_time']:
                errors.append(f"Task {i}: Period {task['period']} < execution time {task['execution_time']}")
            
            # Check utilization
            if task['utilization'] <= 0 or task['utilization'] > 1:
                errors.append(f"Task {i}: Invalid utilization {task['utilization']}")
            
            # Check profit and penalty
            if task['profit'] <= 0:
                errors.append(f"Task {i}: Invalid profit {task['profit']}")
            if task['penalty'] < 0:
                errors.append(f"Task {i}: Invalid penalty {task['penalty']}")
            
            # Accumulate statistics
            total_utilization += task['utilization']
            
            if task['task_type'] == TaskType.HARD_REAL_TIME:
                hard_tasks += 1
            else:
                soft_tasks += 1
        
        # Check total utilization
        if total_utilization > 1.0:
            warnings.append(f"Total utilization {total_utilization:.3f} > 1.0")
        
        # Check task type distribution
        expected_hard_ratio = self.config.get('hard_tasks_ratio', 0.3)
        actual_hard_ratio = hard_tasks / len(tasks)
        if abs(actual_hard_ratio - expected_hard_ratio) > 0.1:
            warnings.append(f"Hard task ratio {actual_hard_ratio:.3f} differs from expected {expected_hard_ratio}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'total_utilization': total_utilization,
            'hard_tasks': hard_tasks,
            'soft_tasks': soft_tasks,
            'total_tasks': len(tasks)
        }
    
    def __str__(self) -> str:
        """String representation of the UUNIFAST generator"""
        return f"UUNIFASTGenerator(total_utilization={self.config.get('total_utilization', 0.8)})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the UUNIFAST generator"""
        return self.__str__()
