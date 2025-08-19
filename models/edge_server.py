"""
Edge Server model for IoT Edge Computing Real-Time Scheduling System
Represents a heterogeneous edge server with fault tolerance capabilities
"""

import uuid
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class ServerStatus(Enum):
    """Server status enumeration"""
    AVAILABLE = "available"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


@dataclass
class EdgeServer:
    """
    Edge Server model representing a heterogeneous computing node
    
    Attributes:
        server_id: Unique identifier for the server
        cpu_cores: Number of CPU cores available
        memory_gb: Available memory in GB
        storage_gb: Available storage in GB
        energy_efficiency: Energy efficiency factor (0.0-1.0)
        failure_rate: Probability of failure per time unit
        recovery_time: Time required for recovery after failure
        status: Current status of the server
        current_load: Current CPU utilization (0.0-1.0)
        assigned_tasks: List of tasks currently assigned
        task_queue: Queue of pending tasks
        energy_consumption: Current energy consumption
        total_energy_consumed: Total energy consumed over time
        failure_history: History of failures and recoveries
        performance_metrics: Dictionary of performance metrics
        location: Geographic location coordinates
        network_latency: Network latency to decision unit
    """
    
    server_id: str
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    energy_efficiency: float
    failure_rate: float
    recovery_time: float
    status: ServerStatus = ServerStatus.AVAILABLE
    current_load: float = 0.0
    assigned_tasks: List[str] = field(default_factory=list)
    task_queue: List[str] = field(default_factory=list)
    energy_consumption: float = 0.0
    total_energy_consumed: float = 0.0
    failure_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    location: Optional[tuple] = None
    network_latency: float = 0.0
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if not self.server_id:
            self.server_id = str(uuid.uuid4())
        if self.location is None:
            # Random location within a 100x100 grid
            self.location = (random.uniform(0, 100), random.uniform(0, 100))
        if self.network_latency == 0.0:
            # Random network latency between 1-10 time units
            self.network_latency = random.uniform(1, 10)
    
    def can_accept_task(self, task_utilization: float) -> bool:
        """
        Check if server can accept a new task
        
        Args:
            task_utilization: CPU utilization requirement of the task
            
        Returns:
            True if server can accept the task, False otherwise
        """
        # print(self.status)
        if self.status != ServerStatus.AVAILABLE:
            return False
        
        # Check if adding the task would exceed capacity
        # print('-------', self.current_load, " -------", task_utilization)
        new_load = self.current_load + task_utilization
        # print(len(self.assigned_tasks) )
        return new_load <= 1.0 and len(self.assigned_tasks) < self.cpu_cores
    
    def assign_task(self, task_id: str, task_utilization: float) -> bool:
        """
        Assign a task to this server
        
        Args:
            task_id: ID of the task to assign
            task_utilization: CPU utilization requirement
            
        Returns:
            True if task was assigned successfully, False otherwise
        """
        if not self.can_accept_task(task_utilization):
            return False
        
        self.assigned_tasks.append(task_id)
        self.current_load += task_utilization
        self.energy_consumption = self._calculate_energy_consumption()
        
        # Update status based on load
        if self.current_load > 0.8:
            pass
            # self.status = ServerStatus.OVERLOADED
        elif self.current_load > 0.0:
            pass
            # self.status = ServerStatus.BUSY
        
        return True
    
    def remove_task(self, task_id: str, task_utilization: float):
        """
        Remove a completed task from the server
        
        Args:
            task_id: ID of the task to remove
            task_utilization: CPU utilization of the task
        """
        if task_id in self.assigned_tasks:
            self.assigned_tasks.remove(task_id)
            self.current_load = max(0.0, self.current_load - task_utilization)
            self.energy_consumption = self._calculate_energy_consumption()
            
            # Update status based on new load
            if self.current_load == 0.0:
                self.status = ServerStatus.AVAILABLE
            elif self.current_load <= 0.8:
                pass
                # self.status = ServerStatus.BUSY
    
    def add_to_queue(self, task_id: str):
        """
        Add a task to the server's queue
        
        Args:
            task_id: ID of the task to queue
        """
        if task_id not in self.task_queue:
            self.task_queue.append(task_id)
    
    def get_next_queued_task(self) -> Optional[str]:
        """
        Get the next task from the queue (FIFO)
        
        Returns:
            Next task ID or None if queue is empty
        """
        if self.task_queue:
            return self.task_queue.pop(0)
        return None
    
    def _calculate_energy_consumption(self) -> float:
        """
        Calculate current energy consumption based on load
        
        Returns:
            Current energy consumption in watts
        """
        # Base energy consumption + load-dependent consumption
        base_consumption = 50.0  # Base power in watts
        load_factor = 1.0 + (self.current_load * 2.0)  # Load multiplier
        efficiency_factor = 1.0 / self.energy_efficiency
        
        return base_consumption * load_factor * efficiency_factor
    
    def update_energy_consumption(self, time_step: float):
        """
        Update total energy consumption over time
        
        Args:
            time_step: Time step for energy calculation
        """
        self.total_energy_consumed += self.energy_consumption * time_step
    
    def check_failure(self, current_time: float) -> bool:
        """
        Check if server fails based on failure rate
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if failure occurs, False otherwise
        """
        if self.status == ServerStatus.FAILED or self.status == ServerStatus.RECOVERING:
            return False
        
        # Random failure check based on failure rate
        if random.random() < self.failure_rate:
            self._handle_failure(current_time)
            return True
        return False
    
    def _handle_failure(self, current_time: float):
        """
        Handle server failure
        
        Args:
            current_time: Time when failure occurred
        """
        self.status = ServerStatus.FAILED
        self.failure_history.append({
            'failure_time': current_time,
            'assigned_tasks': self.assigned_tasks.copy(),
            'queue_length': len(self.task_queue)
        })
        
        # Clear all assigned tasks and queue
        failed_tasks = self.assigned_tasks.copy()
        self.assigned_tasks.clear()
        self.task_queue.clear()
        self.current_load = 0.0
        self.energy_consumption = 0.0
        
        return failed_tasks
    
    def start_recovery(self, current_time: float):
        """
        Start server recovery process
        
        Args:
            current_time: Time when recovery started
        """
        if self.status == ServerStatus.FAILED:
            self.status = ServerStatus.RECOVERING
            self.performance_metrics['recovery_start_time'] = current_time
    
    def complete_recovery(self, current_time: float):
        """
        Complete server recovery
        
        Args:
            current_time: Time when recovery completed
        """
        if self.status == ServerStatus.RECOVERING:
            self.status = ServerStatus.AVAILABLE
            recovery_time = current_time - self.performance_metrics.get('recovery_start_time', current_time)
            self.performance_metrics['last_recovery_time'] = recovery_time
    
    def get_availability(self) -> float:
        """
        Calculate server availability based on failure history
        
        Returns:
            Availability percentage (0.0-1.0)
        """
        if not self.failure_history:
            return 1.0
        
        total_failures = len(self.failure_history)
        total_recovery_time = sum(failure.get('recovery_time', self.recovery_time) 
                                for failure in self.failure_history)
        
        # Simplified availability calculation
        return 1.0 / (1.0 + total_failures * self.failure_rate)
    
    def get_performance_score(self) -> float:
        """
        Calculate overall performance score
        
        Returns:
            Performance score (0.0-1.0)
        """
        # Factors: availability, energy efficiency, current load
        availability = self.get_availability()
        energy_score = self.energy_efficiency
        load_score = 1.0 - self.current_load
        
        # Weighted average
        return (0.4 * availability + 0.3 * energy_score + 0.3 * load_score)
    
    def to_dict(self) -> dict:
        """
        Convert server to dictionary representation
        
        Returns:
            Dictionary representation of the server
        """
        return {
            'server_id': self.server_id,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'storage_gb': self.storage_gb,
            'energy_efficiency': self.energy_efficiency,
            'status': self.status.value,
            'current_load': self.current_load,
            'assigned_tasks_count': len(self.assigned_tasks),
            'queue_length': len(self.task_queue),
            'energy_consumption': self.energy_consumption,
            'total_energy_consumed': self.total_energy_consumed,
            'availability': self.get_availability(),
            'performance_score': self.get_performance_score(),
            'location': self.location,
            'network_latency': self.network_latency
        }
    
    def __str__(self) -> str:
        """String representation of the server"""
        return (f"EdgeServer(id={self.server_id}, cores={self.cpu_cores}, "
                f"load={self.current_load:.2f}, status={self.status.value})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the server"""
        return self.__str__()
