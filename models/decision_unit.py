"""
Decision Unit model for IoT Edge Computing Real-Time Scheduling System
Coordinates task scheduling and allocation using different optimization algorithms
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from .task import Task, TaskStatus
from .edge_server import EdgeServer
from .iot_device import IoTDevice
import numpy as np


class SchedulingAlgorithm(Enum):
    """Scheduling algorithm enumeration"""
    MBPT = "mbpt"
    PSO = "pso"
    GENETIC = "genetic"
    EDF = "edf"  # Earliest Deadline First
    ROUND_ROBIN = "round_robin"


class DecisionStatus(Enum):
    """Decision unit status enumeration"""
    IDLE = "idle"
    PROCESSING = "processing"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class DecisionUnit:
    """
    Decision Unit model representing the central scheduling coordinator
    
    Attributes:
        unit_id: Unique identifier for the decision unit
        location: Geographic location coordinates
        status: Current status of the decision unit
        scheduling_algorithm: Currently active scheduling algorithm
        edge_servers: List of registered edge servers
        iot_devices: List of registered IoT devices
        task_queue: Queue of pending tasks
        scheduled_tasks: Dictionary of scheduled tasks by server
        completed_tasks: List of completed tasks
        failed_tasks: List of failed tasks
        performance_metrics: Dictionary of performance metrics
        algorithm_performance: Performance tracking for each algorithm
        fault_tolerance_config: Fault tolerance configuration
    """
    
    unit_id: str
    location: tuple
    status: DecisionStatus = DecisionStatus.IDLE
    scheduling_algorithm: str = "mbpt"
    edge_servers: List[EdgeServer] = field(default_factory=list)
    iot_devices: List[IoTDevice] = field(default_factory=list)
    task_queue: List[Task] = field(default_factory=list)
    scheduled_tasks: Dict[str, List[Task]] = field(default_factory=dict)
    completed_tasks: List[Task] = field(default_factory=list)
    failed_tasks: List[Task] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    algorithm_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fault_tolerance_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if not self.unit_id:
            self.unit_id = str(uuid.uuid4())
        if not self.fault_tolerance_config:
            self.fault_tolerance_config = {
                'replication_factor': 3,
                'checkpoint_interval': 5,
                'recovery_strategy': 'fast_restart'
            }
    
    def register_edge_server(self, server: EdgeServer):
        """
        Register an edge server with the decision unit
        
        Args:
            server: Edge server to register
        """
        if server not in self.edge_servers:
            self.edge_servers.append(server)
            self.scheduled_tasks[server.server_id] = []
    
    def register_iot_device(self, device: IoTDevice):
        """
        Register an IoT device with the decision unit
        
        Args:
            device: IoT device to register
        """
        if device not in self.iot_devices:
            self.iot_devices.append(device)
    
    def receive_task(self, task: Task, device: IoTDevice):
        """
        Receive a task from an IoT device
        
        Args:
            task: Task to receive
            device: IoT device that generated the task
        """
        # Add task to queue
        self.task_queue.append(task)
        # print("task added to queue")
        
        # Update task status
        task.update_status(TaskStatus.PENDING)
        
        # Update performance metrics
        self._update_received_task_metrics(task, device)
    
    def schedule_tasks(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using the active algorithm
        
        Returns:
            Dictionary mapping server IDs to assigned tasks
        """
        if not self.task_queue or not self.edge_servers:
            return {}
        
        self.status = DecisionStatus.OPTIMIZING
        
        try:
            # Apply scheduling algorithm
            if self.scheduling_algorithm == "mbpt":
                assignments = self._schedule_with_mbpt()
            elif self.scheduling_algorithm == "pso":
                assignments = self._schedule_with_pso()
            elif self.scheduling_algorithm == "genetic":
                assignments = self._schedule_with_genetic()
            elif self.scheduling_algorithm == "edf":
                assignments = self._schedule_with_edf()
            elif self.scheduling_algorithm == "round_robin":
                assignments = self._schedule_with_round_robin()
            else:
                # Default to MBPT
                assignments = self._schedule_with_mbpt()
            
            # Apply fault tolerance
            assignments = self._apply_fault_tolerance(assignments)
            
            # Update scheduled tasks
            self._update_scheduled_tasks(assignments)
            
            # Clear task queue
            # self.task_queue.clear()
            
            self.status = DecisionStatus.IDLE
            return assignments
            
        except Exception as e:
            self.status = DecisionStatus.ERROR
            print(f"Error in task scheduling: {e}")
            return {}
    
    def _schedule_with_mbpt(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using MBPT algorithm
        
        Returns:
            Task assignments by server
        """
        # Sort tasks by MBPT priority
        sorted_tasks = sorted(self.task_queue, 
                            key=lambda t: t.calculate_priority('mbpt'), 
                            reverse=True)
        
        # Sort servers by performance score
        sorted_servers = sorted(self.edge_servers, 
                              key=lambda s: s.get_performance_score(), 
                              reverse=True)

        return self._assign_tasks_to_servers(sorted_tasks, sorted_servers)
    
    def _schedule_with_pso(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using PSO algorithm
        
        Returns:
            Task assignments by server
        """
        # For simplicity, use a simplified PSO approach
        # In practice, this would call the actual PSO algorithm
        sorted_tasks = sorted(self.task_queue, 
                            key=lambda t: t.profit / max(t.execution_time, 1), 
                            reverse=True)
        
        sorted_servers = sorted(self.edge_servers, 
                              key=lambda s: s.cpu_cores, 
                              reverse=True)
        
        return self._assign_tasks_to_servers(sorted_tasks, sorted_servers)
    
    def _schedule_with_genetic(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using Genetic Algorithm
        
        Returns:
            Task assignments by server
        """
        # For simplicity, use a simplified GA approach
        # In practice, this would call the actual GA algorithm
        sorted_tasks = sorted(self.task_queue, 
                            key=lambda t: t.deadline, 
                            reverse=False)  # EDF-like
        
        sorted_servers = sorted(self.edge_servers, 
                              key=lambda s: s.memory_gb, 
                              reverse=True)
        
        return self._assign_tasks_to_servers(sorted_tasks, sorted_servers)
    
    def _schedule_with_edf(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using Earliest Deadline First
        
        Returns:
            Task assignments by server
        """
        # Sort tasks by deadline (earliest first)
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.deadline)
        
        # Sort servers by availability
        sorted_servers = sorted(self.edge_servers, 
                              key=lambda s: s.get_availability(), 
                              reverse=True)
        
        return self._assign_tasks_to_servers(sorted_tasks, sorted_servers)
    
    def _schedule_with_round_robin(self) -> Dict[str, List[Task]]:
        """
        Schedule tasks using Round Robin approach
        
        Returns:
            Task assignments by server
        """
        # Round robin assignment
        assignments = {server.server_id: [] for server in self.edge_servers}
        
        for i, task in enumerate(self.task_queue):
            server_idx = i % len(self.edge_servers)
            server = self.edge_servers[server_idx]
            assignments[server.server_id].append(task)
        
        return assignments
    
    def _assign_tasks_to_servers(self, 
                                sorted_tasks: List[Task], 
                                sorted_servers: List[EdgeServer]) -> Dict[str, List[Task]]:
        """
        Assign tasks to servers using best-fit approach
        
        Args:
            sorted_tasks: Tasks sorted by priority
            sorted_servers: Servers sorted by capability
            
        Returns:
            Task assignments by server
        """
        assignments = {server.server_id: [] for server in self.edge_servers}
        # print(" trying to assign ", sorted_tasks)

        for task in sorted_tasks:
            assigned = False

            # Try to assign to best available server
            for server in sorted_servers:
                # print(" trying to assign ", task)
                if server.can_accept_task(task.utilization):
                    print(f"Assigning task {task.task_id} to server {server.server_id}")
                    assignments[server.server_id].append(task)
                    server.assign_task(task.task_id, task.utilization)
                    assigned = True
                    break
            
            if not assigned:
                # Task couldn't be assigned, add to failed tasks
                self.failed_tasks.append(task)
                # task.update_status(TaskStatus.FAILED)
        
        return assignments
    
    def _apply_fault_tolerance(self, assignments: Dict[str, List[Task]]) -> Dict[str, List[Task]]:
        """
        Apply fault tolerance measures to task assignments
        
        Args:
            assignments: Original task assignments
            
        Returns:
            Fault-tolerant task assignments
        """
        replication_factor = self.fault_tolerance_config['replication_factor']
        
        for server_id, tasks in assignments.items():
            for task in tasks:
                # Create replicas
                for _ in range(replication_factor - 1):
                    replica = task.create_replica()
                    
                    # Find alternative server for replica
                    for alt_server in self.edge_servers:
                        if (alt_server.server_id != server_id and 
                            alt_server.can_accept_task(replica.utilization)):
                            if alt_server.server_id not in assignments:
                                assignments[alt_server.server_id] = []
                            assignments[alt_server.server_id].append(replica)
                            alt_server.assign_task(replica.task_id, replica.utilization)
                            break
        
        return assignments
    
    def _update_scheduled_tasks(self, assignments: Dict[str, List[Task]]):
        """
        Update the scheduled tasks tracking
        
        Args:
            assignments: Task assignments by server
        """
        for server_id, tasks in assignments.items():
            if server_id in self.scheduled_tasks:
                self.scheduled_tasks[server_id].extend(tasks)
            
            # Update task status
            for task in tasks:
                print(task.task_id, " Scheduled ")
                task.update_status(TaskStatus.SCHEDULED)
                task.assigned_server = server_id
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tasks in the system
        
        Returns:
            Dictionary with task statistics
        """
        total_tasks = len(self.task_queue) + sum(len(tasks) for tasks in self.scheduled_tasks.values())
        completed_tasks = len(self.completed_tasks)
        failed_tasks = len(self.failed_tasks)
        
        return {
            'total_tasks': total_tasks,
            'queued_tasks': len(self.task_queue),
            'scheduled_tasks': sum(len(tasks) for tasks in self.scheduled_tasks.values()),
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': completed_tasks / max(total_tasks, 1)
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """
        Get overall system performance metrics
        
        Returns:
            Dictionary with system performance metrics
        """
        if not self.edge_servers:
            return {}
        
        # Calculate average server performance
        avg_load = np.mean([server.current_load for server in self.edge_servers])
        avg_availability = np.mean([server.get_availability() for server in self.edge_servers])
        total_energy = sum(server.total_energy_consumed for server in self.edge_servers)
        
        # Calculate task completion metrics
        task_stats = self.get_task_statistics()
        
        return {
            'average_server_load': avg_load,
            'average_server_availability': avg_availability,
            'total_energy_consumption': total_energy,
            'task_success_rate': task_stats['success_rate'],
            'active_servers': len([s for s in self.edge_servers if s.status.value == 'available']),
            'total_registered_devices': len(self.iot_devices)
        }
    
    def _update_received_task_metrics(self, task: Task, device: IoTDevice):
        """
        Update metrics when receiving a task
        
        Args:
            task: Received task
            device: IoT device that generated the task
        """
        if 'total_received_tasks' not in self.performance_metrics:
            self.performance_metrics['total_received_tasks'] = 0
            self.performance_metrics['tasks_by_device_type'] = {}
            self.performance_metrics['total_profit_potential'] = 0.0
        
        self.performance_metrics['total_received_tasks'] += 1
        self.performance_metrics['total_profit_potential'] += task.profit
        
        device_type = device.device_type.value
        if device_type not in self.performance_metrics['tasks_by_device_type']:
            self.performance_metrics['tasks_by_device_type'][device_type] = 0
        self.performance_metrics['tasks_by_device_type'][device_type] += 1
    
    def to_dict(self) -> dict:
        """
        Convert decision unit to dictionary representation
        
        Returns:
            Dictionary representation of the decision unit
        """
        return {
            'unit_id': self.unit_id,
            'location': self.location,
            'status': self.status.value,
            'scheduling_algorithm': self.scheduling_algorithm,
            'registered_servers': len(self.edge_servers),
            'registered_devices': len(self.iot_devices),
            'queued_tasks': len(self.task_queue),
            'scheduled_tasks': sum(len(tasks) for tasks in self.scheduled_tasks.values()),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'task_statistics': self.get_task_statistics(),
            'system_performance': self.get_system_performance()
        }
    
    def __str__(self) -> str:
        """String representation of the decision unit"""
        return (f"DecisionUnit(id={self.unit_id}, algorithm={self.scheduling_algorithm}, "
                f"status={self.status.value}, servers={len(self.edge_servers)})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the decision unit"""
        return self.__str__()
