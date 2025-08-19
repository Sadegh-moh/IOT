"""
Task model for IoT Edge Computing Real-Time Scheduling System
Represents a soft periodic task with economic value considerations
"""

import uuid
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Task type enumeration"""
    HARD_REAL_TIME = "hard_real_time"
    SOFT_REAL_TIME = "soft_real_time"


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    MISSED_DEADLINE = "missed_deadline"


@dataclass
class Task:
    """
    Task model representing a soft periodic task with economic value
    
    Attributes:
        task_id: Unique identifier for the task
        task_type: Type of task (hard or soft real-time)
        execution_time: Time required to execute the task
        deadline: Absolute deadline for task completion
        period: Period of the periodic task
        utilization: CPU utilization requirement
        profit: Monetary profit if completed on time
        penalty: Monetary penalty if delayed or failed
        priority: Calculated priority based on MBPT algorithm
        status: Current status of the task
        arrival_time: Time when task arrived
        start_time: Time when task started execution
        completion_time: Time when task completed
        assigned_server: Edge server assigned to this task
        replicas: List of task replicas for fault tolerance
        checkpoint_data: Data for checkpointing and recovery
    """
    
    task_id: str
    task_type: TaskType
    execution_time: float
    deadline: float
    period: float
    utilization: float
    profit: float
    penalty: float
    priority: float = 0.0
    status: TaskStatus = TaskStatus.PENDING
    arrival_time: float = 0.0
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_server: Optional[str] = None
    replicas: List['Task'] = None
    checkpoint_data: dict = None
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.replicas is None:
            self.replicas = []
        if self.checkpoint_data is None:
            self.checkpoint_data = {}
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
    
    def calculate_priority(self, algorithm: str = 'mbpt') -> float:
        """
        Calculate task priority based on the specified algorithm
        
        Args:
            algorithm: Algorithm to use for priority calculation
            
        Returns:
            Calculated priority value
        """
        if algorithm == 'mbpt':
            # MBPT: Maximum Benefit Per Unit Time
            if self.execution_time > 0:
                self.priority = self.profit / self.execution_time
            else:
                self.priority = 0.0
        elif algorithm == 'deadline':
            # Earliest Deadline First
            self.priority = 1.0 / self.deadline
        elif algorithm == 'utilization':
            # Utilization-based priority
            self.priority = self.utilization
        else:
            # Default: profit-based priority
            self.priority = self.profit
            
        return self.priority
    
    def is_deadline_missed(self, current_time: float) -> bool:
        """
        Check if task deadline is missed
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if deadline is missed, False otherwise
        """
        if self.status == TaskStatus.COMPLETED:
            return self.completion_time > self.deadline
        return current_time > self.deadline
    
    def calculate_profit_loss(self, current_time: float) -> float:
        """
        Calculate profit or loss based on completion status
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Net profit (positive) or loss (negative)
        """
        if self.status == TaskStatus.COMPLETED:
            if self.completion_time <= self.deadline:
                return self.profit
            else:
                # Penalty for missing deadline
                delay = self.completion_time - self.deadline
                return self.profit - (self.penalty * delay / self.deadline)
        elif self.status == TaskStatus.MISSED_DEADLINE:
            return -self.penalty
        elif self.status == TaskStatus.FAILED:
            return -self.penalty
        else:
            return 0.0
    
    def get_remaining_time(self, current_time: float) -> float:
        """
        Get remaining time until deadline
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Remaining time until deadline
        """
        return max(0, self.deadline - current_time)
    
    def get_slack_time(self, current_time: float) -> float:
        """
        Calculate slack time (time available minus execution time)
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Slack time available
        """
        remaining = self.get_remaining_time(current_time)
        return max(0, remaining - self.execution_time)
    
    def create_replica(self) -> 'Task':
        """
        Create a replica of this task for fault tolerance
        
        Returns:
            New task replica
        """
        replica = Task(
            task_id=f"{self.task_id}_replica_{len(self.replicas)}",
            task_type=self.task_type,
            execution_time=self.execution_time,
            deadline=self.deadline,
            period=self.period,
            utilization=self.utilization,
            profit=self.profit,
            penalty=self.penalty,
            priority=self.priority,
            arrival_time=self.arrival_time
        )
        self.replicas.append(replica)
        return replica
    
    def update_status(self, new_status: TaskStatus, timestamp: float = None):
        """
        Update task status and related timestamps
        
        Args:
            new_status: New status to set
            timestamp: Timestamp for the status change
        """
        self.status = new_status
        
        if timestamp is not None:
            if new_status == TaskStatus.EXECUTING:
                self.start_time = timestamp
            elif new_status == TaskStatus.COMPLETED:
                self.completion_time = timestamp
    
    def to_dict(self) -> dict:
        """
        Convert task to dictionary representation
        
        Returns:
            Dictionary representation of the task
        """
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'execution_time': self.execution_time,
            'deadline': self.deadline,
            'period': self.period,
            'utilization': self.utilization,
            'profit': self.profit,
            'penalty': self.penalty,
            'priority': self.priority,
            'status': self.status.value,
            'arrival_time': self.arrival_time,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'assigned_server': self.assigned_server,
            'replica_count': len(self.replicas)
        }
    
    def __str__(self) -> str:
        """String representation of the task"""
        return (f"Task(id={self.task_id}, type={self.task_type.value}, "
                f"exec={self.execution_time}, deadline={self.deadline}, "
                f"profit={self.profit}, priority={self.priority:.2f})")
    
    def __repr__(self) -> str:
        """Detailed string representation of the task"""
        return self.__str__()
