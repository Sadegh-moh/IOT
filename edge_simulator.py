"""
EdgeSimPy-based IoT Edge Computing Simulator
Integrates EdgeSimPy with custom MBPT scheduling algorithms
"""

import sys
import os
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd

# Add EdgeSimPy to path
sys.path.append('edge_sim_py')

# Import EdgeSimPy components

import edge_sim_py
from edge_sim_py import Simulator
from edge_sim_py.components import *


from models.task import Task, TaskType, TaskStatus
from models.edge_server import EdgeServer as CustomEdgeServer, ServerStatus
from models.iot_device import IoTDevice, DeviceType, DeviceStatus
from models.decision_unit import DecisionUnit
from algorithms.mbpt import MBPTAlgorithm
from algorithms.pso import PSOAlgorithm
from algorithms.genetic_algorithm import GeneticAlgorithm
from algorithms.uunifast import UUNIFASTGenerator
import config


class EdgeComputingSimulator:
    """
    Main simulator class that integrates EdgeSimPy with custom scheduling algorithms
    """
    
    def __init__(self, scenario_config: dict):
        """
        Initialize the simulator
        
        Args:
            scenario_config: Configuration for the simulation scenario
        """
        self.scenario_config = scenario_config
        self.current_time = 0.0
        self.time_step = config.TIME_STEP
        
        # Initialize components
        self.edge_sim = None
        self.decision_unit = None
        self.edge_servers = []
        self.iot_devices = []
        self.tasks = []
        
        # Performance tracking
        self.performance_metrics = {
            'mbpt': [],
            'pso': [],
            'genetic': []
        }
        
        # Initialize algorithms
        self.mbpt_algorithm = MBPTAlgorithm()
        self.pso_algorithm = PSOAlgorithm()
        self.ga_algorithm = GeneticAlgorithm()
        
        # Task generator
        self.task_generator = UUNIFASTGenerator()
        
        # Results storage
        self.results = {
            'scenario': scenario_config,
            'metrics': {},
            'task_specifications': [],
            'comparisons': {}
        }
    
    def setup_scenario(self):
        """Setup the simulation scenario with EdgeSimPy"""
        try:
            # Initialize EdgeSimPy
            self.edge_sim = Simulator()
            
            # Create edge servers
            self._create_edge_servers()
            
            # Create IoT devices
            self._create_iot_devices()
            
            # Create decision unit
            self._create_decision_unit()
            
            # Generate tasks using UUNIFAST
            self._generate_tasks()
            
            print(f"Scenario setup complete: {len(self.edge_servers)} servers, "
                  f"{len(self.iot_devices)} devices, {len(self.tasks)} tasks")
            
        except Exception as e:
            print(f"Error setting up scenario: {e}")
            # Fallback to mock setup
            self._setup_mock_scenario()
    
    def _create_edge_servers(self):
        """Create heterogeneous edge servers"""
        num_servers = self.scenario_config.get('edge_servers', 100)
        
        for i in range(num_servers):
            server = CustomEdgeServer(
                server_id=f"edge_server_{i}",
                cpu_cores=random.randint(
                    config.EDGE_SERVER_CONFIG['cpu_cores_range'][0],
                    config.EDGE_SERVER_CONFIG['cpu_cores_range'][1]
                ),
                memory_gb=random.uniform(
                    config.EDGE_SERVER_CONFIG['memory_range'][0],
                    config.EDGE_SERVER_CONFIG['memory_range'][1]
                ),
                storage_gb=random.uniform(
                    config.EDGE_SERVER_CONFIG['storage_range'][0],
                    config.EDGE_SERVER_CONFIG['storage_range'][1]
                ),
                energy_efficiency=random.uniform(0.6, 1.0),
                failure_rate=config.EDGE_SERVER_CONFIG['failure_rate'],
                recovery_time=config.EDGE_SERVER_CONFIG['recovery_time']
            )
            self.edge_servers.append(server)
    
    def _create_iot_devices(self):
        """Create IoT devices"""
        num_devices = self.scenario_config.get('iot_devices', 300)
        
        for i in range(num_devices):
            device = IoTDevice(
                device_id=f"iot_device_{i}",
                device_type=random.choice(list(DeviceType)),
                location=(random.uniform(0, 100), random.uniform(0, 100))
            )
            self.iot_devices.append(device)
    
    def _create_decision_unit(self):
        """Create the decision unit"""
        self.decision_unit = DecisionUnit(
            unit_id="decision_unit_1",
            location=(50.0, 50.0)
        )
        
        # Register components
        for server in self.edge_servers:
            self.decision_unit.register_edge_server(server)
        
        for device in self.iot_devices:
            self.decision_unit.register_iot_device(device)
    
    def _generate_tasks(self):
        """Generate tasks using UUNIFAST algorithm"""
        tasks_per_device = self.scenario_config.get('tasks_per_device', 200)
        total_tasks = len(self.iot_devices) * tasks_per_device
        
        # Generate task parameters using UUNIFAST
        task_params = self.task_generator.generate_tasks(
            total_tasks, 
            config.UUNIFAST_CONFIG
        )
        
        # Create Task objects
        for i, params in enumerate(task_params):
            task = Task(
                task_id=f"task_{i}",
                task_type=params['task_type'],
                execution_time=params['execution_time'],
                deadline=params['deadline'],
                period=params['period'],
                utilization=params['utilization'],
                profit=params['profit'],
                penalty=params['penalty'],
                arrival_time=random.uniform(0, config.SIMULATION_TIME * 0.1)
            )
            
            # Calculate priority
            task.calculate_priority('mbpt')
            
            # Assign to random IoT device
            device = random.choice(self.iot_devices)
            device.generated_tasks.append(task)
            
            # Add task to decision unit's queue
            self.decision_unit.receive_task(task, device)
            
            self.tasks.append(task)
    
    def _setup_mock_scenario(self):
        """Setup mock scenario when EdgeSimPy is not available"""
        print("Setting up mock scenario...")
        
        # Create edge servers
        self._create_edge_servers()
        
        # Create IoT devices
        self._create_iot_devices()
        
        # Create decision unit
        self._create_decision_unit()
        
        # Generate tasks
        self._generate_tasks()
    
    def run_simulation(self):
        """Run the main simulation loop"""
        print(f"Starting simulation for {config.SIMULATION_TIME} time units...")
        
        start_time = time.time()
        
        # Run simulation with different algorithms
        algorithms = ['mbpt', 'pso', 'genetic']
        
        for algorithm in algorithms:
            print(f"\nRunning simulation with {algorithm.upper()} algorithm...")
            self._run_algorithm_simulation(algorithm)
        
        # Calculate final metrics
        self._calculate_final_metrics()
        
        # Generate results
        self._generate_results()
        
        simulation_time = time.time() - start_time
        print(f"\nSimulation completed in {simulation_time:.2f} seconds")
        
        return self.results
    
    def _run_algorithm_simulation(self, algorithm: str):
        """Run simulation with a specific algorithm"""
        # Reset system state
        self._reset_system_state()
        
        # Set algorithm in decision unit
        if algorithm == 'mbpt':
            self.decision_unit.scheduling_algorithm = 'mbpt'
        elif algorithm == 'pso':
            self.decision_unit.scheduling_algorithm = 'pso'
        elif algorithm == 'genetic':
            self.decision_unit.scheduling_algorithm = 'genetic'
        
        # Simulation loop
        for time_step in range(0, config.SIMULATION_TIME, self.time_step):
            self.current_time = time_step
            
            # Update system state
            self._update_system_state()
            
            # Schedule tasks
            self._schedule_tasks(algorithm)
            
            # Execute tasks
            self._execute_tasks()
            
            # Update metrics
            self._update_metrics(algorithm, time_step)
            
            # Progress reporting every 50 time steps (reduced from 100 for faster feedback)
            if time_step % 50 == 0:
                completed = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
                total = len(self.tasks)
                print(f"  Time {time_step}: {completed}/{total} tasks completed ({completed/total*100:.1f}%)")
            
            # Check for completion
            if self._is_simulation_complete():
                print(f"  Simulation completed early at time {time_step}")
                break
        
        # Store algorithm results - we'll collect metrics during the simulation
        # The metrics are already being stored in _update_metrics method
    
    def _reset_system_state(self):
        """Reset system state for new algorithm run"""
        # Reset all tasks to pending
        for task in self.tasks:
            task.status = TaskStatus.PENDING
            task.start_time = None
            task.completion_time = None
            task.assigned_server = None
        
        # Reset edge servers
        for server in self.edge_servers:
            server.assigned_tasks.clear()
            server.task_queue.clear()
            server.current_load = 0.0
            server.status = ServerStatus.AVAILABLE  # Fixed: Use enum instead of string
            server.total_energy_consumed = 0.0  # Reset energy consumption
        
        # Reset decision unit
        self.decision_unit.task_queue.clear()
        self.decision_unit.scheduled_tasks.clear()
        self.decision_unit.completed_tasks.clear()
        self.decision_unit.failed_tasks.clear()
        
        # Re-add all tasks to the decision unit's queue
        for task in self.tasks:
            # Find the IoT device that originally generated this task
            for device in self.iot_devices:
                if any(t.task_id == task.task_id for t in device.generated_tasks):
                    self.decision_unit.receive_task(task, device)
                    break
    
    def _update_system_state(self):
        """Update system state for current time step"""
        # Check for server failures
        for server in self.edge_servers:
            if server.check_failure(self.current_time):
                print(f"Server {server.server_id} failed at time {self.current_time}")
        
        # Update energy consumption
        for server in self.edge_servers:
            server.update_energy_consumption(self.time_step)
        
        # Generate new tasks from IoT devices
        for device in self.iot_devices:
            if random.random() < 0.01:  # 1% chance per time step
                task = device.generate_task(self.current_time, config.TASK_CONFIG)
                if task:
                    self.decision_unit.receive_task(task, device)
    
    def _schedule_tasks(self, algorithm: str):
        """Schedule tasks using the specified algorithm"""
        if not self.decision_unit.task_queue:
            return
        
        # Set the scheduling algorithm
        self.decision_unit.scheduling_algorithm = algorithm
        
        # Schedule tasks
        assignments = self.decision_unit.schedule_tasks()
        
        # Debug: Print scheduling information
        if assignments:
            total_assigned = sum(len(tasks) for tasks in assignments.values())
            # print(f"    Scheduled {total_assigned} tasks to {len(assignments)} servers")
        
        # Apply fault tolerance
        self._apply_fault_tolerance(assignments)
        
        # Move tasks from decision unit to servers
        for server_id, tasks in assignments.items():
            server = next((s for s in self.edge_servers if s.server_id == server_id), None)
            if server:
                for task in tasks:
                    # print("000000000" , task.status)
                    if task.status == TaskStatus.SCHEDULED:
                        # print(f"Task {task.task_id} scheduled at time {self.current_time}")
                        # task.update_status(TaskStatus.SCHEDULED)
                        task.assigned_server = server.server_id
                        server.assigned_tasks.append(task.task_id)
                        server.current_load += task.utilization
                        
                        # Add task to server's task queue for execution
                        server.task_queue.append(task.task_id)
    
    def _execute_tasks(self):
        """Execute scheduled tasks"""
        for server in self.edge_servers:
            if server.task_queue:
                # Process tasks in the queue
                completed_tasks = []
                for task_id in server.task_queue[:]:  # Copy list to avoid modification during iteration
                    task = self._find_task_by_id(task_id)
                    if task and (task.status == TaskStatus.EXECUTING or task.status == TaskStatus.SCHEDULED):
                        # Simulate task execution
                        if self._execute_single_task(task, server):
                            completed_tasks.append(task_id)
                            # Send result back to IoT device
                            self._send_task_result(task)
                
                # Remove completed tasks from both queue and assigned tasks
                for task_id in completed_tasks:
                    if task_id in server.task_queue:
                        server.task_queue.remove(task_id)
                    if task_id in server.assigned_tasks:
                        server.assigned_tasks.remove(task_id)
                        # Find the task to get its utilization
                        task = self._find_task_by_id(task_id)
                        if task:
                            server.current_load = max(0, server.current_load - task.utilization)
    
    def _execute_single_task(self, task: Task, server: CustomEdgeServer) -> bool:
        """Execute a single task on a server"""
        # Check if task can start
        if task.status == TaskStatus.SCHEDULED:
            task.update_status(TaskStatus.EXECUTING, self.current_time)
            task.start_time = self.current_time
                # Check if task is completed
        if task.status == TaskStatus.EXECUTING and task.start_time is not None:
            execution_progress = (self.current_time - task.start_time) / task.execution_time
            if execution_progress >= 1.0:
                task.update_status(TaskStatus.COMPLETED, self.current_time)
                task.completion_time = self.current_time
                return True
        
        return False
    
    def _send_task_result(self, task: Task):
        """Send task result back to IoT device"""
        # Find the IoT device that generated this task
        for device in self.iot_devices:
            if any(t.task_id == task.task_id for t in device.generated_tasks):
                result = {
                    'success': task.status == TaskStatus.COMPLETED,
                    'completion_time': task.completion_time,
                    'profit_loss': task.calculate_profit_loss(self.current_time)
                }
                device.receive_task_result(task.task_id, result, self.current_time)
                break
    
    def _apply_fault_tolerance(self, assignments: Dict[str, List[Task]]):
        """Apply fault tolerance measures"""
        replication_factor = config.FAULT_TOLERANCE['replication_factor']
        
        for server_id, tasks in assignments.items():
            for task in tasks:
                # Create replicas
                for _ in range(replication_factor - 1):
                    replica = task.create_replica()
                    
                    # Find alternative server for replica
                    for alt_server in self.edge_servers:
                        if (alt_server.server_id != server_id and 
                            alt_server.can_accept_task(replica.utilization)):
                            alt_server.assign_task(replica.task_id, replica.utilization)
                            break
    
    def _find_task_by_id(self, task_id: str) -> Optional[Task]:
        """Find task by ID"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def _update_metrics(self, algorithm: str, time_step: int):
        """Update performance metrics"""
        # Calculate current metrics with more realistic values
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(self.tasks)
        
        # Calculate profit based on completed tasks
        total_profit = 0
        for task in self.tasks:
            if task.status == TaskStatus.COMPLETED:
                # Profit for completed tasks
                total_profit += task.profit
            elif task.status == TaskStatus.FAILED or task.is_deadline_missed(self.current_time):
                # Penalty for failed or missed deadline tasks
                total_profit -= task.penalty
        
        # Calculate energy consumption
        total_energy = sum(s.total_energy_consumed for s in self.edge_servers)
        
        # Calculate makespan (time to complete all tasks)
        makespan = 0
        if completed_tasks > 0:
            completion_times = [t.completion_time for t in self.tasks if t.completion_time is not None]
            if completion_times:
                makespan = max(completion_times) - min(completion_times)
        
        metrics = {
            'time_step': time_step,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'failed_tasks': len([t for t in self.tasks if t.status == TaskStatus.FAILED]),
            'pending_tasks': len([t for t in self.tasks if t.status == TaskStatus.PENDING]),
            'total_profit': total_profit,
            'energy_consumption': total_energy,
            'makespan': makespan,
            'deadline_miss_ratio': self._calculate_deadline_miss_ratio(),
            'reliability': self._calculate_system_reliability(),
            'completion_rate': completed_tasks / max(total_tasks, 1)
        }
        
        # Store metrics
        if algorithm not in self.performance_metrics:
            self.performance_metrics[algorithm] = []
        self.performance_metrics[algorithm].append(metrics)
    
    def _calculate_deadline_miss_ratio(self) -> float:
        """Calculate deadline miss ratio"""
        if not self.tasks:
            return 0.0
        
        missed_deadlines = sum(1 for task in self.tasks if task.is_deadline_missed(self.current_time))
        return missed_deadlines / len(self.tasks)
    
    def _calculate_system_reliability(self) -> float:
        """Calculate system reliability"""
        if not self.edge_servers:
            return 0.0
        
        # Calculate average server availability
        total_availability = sum(server.get_availability() for server in self.edge_servers)
        avg_availability = total_availability / len(self.edge_servers)
        
        # Apply fault tolerance factor
        fault_tolerance_factor = config.FAULT_TOLERANCE['replication_factor'] / 3.0
        reliability = avg_availability * fault_tolerance_factor
        
        return min(1.0, reliability)
    
    def _is_simulation_complete(self) -> bool:
        """Check if simulation is complete"""
        completed_tasks = len([t for t in self.tasks if t.status == TaskStatus.COMPLETED])
        total_tasks = len(self.tasks)
        
        return completed_tasks >= total_tasks * 0.95  # 95% completion
    

    
    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        for algorithm, metrics_list in self.performance_metrics.items():
            if not metrics_list:
                continue
            
            try:
                # Calculate averages with error handling
                final_metrics = {
                    'avg_completion_rate': np.mean([m.get('completion_rate', 0.0) for m in metrics_list]),
                    'avg_profit': np.mean([m.get('total_profit', 0.0) for m in metrics_list]),
                    'avg_energy_consumption': np.mean([m.get('energy_consumption', 0.0) for m in metrics_list]),
                    'avg_makespan': np.mean([m.get('makespan', 0.0) for m in metrics_list]),
                    'avg_deadline_miss_ratio': np.mean([m.get('deadline_miss_ratio', 0.0) for m in metrics_list]),
                    'avg_reliability': np.mean([m.get('reliability', 0.0) for m in metrics_list]),
                    'final_completion_rate': metrics_list[-1].get('completion_rate', 0.0),
                    'final_profit': metrics_list[-1].get('total_profit', 0.0),
                    'final_energy_consumption': metrics_list[-1].get('energy_consumption', 0.0),
                    'final_makespan': metrics_list[-1].get('makespan', 0.0)
                }
                
                self.results['metrics'][algorithm] = final_metrics
            except (KeyError, TypeError, ZeroDivisionError) as e:
                print(f"Warning: Error calculating metrics for {algorithm}: {e}")
                # Use default values if calculation fails
                self.results['metrics'][algorithm] = {
                    'avg_completion_rate': 0.0,
                    'avg_profit': 0.0,
                    'avg_energy_consumption': 0.0,
                    'avg_deadline_miss_ratio': 0.0,
                    'avg_reliability': 0.0,
                    'final_completion_rate': 0.0,
                    'final_profit': 0.0,
                    'final_energy_consumption': 0.0
                }
    
    def _generate_results(self):
        """Generate final results and comparisons"""
        try:
            # Task specifications table
            self.results['task_specifications'] = [task.to_dict() for task in self.tasks[:100]]  # First 100 tasks
            
            # Algorithm comparisons
            algorithms = list(self.results['metrics'].keys())
            if len(algorithms) >= 2:
                self.results['comparisons'] = {
                    'best_completion_rate': max(algorithms, key=lambda a: self.results['metrics'][a].get('final_completion_rate', 0.0)),
                    'best_profit': max(algorithms, key=lambda a: self.results['metrics'][a].get('final_profit', 0.0)),
                    'best_energy_efficiency': min(algorithms, key=lambda a: self.results['metrics'][a].get('final_energy_consumption', float('inf'))),
                    'best_reliability': max(algorithms, key=lambda a: self.results['metrics'][a].get('avg_reliability', 0.0))
                }
        except Exception as e:
            print(f"Warning: Error generating results: {e}")
            # Set default values
            self.results['task_specifications'] = []
            self.results['comparisons'] = {}
    
    def get_results(self) -> Dict[str, Any]:
        """Get simulation results"""
        return self.results


def run_scenario(scenario_name: str, scenario_config: dict):
    """Run a specific scenario"""
    print(f"\n{'='*50}")
    print(f"Running {scenario_name}")
    print(f"{'='*50}")
    
    # Create simulator
    simulator = EdgeComputingSimulator(scenario_config)
    
    # Setup scenario
    simulator.setup_scenario()
    
    # Run simulation
    results = simulator.run_simulation()
    
    return results


if __name__ == "__main__":
    # Run both scenarios
    print("IoT Edge Computing Real-Time Scheduling System")
    print("Using EdgeSimPy Integration")
    
    # Scenario A
    scenario_a_results = run_scenario("Scenario A", config.SCENARIO_A)
    
    # Scenario B
    scenario_b_results = run_scenario("Scenario B", config.SCENARIO_B)
    
    # Save results
    import json
    with open('simulation_results.json', 'w') as f:
        json.dump({
            'scenario_a': scenario_a_results,
            'scenario_b': scenario_b_results
        }, f, indent=2, default=str)
    
    print("\nSimulation completed. Results saved to 'simulation_results.json'")
