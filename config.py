"""
Configuration file for IoT Edge Computing Real-Time Scheduling System
"""

# Simulation Parameters
SIMULATION_TIME = 200  # Reduced from 1000 to 200 time units
TIME_STEP = 2  # Increased from 1 to 2 for faster simulation

# Scenario A: Reduced scale for faster execution
SCENARIO_A = {
    'iot_devices': 50,  # Reduced from 600-1000 to 50
    'edge_servers': 20,  # Reduced from 100 to 20
    'tasks_per_device': 30,  # Reduced from 200 to 30
    'environment': 'single'
}

# Scenario B: Reduced scale for faster execution
SCENARIO_B = {
    'iot_devices': 30,  # Reduced from 300 to 30
    'edge_servers': 15,  # Reduced from 30-100 to 15
    'tasks_per_device': 25,  # Reduced from 200 to 25
    'environment': 'distributed'
}

# Task Parameters
TASK_CONFIG = {
    'execution_time_range': (1, 20),  # Reduced from (1, 50) to (1, 20) time units
    'deadline_range': (5, 50),        # Reduced from (10, 200) to (5, 50) time units
    'period_range': (10, 100),        # Reduced from (20, 400) to (10, 100) time units
    'profit_range': (10, 500),        # Reduced from (10, 1000) to (10, 500) monetary units
    'penalty_range': (5, 200),        # Reduced from (5, 500) to (5, 200) monetary units
    'utilization_range': (0.01, 0.15),  # Fixed: Reduced from (0.1, 0.6) to (0.01, 0.15) - much smaller CPU utilization per task
    'reliability_target': 0.999       # 1 - 10^-3
}

# Edge Server Parameters
EDGE_SERVER_CONFIG = {
    'cpu_cores_range': (4, 16),
    'memory_range': (8, 64),          # GB
    'storage_range': (100, 1000),     # GB
    'energy_efficiency': 0.8,         # Energy efficiency factor
    'failure_rate': 0.001,            # Per time unit
    'recovery_time': 10               # Time units
}

# Algorithm Parameters
MBPT_CONFIG = {
    'profit_weight': 0.6,
    'deadline_weight': 0.3,
    'utilization_weight': 0.1
}

PSO_CONFIG = {
    'particles': 20,        # Reduced from 50 to 20
    'iterations': 30,       # Reduced from 100 to 30
    'cognitive_weight': 2.0,
    'social_weight': 2.0,
    'inertia_weight': 0.7
}

GA_CONFIG = {
    'population_size': 30,  # Reduced from 100 to 30
    'generations': 50,      # Reduced from 200 to 50
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'elite_size': 5         # Reduced from 10 to 5
}

# UUNIFAST Parameters
UUNIFAST_CONFIG = {
    'hard_tasks_ratio': 0.3,          # 30% hard real-time tasks
    'soft_tasks_ratio': 0.7,          # 70% soft real-time tasks
    'total_utilization': 0.8          # Total system utilization
}

# Fault Tolerance Parameters
FAULT_TOLERANCE = {
    'replication_factor': 2,          # Reduced from 3 to 2 replicas for faster execution
    'checkpoint_interval': 10,        # Increased from 5 to 10 time units
    'recovery_strategy': 'fast_restart'
}

# Performance Metrics
METRICS = {
    'qos_threshold': 0.95,            # Quality of Service threshold
    'reliability_threshold': 0.999,   # Reliability threshold
    'energy_threshold': 0.8,          # Energy efficiency threshold
    'deadline_miss_threshold': 0.05   # Maximum allowed deadline miss ratio
}

# Output Configuration
OUTPUT_CONFIG = {
    'charts_format': 'png',
    'dpi': 300,
    'figure_size': (12, 8),
    'save_results': True,
    'results_directory': 'results'
}
