"""
Main execution file for IoT Edge Computing Real-Time Scheduling System
Runs simulations for both scenarios and generates comprehensive results
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from edge_simulator import EdgeComputingSimulator, run_scenario
from results_visualizer import ResultsVisualizer
import config


def main():
    """Main execution function"""
    print("=" * 80)
    print("IoT Edge Computing Real-Time Scheduling System")
    print("Real-time scheduling based on economic value using MBPT algorithm")
    print("=" * 80)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    start_time = time.time()
    
    try:
        # Run Scenario A: 600-1000 IoT devices, 100 edge servers
        print("\n" + "="*60)
        print("SCENARIO A: Large-scale deployment")
        print("="*60)
        print(f"IoT Devices: {config.SCENARIO_A['iot_devices']}")
        print(f"Edge Servers: {config.SCENARIO_A['edge_servers']}")
        print(f"Tasks per Device: {config.SCENARIO_A['tasks_per_device']}")
        
        scenario_a_results = run_scenario("Scenario A", config.SCENARIO_A)
        
        # Run Scenario B: 300 IoT devices, 30-100 edge servers
        print("\n" + "="*60)
        print("SCENARIO B: Medium-scale deployment")
        print("="*60)
        print(f"IoT Devices: {config.SCENARIO_B['iot_devices']}")
        print(f"Edge Servers: {config.SCENARIO_B['edge_servers']}")
        print(f"Tasks per Device: {config.SCENARIO_B['tasks_per_device']}")
        
        scenario_b_results = run_scenario("Scenario B", config.SCENARIO_B)
        
        # Generate comprehensive results
        print("\n" + "="*60)
        print("GENERATING RESULTS AND VISUALIZATIONS")
        print("="*60)
        
        # Create results directory
        results_dir = config.OUTPUT_CONFIG['results_directory']
        os.makedirs(results_dir, exist_ok=True)
        
        # Combine results
        all_results = {
            'scenario_a': scenario_a_results,
            'scenario_b': scenario_b_results,
            'simulation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'simulation_time': config.SIMULATION_TIME,
                    'time_step': config.TIME_STEP,
                    'algorithms': ['mbpt', 'pso', 'genetic'],
                    'fault_tolerance': config.FAULT_TOLERANCE,
                    'reliability_target': config.TASK_CONFIG['reliability_target']
                }
            }
        }
        
        # Save raw results
        results_file = os.path.join(results_dir, 'simulation_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"Raw results saved to: {results_file}")
        
        # Generate visualizations
        visualizer = ResultsVisualizer(all_results, results_dir)
        
        # Generate all required charts and tables
        print("\nGenerating performance comparison charts...")
        visualizer.generate_performance_comparison_charts()
        
        print("Generating algorithm comparison charts...")
        visualizer.generate_algorithm_comparison_charts()
        
        print("Generating scenario comparison charts...")
        visualizer.generate_scenario_comparison_charts()
        
        print("Generating task specifications table...")
        visualizer.generate_task_specifications_table()
        
        print("Generating comprehensive report...")
        visualizer.generate_comprehensive_report()
        
        # Print summary
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Results directory: {results_dir}")
        
        # Print key findings
        print("\nKEY FINDINGS:")
        print("-" * 40)
        
        # Scenario A results
        if 'scenario_a' in all_results and 'metrics' in all_results['scenario_a']:
            mbpt_metrics = all_results['scenario_a']['metrics'].get('mbpt', {})
            if mbpt_metrics:
                print(f"Scenario A - MBPT Algorithm:")
                print(f"  Completion Rate: {mbpt_metrics.get('final_completion_rate', 0):.3f}")
                print(f"  Net Profit: {mbpt_metrics.get('final_profit', 0):.2f}")
                print(f"  Energy Consumption: {mbpt_metrics.get('final_energy_consumption', 0):.2f}")
        
        # Scenario B results
        if 'scenario_b' in all_results and 'metrics' in all_results['scenario_b']:
            mbpt_metrics = all_results['scenario_b']['metrics'].get('mbpt', {})
            if mbpt_metrics:
                print(f"Scenario B - MBPT Algorithm:")
                print(f"  Completion Rate: {mbpt_metrics.get('final_completion_rate', 0):.3f}")
                print(f"  Net Profit: {mbpt_metrics.get('final_profit', 0):.2f}")
                print(f"  Energy Consumption: {mbpt_metrics.get('final_energy_consumption', 0):.2f}")
        
        print(f"\nAll results and visualizations saved to: {results_dir}")
        
    except Exception as e:
        print(f"\nERROR: Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def run_quick_test():
    """Run a quick test to verify system functionality"""
    print("Running quick functionality test...")
    
    try:
        # Test with minimal configuration
        test_config = {
            'iot_devices': 10,
            'edge_servers': 5,
            'tasks_per_device': 20
        }
        
        print(f"Test configuration: {test_config}")
        
        # Create simulator
        simulator = EdgeComputingSimulator(test_config)
        
        # Setup scenario
        simulator.setup_scenario()
        
        print("✓ System setup completed successfully")
        print("✓ All components initialized")
        print("✓ Ready for full simulation")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if quick test is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    # Check if fast mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--fast":
        print("Running in FAST MODE with reduced complexity...")
        # Override config for faster execution
        config.SIMULATION_TIME = 100  # Even faster
        config.TIME_STEP = 5          # Larger time steps
        config.SCENARIO_A['iot_devices'] = 20      # Even smaller
        config.SCENARIO_A['edge_servers'] = 10     # Even smaller
        config.SCENARIO_A['tasks_per_device'] = 15 # Even smaller
        config.SCENARIO_B['iot_devices'] = 15      # Even smaller
        config.SCENARIO_B['edge_servers'] = 8      # Even smaller
        config.SCENARIO_B['tasks_per_device'] = 10 # Even smaller
    
    # Run main simulation
    exit_code = main()
    sys.exit(exit_code)
