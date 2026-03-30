#!/usr/bin/env python3
"""
Simulation Control Module
"""

import numpy as np
import json
import csv
from datetime import datetime

class SimulationModule:
    """Simulation control and parameter management"""
    
    def __init__(self):
        """Initialize simulation parameters"""
        self.scenarios = {}
        self.current_scenario = 1
        self.simulation_time = 0
        self.real_time_visualization = True
        self.monte_carlo_trials = 1000
        
        # System parameters database
        self.system_parameters = {
            # Satellite Parameters
            "sat_power_min": 10,
            "sat_power_max": 1000,
            "sat_power_step": 50,
            "sat_frequency": 28e9,
            
            # RIS Parameters
            "ris_elements_min": 16,
            "ris_elements_max": 256,
            "uav_altitudes": [100, 200, 300, 400, 500],
            "discrete_phases": True,
            "phase_resolution": 4,
            
            # Channel Parameters
            "path_loss_exponent": 2.7,
            "shadowing_std": 8.0,
            "rician_k_factor": 10.0,
            "noise_power": 1e-12,
            
            # User Positions
            "user_position": [100, 100, 0],
            "eavesdropper_position": [200, 0, 0],
            
            # AO Parameters
            "convergence_threshold": 1e-4,
            "max_iterations": 50,
            
            # Simulation Control
            "parameter_changed": False
        }
        
        # Results storage
        self.results = {}
        self.parameter_sweep_results = {}
        
        print("Simulation Module: Initialized")
    
    def initialize_system_parameters(self):
        """Initialize system parameters with default values"""
        print("Simulation Module: Initializing system parameters")
        
        # Load default scenarios
        self.load_default_scenarios()
        
        # Print summary
        print(f"  Satellite power range: {self.get_parameter('sat_power_min')} - "
              f"{self.get_parameter('sat_power_max')} W")
        print(f"  RIS elements range: {self.get_parameter('ris_elements_min')} - "
              f"{self.get_parameter('ris_elements_max')}")
        print(f"  Convergence threshold: {self.get_parameter('convergence_threshold')}")
        print(f"  Max iterations: {self.get_parameter('max_iterations')}")
    
    def load_default_scenarios(self):
        """Load default simulation scenarios"""
        print("Simulation Module: Loading default scenarios")
        
        # Scenario 1: Baseline
        self.scenarios[1] = {
            "name": "Baseline Configuration",
            "description": "Standard configuration with moderate parameters",
            "sat_power": 100,
            "ris_elements": 64,
            "uav_position": [0, 0, 200],
            "user_position": [100, 100, 0],
            "eavesdropper_position": [200, 0, 0],
            "discrete_phases": True,
            "phase_resolution": 4
        }
        
        print(f"  Loaded {len(self.scenarios)} scenarios")
    
    def load_scenario(self, scenario_id):
        """
        Load specific simulation scenario
        """
        if scenario_id not in self.scenarios:
            print(f"Simulation Module: Scenario {scenario_id} not found")
            return False
        
        scenario = self.scenarios[scenario_id]
        self.current_scenario = scenario_id
        
        # Update system parameters
        self.system_parameters["sat_power"] = scenario["sat_power"]
        self.system_parameters["ris_elements"] = scenario["ris_elements"]
        self.system_parameters["uav_position"] = scenario["uav_position"]
        self.system_parameters["user_position"] = scenario["user_position"]
        self.system_parameters["eavesdropper_position"] = scenario["eavesdropper_position"]
        self.system_parameters["discrete_phases"] = scenario["discrete_phases"]
        self.system_parameters["phase_resolution"] = scenario["phase_resolution"]
        
        print(f"Simulation Module: Scenario {scenario_id} loaded: {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        
        return True
    
    def get_parameter(self, parameter_name, default_value=None):
        """
        Get simulation parameter
        """
        return self.system_parameters.get(parameter_name, default_value)
    
    def update_parameter(self, parameter_name, value):
        """
        Update simulation parameter
        """
        self.system_parameters[parameter_name] = value
        self.system_parameters["parameter_changed"] = True
        
        print(f"Simulation Module: Parameter updated - {parameter_name} = {value}")
    
    def export_results(self, secrecy_rate_history=None, beamforming_vector=None,
                      ris_phase_matrix=None, system_config=None):
        """
        Export simulation results to files
        """
        print("\nSimulation Module: Exporting results")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        export_data = {
            "timestamp": timestamp,
            "scenario": self.current_scenario,
            "system_parameters": self.system_parameters.copy(),
            "results": self.results.copy()
        }
        
        if secrecy_rate_history is not None:
            export_data["secrecy_rate_history"] = list(secrecy_rate_history)
        
        # Export to JSON
        json_filename = f"simulation_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Export to CSV (rate history)
        if secrecy_rate_history:
            csv_filename = f"secrecy_rates_{timestamp}.csv"
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Secrecy_Rate_bps_Hz'])
                for i, rate in enumerate(secrecy_rate_history):
                    writer.writerow([i, rate])
        
        print(f"  Results exported to:")
        print(f"    {json_filename}")
        if secrecy_rate_history:
            print(f"    {csv_filename}")