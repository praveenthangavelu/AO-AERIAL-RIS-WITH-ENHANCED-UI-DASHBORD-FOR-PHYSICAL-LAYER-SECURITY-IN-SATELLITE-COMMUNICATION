#!/usr/bin/env python3
"""
Configuration File
Centralized configuration for the Satellite-RIS system
"""

import numpy as np

# ============================================================================
# SYSTEM-WIDE CONSTANTS
# ============================================================================

# Physics Constants
SPEED_OF_LIGHT = 3e8  # m/s
BOLTZMANN_CONSTANT = 1.38e-23  # J/K
ROOM_TEMPERATURE = 290  # Kelvin

# ============================================================================
# SATELLITE CONFIGURATION
# ============================================================================

SATELLITE_CONFIG = {
    "default_power": 100.0,  # Watts
    "frequency": 28e9,  # 28 GHz Ka-band
    "position": np.array([0, 0, 35786000]),  # Geostationary orbit (35,786 km)
    "antenna_elements": 8,
    "max_power": 1000.0,  # Maximum transmit power
    "beamforming_type": "MRT",  # Maximal Ratio Transmission
    "modulation": "QPSK",  # Default modulation
}

# ============================================================================
# RIS (UAV) CONFIGURATION
# ============================================================================

RIS_CONFIG = {
    "default_elements": 64,  # 8x8 RIS
    "default_position": np.array([0, 0, 200]),  # [x, y, altitude] in meters
    "element_spacing": 0.5,  # Half wavelength at 28 GHz
    "discrete_phases": True,  # Practical hardware
    "phase_resolution": 4,  # Number of discrete phase levels
    "max_phase_shift": 2 * np.pi,
    "min_phase_shift": 0,
    "uav_max_altitude": 500,  # meters
    "uav_min_altitude": 50,  # meters
}

# ============================================================================
# CHANNEL MODEL CONFIGURATION
# ============================================================================

CHANNEL_CONFIG = {
    "path_loss_exponent": 2.7,  # Urban environment
    "shadowing_variance": 8.0,  # 8 dB standard deviation
    "rician_k_factor": 10.0,  # K-factor for Rician fading
    "rayleigh_variance": 1.0,  # For Rayleigh fading
    "carrier_frequency": 28e9,  # Hz
    "bandwidth": 10e6,  # 10 MHz
    "noise_temperature": 300,  # Kelvin
    "noise_figure": 5.0,  # dB
    
    # Channel types
    "channel_types": {
        "satellite_ris": "freespace",
        "ris_user": "rician",
        "ris_eavesdropper": "rayleigh",
        "satellite_direct": "freespace"
    }
}

# ============================================================================
# OPTIMIZATION ALGORITHM CONFIGURATION
# ============================================================================

OPTIMIZATION_CONFIG = {
    "algorithm": "Alternating_Optimization",
    "convergence_threshold": 1e-4,
    "max_iterations": 100,
    "step_size": 0.1,
    "momentum_factor": 0.9,
    "use_acceleration": True,
    "regularization_factor": 1e-6,
    
    # Beamforming optimization
    "beamforming_method": "generalized_eigenvalue",
    
    # RIS phase optimization
    "phase_optimization_method": "phase_alignment",
    "eavesdropper_weight": 0.3,  # Weight for eavesdropper suppression
}

# ============================================================================
# SECRECY RATE CONFIGURATION
# ============================================================================

SECRECY_CONFIG = {
    "target_secrecy_rate": 1.0,  # bps/Hz
    "noise_power_user": 1e-12,  # Watts
    "noise_power_eavesdropper": 1e-12,  # Watts
    "bandwidth": 10e6,  # Hz
    "outage_threshold": 0.1,  # 10% outage probability
    "monte_carlo_trials": 1000,
    
    # SNR calculation
    "snr_calculation_method": "shannon",
    "minimum_snr": 1e-6,  # Minimum SNR for calculation
}

# ============================================================================
# SIMULATION CONFIGURATION
# ============================================================================

SIMULATION_CONFIG = {
    "default_scenario": 1,
    "real_time_visualization": True,
    "visualization_refresh_rate": 1.0,  # Hz
    "parameter_sweep_steps": 20,
    
    # Default positions
    "default_user_position": np.array([100, 100, 0]),
    "default_eavesdropper_position": np.array([200, 0, 0]),
    
    # Parameter ranges for sweeps
    "parameter_ranges": {
        "sat_power": {"min": 10, "max": 1000, "unit": "W"},
        "ris_elements": {"min": 16, "max": 256, "unit": "elements"},
        "uav_altitude": {"min": 50, "max": 500, "unit": "m"},
        "phase_resolution": {"min": 1, "max": 16, "unit": "levels"},
        "convergence_threshold": {"min": 1e-6, "max": 1e-2, "unit": ""},
    },
    
    # Output configuration
    "output_directory": "results/",
    "save_results": True,
    "results_format": ["json", "csv"],
    "generate_report": True,
}

# ============================================================================
# DASHBOARD/VISUALIZATION CONFIGURATION
# ============================================================================

DASHBOARD_CONFIG = {
    "enabled": True,
    "type": "console",  # Options: console, dash, streamlit, matplotlib
    "refresh_rate": 1.0,  # Hz
    
    # Plot configuration
    "plots": {
        "secrecy_rate_convergence": True,
        "ris_phase_distribution": True,
        "system_geometry": True,
        "performance_metrics": True,
        "parameter_sensitivity": True,
    },
    
    # Display limits
    "display_limits": {
        "max_iterations_display": 100,
        "max_ris_elements_display": 100,
        "phase_display_precision": 3,
    }
}

# ============================================================================
# SCENARIO DEFINITIONS
# ============================================================================

SCENARIOS = {
    1: {
        "name": "Baseline",
        "description": "Standard configuration with moderate parameters",
        "sat_power": 100,
        "ris_elements": 64,
        "uav_position": [0, 0, 200],
        "user_position": [100, 100, 0],
        "eavesdropper_position": [200, 0, 0],
        "discrete_phases": True,
        "phase_resolution": 4,
    },
    2: {
        "name": "High Security",
        "description": "Optimized for maximum secrecy rate",
        "sat_power": 500,
        "ris_elements": 128,
        "uav_position": [50, 50, 300],
        "user_position": [100, 100, 0],
        "eavesdropper_position": [200, -100, 0],
        "discrete_phases": True,
        "phase_resolution": 8,
    },
    3: {
        "name": "Low Power",
        "description": "Energy-efficient configuration",
        "sat_power": 50,
        "ris_elements": 32,
        "uav_position": [0, 0, 150],
        "user_position": [50, 50, 0],
        "eavesdropper_position": [100, 0, 0],
        "discrete_phases": True,
        "phase_resolution": 2,
    },
    4: {
        "name": "Continuous Phases",
        "description": "Ideal RIS with continuous phase shifts",
        "sat_power": 200,
        "ris_elements": 64,
        "uav_position": [0, 0, 250],
        "user_position": [100, 100, 0],
        "eavesdropper_position": [200, 0, 0],
        "discrete_phases": False,
        "phase_resolution": 0,
    },
}

# ============================================================================
# PERFORMANCE METRICS THRESHOLDS
# ============================================================================

PERFORMANCE_THRESHOLDS = {
    "excellent_secrecy_rate": 5.0,  # bps/Hz
    "good_secrecy_rate": 2.0,      # bps/Hz
    "poor_secrecy_rate": 0.5,      # bps/Hz
    
    "excellent_convergence_iterations": 20,
    "good_convergence_iterations": 50,
    "poor_convergence_iterations": 100,
    
    "acceptable_outage_probability": 0.05,  # 5%
    "poor_outage_probability": 0.20,        # 20%
}

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

VALIDATION_CONFIG = {
    "validate_channels": True,
    "validate_parameters": True,
    "check_bounds": True,
    
    "parameter_bounds": {
        "sat_power": {"min": 0.1, "max": 10000},
        "ris_elements": {"min": 1, "max": 1024},
        "uav_altitude": {"min": 0, "max": 1000},
        "phase_resolution": {"min": 0, "max": 32},
    },
    
    "warning_thresholds": {
        "low_snr": 0,      # dB
        "high_power": 500, # W
        "many_elements": 256,
    }
}

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def calculate_noise_power(temperature=ROOM_TEMPERATURE, 
                         noise_figure=5.0, 
                         bandwidth=10e6):
    """
    Calculate thermal noise power
    
    Args:
        temperature: Noise temperature in Kelvin
        noise_figure: Receiver noise figure in dB
        bandwidth: Bandwidth in Hz
        
    Returns:
        Noise power in Watts
    """
    # Convert noise figure from dB to linear
    noise_figure_linear = 10 ** (noise_figure / 10)
    
    # Calculate noise power
    noise_power = (BOLTZMANN_CONSTANT * temperature * bandwidth * 
                   noise_figure_linear)
    
    return noise_power

def calculate_wavelength(frequency):
    """
    Calculate wavelength for given frequency
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        Wavelength in meters
    """
    return SPEED_OF_LIGHT / frequency

def calculate_free_space_path_loss(distance, frequency):
    """
    Calculate free space path loss
    
    Args:
        distance: Distance in meters
        frequency: Frequency in Hz
        
    Returns:
        Path loss in linear scale (not dB)
    """
    wavelength = calculate_wavelength(frequency)
    fspl = (4 * np.pi * distance / wavelength) ** 2
    return 1 / fspl

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================

def get_full_config():
    """Get complete configuration dictionary"""
    config = {
        "satellite": SATELLITE_CONFIG,
        "ris": RIS_CONFIG,
        "channel": CHANNEL_CONFIG,
        "optimization": OPTIMIZATION_CONFIG,
        "secrecy": SECRECY_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "dashboard": DASHBOARD_CONFIG,
        "scenarios": SCENARIOS,
        "performance_thresholds": PERFORMANCE_THRESHOLDS,
        "validation": VALIDATION_CONFIG,
    }
    return config

def print_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("SATELLITE-RIS SYSTEM CONFIGURATION SUMMARY")
    print("=" * 60)
    
    config = get_full_config()
    
    for section, settings in config.items():
        if section == "scenarios":
            print(f"\n{section.upper()}:")
            for scenario_id, scenario in settings.items():
                print(f"  Scenario {scenario_id}: {scenario['name']}")
        elif section == "performance_thresholds":
            print(f"\n{section.upper()}:")
            for metric, threshold in settings.items():
                print(f"  {metric}: {threshold}")
        else:
            print(f"\n{section.upper()}:")
            for key, value in settings.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)