#!/usr/bin/env python3
"""
Aerial RIS (UAV) Module
"""

import numpy as np

class RISModule:
    """Reconfigurable Intelligent Surface (RIS) module mounted on UAV"""
    
    def __init__(self):
        """Initialize RIS parameters"""
        self.num_elements = 64  # Default: 8x8 RIS
        self.position = np.array([0, 0, 200])  # UAV position [x, y, altitude]
        self.element_spacing = 0.5  # Half wavelength at 28 GHz
        self.max_phase_shift = 2 * np.pi
        self.min_phase_shift = 0
        self.discrete_phases = True  # Practical hardware constraint
        self.phase_resolution = 4  # Number of discrete phase levels (e.g., 4 = 90° steps)
        self.phase_matrix = None  # Phase shift matrix Θ
        
        print("RIS Module: Initialized")
    
    def initialize_ris(self):
        """Initialize RIS with default parameters"""
        print(f"RIS Module: Initializing {self.num_elements} elements")
        print(f"  UAV Position: {self.position}")
        print(f"  Discrete Phases: {self.discrete_phases}")
        print(f"  Phase Resolution: {self.phase_resolution} levels")
        
        # Initialize phase matrix
        self.phase_matrix = self.initialize_random_phases()
        
        return self.phase_matrix
    
    def set_parameters(self, num_elements=64, position=None, 
                       discrete_phases=True, phase_resolution=4):
        """Set RIS parameters"""
        self.num_elements = num_elements
        
        if position is not None:
            self.position = np.array(position)
        
        self.discrete_phases = discrete_phases
        self.phase_resolution = phase_resolution
        
        print(f"RIS Module: Parameters set - "
              f"Elements={num_elements}, Position={self.position}, "
              f"Discrete={discrete_phases}")
    
    def initialize_random_phases(self):
        """
        Initialize RIS with random phase shifts
        """
        # Generate random phases
        random_phases = np.random.uniform(0, 2*np.pi, self.num_elements)
        
        # Quantize if discrete phases are enabled
        if self.discrete_phases:
            random_phases = self.quantize_phases(random_phases)
        
        # Create diagonal phase matrix: Θ = diag(e^{jθ₁}, ..., e^{jθₙ})
        phase_values = np.exp(1j * random_phases)
        phase_matrix = np.diag(phase_values)
        
        self.phase_matrix = phase_matrix
        return phase_matrix
    
    def quantize_phases(self, phases):
        """
        Quantize phases to discrete levels
        """
        if self.phase_resolution <= 1:
            return phases  # No quantization
        
        step_size = 2 * np.pi / self.phase_resolution
        quantized_phases = np.round(phases / step_size) * step_size
        
        # Ensure within [0, 2π)
        quantized_phases = quantized_phases % (2 * np.pi)
        
        return quantized_phases
    
    def set_phase_matrix(self, phases):
        """
        Set specific phase values for RIS elements
        """
        if len(phases) != self.num_elements:
            raise ValueError(f"Expected {self.num_elements} phases, got {len(phases)}")
        
        # Quantize if needed
        if self.discrete_phases:
            phases = self.quantize_phases(phases)
        
        # Create diagonal matrix
        phase_values = np.exp(1j * phases)
        self.phase_matrix = np.diag(phase_values)
        
        return self.phase_matrix
    
    def reflect_signal(self, incident_signal, phase_matrix=None):
        """
        Reflect signal with RIS phase shifts
        """
        if phase_matrix is None:
            if self.phase_matrix is None:
                self.initialize_random_phases()
            phase_matrix = self.phase_matrix
        
        # Ensure incident_signal is numpy array
        incident_signal = np.array(incident_signal, dtype=complex)
        
        # Reflected signal: y = Θ * x
        reflected_signal = np.dot(phase_matrix, incident_signal)
        
        return reflected_signal
    
    def update_uav_position(self, new_position):
        """
        Update UAV position (for dynamic scenarios)
        """
        self.position = np.array(new_position)
        print(f"RIS Module: UAV moved to {self.position}")
    
    def get_phase_statistics(self):
        """Get statistics of current phase shifts"""
        if self.phase_matrix is None:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        # Extract phases from diagonal
        phases = np.angle(np.diag(self.phase_matrix))
        
        stats = {
            "mean": np.mean(phases),
            "std": np.std(phases),
            "min": np.min(phases),
            "max": np.max(phases),
            "discrete_levels": self.phase_resolution if self.discrete_phases else "continuous"
        }
        
        return stats
    
    def get_status(self):
        """Get RIS status"""
        status = {
            "num_elements": self.num_elements,
            "position": self.position.tolist(),
            "discrete_phases": self.discrete_phases,
            "phase_resolution": self.phase_resolution,
            "phase_matrix_shape": self.phase_matrix.shape if self.phase_matrix is not None else None
        }
        return status