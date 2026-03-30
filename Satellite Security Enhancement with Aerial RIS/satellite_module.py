#!/usr/bin/env python3
"""
Satellite Communication Module
"""

import numpy as np

class SatelliteModule:
    """Satellite communication system module"""
    
    def __init__(self):
        """Initialize satellite parameters"""
        self.transmit_power = 100.0  # Watts
        self.frequency = 28e9  # 28 GHz Ka-band
        self.position = np.array([0, 0, 35786000])  # Geostationary orbit
        self.antenna_elements = 8
        self.max_power_constraint = 1000.0  # Maximum transmit power
        self.beamforming_vector = None
        
        print("Satellite Module: Initialized")
    
    def initialize_transmitter(self):
        """Initialize satellite transmitter parameters"""
        print(f"Satellite Module: Transmitter initialized at {self.position/1000} km")
        print(f"  Frequency: {self.frequency/1e9} GHz")
        print(f"  Max Power: {self.max_power_constraint} W")
    
    def set_parameters(self, transmit_power=100.0, frequency=28e9, 
                       position=None, antenna_elements=8):
        """Set satellite parameters"""
        self.transmit_power = transmit_power
        self.frequency = frequency
        
        if position is not None:
            self.position = np.array(position)
        
        self.antenna_elements = antenna_elements
        
        print(f"Satellite Module: Parameters set - "
              f"Power={transmit_power}W, Frequency={frequency/1e9}GHz")
    
    def generate_beamforming_vector(self, channel_state):
        """
        Generate beamforming vector using Maximal Ratio Transmission (MRT)
        """
        # Ensure channel_state is numpy array
        channel_state = np.array(channel_state, dtype=complex)
        
        # MRT strategy: align with channel
        if np.linalg.norm(channel_state) > 0:
            w = channel_state / np.linalg.norm(channel_state)
        else:
            # Default to uniform if channel is zero
            w = np.ones_like(channel_state) / np.sqrt(len(channel_state))
        
        # Apply power constraint
        w = np.sqrt(self.transmit_power) * w
        
        # Ensure power constraint is satisfied
        power = np.linalg.norm(w) ** 2
        if power > self.max_power_constraint:
            w = w * np.sqrt(self.max_power_constraint / power)
        
        self.beamforming_vector = w
        return w
    
    def transmit_signal(self, data_symbol, beamforming_vector=None):
        """
        Generate transmitted signal
        """
        if beamforming_vector is None:
            if self.beamforming_vector is None:
                # Create default beamforming vector
                self.beamforming_vector = np.ones(self.antenna_elements, dtype=complex) / np.sqrt(self.antenna_elements)
            beamforming_vector = self.beamforming_vector
        
        # Transmitted signal: x = w * s
        transmitted_signal = beamforming_vector * data_symbol
        
        return transmitted_signal
    
    def calculate_beamforming_gain(self, direction):
        """
        Calculate beamforming gain in specific direction
        """
        if self.beamforming_vector is None:
            return 0
        
        # Simple beamforming gain calculation
        gain = np.abs(np.dot(self.beamforming_vector.conj().T, direction)) ** 2
        
        return gain
    
    def get_status(self):
        """Get satellite status"""
        status = {
            "transmit_power": self.transmit_power,
            "frequency": self.frequency,
            "position": self.position.tolist(),
            "antenna_elements": self.antenna_elements,
            "beamforming_vector_norm": np.linalg.norm(self.beamforming_vector) if self.beamforming_vector is not None else 0
        }
        return status