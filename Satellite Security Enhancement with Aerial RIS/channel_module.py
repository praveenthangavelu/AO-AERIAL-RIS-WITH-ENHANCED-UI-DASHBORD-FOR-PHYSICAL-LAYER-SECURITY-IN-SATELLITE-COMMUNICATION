#!/usr/bin/env python3
"""
Channel Modeling Module
"""

import numpy as np

class ChannelModule:
    """Channel modeling for satellite-RIS-ground communication"""
    
    def __init__(self):
        """Initialize channel parameters"""
        self.path_loss_exponent = 2.7  # Urban environment
        self.shadowing_variance = 8.0  # 8 dB standard deviation
        self.fading_parameter = 10.0  # K-factor for Rician fading
        self.carrier_frequency = 28e9  # 28 GHz
        self.wavelength = 3e8 / 28e9  # Wavelength
        self.noise_power = 1e-12  # Thermal noise power (W)
        
        # Speed of light
        self.c = 3e8
        
        print("Channel Module: Initialized")
    
    def establish_all_channels(self, satellite_position, ris_position, 
                               user_position, eavesdropper_position):
        """
        Establish all communication channels
        """
        print("Channel Module: Establishing communication channels...")
        
        channels = {}
        
        # 1. Satellite-to-RIS channel
        print("  Calculating Satellite-to-RIS channel...")
        sat_to_ris = self.calculate_channel(
            tx_position=satellite_position,
            rx_position=ris_position,
            channel_type="satellite_ris",
            has_los=True
        )
        channels["sat_ris"] = sat_to_ris
        
        # 2. RIS-to-User channel
        print("  Calculating RIS-to-User channel...")
        ris_to_user = self.calculate_channel(
            tx_position=ris_position,
            rx_position=user_position,
            channel_type="terrestrial",
            has_los=True
        )
        channels["ris_user"] = ris_to_user
        
        # 3. RIS-to-Eavesdropper channel
        print("  Calculating RIS-to-Eavesdropper channel...")
        ris_to_eavesdropper = self.calculate_channel(
            tx_position=ris_position,
            rx_position=eavesdropper_position,
            channel_type="terrestrial",
            has_los=False  # Assuming NLoS for eavesdropper
        )
        channels["ris_eavesdropper"] = ris_to_eavesdropper
        
        print("Channel Module: All channels established")
        
        return channels
    
    def calculate_channel(self, tx_position, rx_position, 
                          channel_type="terrestrial", has_los=True):
        """
        Calculate channel coefficient between two points
        """
        # Convert to numpy arrays
        tx_pos = np.array(tx_position)
        rx_pos = np.array(rx_position)
        
        # Calculate distance
        distance = np.linalg.norm(tx_pos - rx_pos)
        
        # Path loss calculation based on channel type
        if "satellite" in channel_type:
            # Free Space Path Loss (FSPL) for satellite links
            fspl = (4 * np.pi * distance / self.wavelength) ** 2
            path_loss_linear = 1 / fspl
        else:
            # Terrestrial path loss with path loss exponent
            path_loss_linear = (self.wavelength / (4 * np.pi * distance)) ** 2
        
        # Apply additional path loss exponent for terrestrial
        if channel_type == "terrestrial":
            path_loss_linear *= distance ** (-self.path_loss_exponent)
        
        # Fading model
        if has_los:
            # Rician fading for LoS channels
            fading = self.generate_rician_fading()
        else:
            # Rayleigh fading for NLoS channels
            fading = self.generate_rayleigh_fading()
        
        # Shadowing (log-normal)
        shadowing = 10 ** (np.random.normal(0, self.shadowing_variance) / 10)
        
        # Combined channel coefficient
        channel_coefficient = np.sqrt(path_loss_linear * shadowing) * fading
        
        # Add random phase for multipath
        random_phase = np.exp(1j * np.random.uniform(0, 2*np.pi))
        channel_coefficient *= random_phase
        
        # Calculate SNR for reference
        snr = np.abs(channel_coefficient) ** 2 / self.noise_power
        
        # Return channel information
        channel_info = {
            "coefficient": channel_coefficient,
            "distance": distance,
            "path_loss": path_loss_linear,
            "snr_db": 10 * np.log10(snr) if snr > 0 else -np.inf,
            "fading_type": "rician" if has_los else "rayleigh",
            "has_los": has_los
        }
        
        return channel_info
    
    def generate_rician_fading(self):
        """
        Generate Rician fading coefficient
        """
        # K-factor
        K = self.fading_parameter
        
        # LOS component
        los_magnitude = np.sqrt(K / (K + 1))
        los_phase = np.random.uniform(0, 2*np.pi)
        los_component = los_magnitude * np.exp(1j * los_phase)
        
        # NLOS component (Rayleigh)
        nlos_real = np.random.normal(0, np.sqrt(1/(2*(K+1))))
        nlos_imag = np.random.normal(0, np.sqrt(1/(2*(K+1))))
        nlos_component = nlos_real + 1j * nlos_imag
        
        # Combined fading
        fading = los_component + nlos_component
        
        return fading
    
    def generate_rayleigh_fading(self):
        """
        Generate Rayleigh fading coefficient
        """
        # Real and imaginary parts are independent Gaussian
        real_part = np.random.normal(0, 1/np.sqrt(2))
        imag_part = np.random.normal(0, 1/np.sqrt(2))
        
        fading = real_part + 1j * imag_part
        
        return fading