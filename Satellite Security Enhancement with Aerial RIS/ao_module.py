#!/usr/bin/env python3
"""
Alternating Optimization (AO) Algorithm Module
"""

import numpy as np

class AOModule:
    """Alternating Optimization algorithm for joint beamforming and RIS optimization"""
    
    def __init__(self):
        """Initialize AO parameters"""
        self.convergence_threshold = 1e-4
        self.max_iterations = 100
        self.step_size = 0.1
        self.momentum_factor = 0.9
        self.use_acceleration = True
        
        # For momentum-based optimization
        self.previous_beamforming_update = None
        self.previous_phase_update = None
        
        print("AO Module: Initialized")
    
    def initialize_ao(self):
        """Initialize AO algorithm with parameters"""
        print(f"AO Module: Convergence threshold = {self.convergence_threshold}")
        print(f"AO Module: Max iterations = {self.max_iterations}")
        print(f"AO Module: Step size = {self.step_size}")
        
        # Reset momentum variables
        self.previous_beamforming_update = None
        self.previous_phase_update = None
    
    def optimize_beamforming(self, phase_matrix, channels):
        """
        Optimize satellite beamforming vector (fixed RIS phases)
        """
        # Extract channel information
        sat_ris_channel = channels.get("sat_ris", {}).get("coefficient", 1)
        ris_user_channel = channels.get("ris_user", {}).get("coefficient", 1)
        ris_eaves_channel = channels.get("ris_eavesdropper", {}).get("coefficient", 1)
        
        # Calculate effective channels through RIS
        h_user = sat_ris_channel * ris_user_channel
        h_eaves = sat_ris_channel * ris_eaves_channel
        
        # Apply RIS phase shifts
        if phase_matrix is not None:
            # Assuming phase_matrix is diagonal
            if hasattr(phase_matrix, 'shape') and len(phase_matrix.shape) == 2:
                phase_factor = np.trace(phase_matrix) / phase_matrix.shape[0]
            else:
                phase_factor = phase_matrix if np.isscalar(phase_matrix) else 1.0
            h_user *= phase_factor
            h_eaves *= phase_factor
        
        # Convert to appropriate dimensions
        h_user = np.atleast_1d(h_user)
        h_eaves = np.atleast_1d(h_eaves)
        
        # For multi-antenna case
        if len(h_user) > 1 or len(h_eaves) > 1:
            # Construct channel matrices
            H_user = np.outer(h_user.conj(), h_user)
            H_eaves = np.outer(h_eaves.conj(), h_eaves)
            
            # Add regularization for numerical stability
            H_eaves_reg = H_eaves + 1e-6 * np.eye(len(h_user))
            
            try:
                # Solve generalized eigenvalue problem
                from scipy.linalg import eigh
                eigenvalues, eigenvectors = eigh(H_user, H_eaves_reg)
                
                # Select eigenvector with largest eigenvalue
                max_idx = np.argmax(eigenvalues)
                w_opt = eigenvectors[:, max_idx]
                
            except ImportError:
                # Fallback if scipy not available
                w_opt = h_user / np.linalg.norm(h_user)
            except np.linalg.LinAlgError:
                # Fallback: MRT toward legitimate user
                w_opt = h_user / np.linalg.norm(h_user)
        else:
            # Single-antenna case: simple power allocation
            w_opt = np.array([1.0], dtype=complex)
        
        # Normalize
        w_norm = np.linalg.norm(w_opt)
        if w_norm > 0:
            w_opt = w_opt / w_norm
        
        # Apply momentum if enabled
        if self.use_acceleration and self.previous_beamforming_update is not None:
            w_opt = w_opt + self.momentum_factor * self.previous_beamforming_update
        
        # Store for next iteration
        self.previous_beamforming_update = w_opt.copy()
        
        return w_opt
    
    def optimize_ris_phases(self, beamforming_vector, channels):
        """
        Optimize RIS phase shifts (fixed beamforming)
        """
        # Extract channel information
        sat_ris_channel = channels.get("sat_ris", {}).get("coefficient", 1)
        ris_user_channel = channels.get("ris_user", {}).get("coefficient", 1)
        ris_eaves_channel = channels.get("ris_eavesdropper", {}).get("coefficient", 1)
        
        # Convert beamforming vector to scalar if needed
        if hasattr(beamforming_vector, '__len__') and len(beamforming_vector) > 0:
            bf_scalar = beamforming_vector[0] if len(beamforming_vector) > 0 else 1.0
        else:
            bf_scalar = beamforming_vector
        
        # Calculate combined channels
        combined_user = sat_ris_channel * ris_user_channel * bf_scalar
        combined_eaves = sat_ris_channel * ris_eaves_channel * bf_scalar
        
        # Calculate optimal phase
        # Phase alignment for legitimate user, while reducing for eavesdropper
        desired_phase = -np.angle(combined_user)
        undesired_phase = np.angle(combined_eaves)
        
        # Weighted optimization
        user_power = np.abs(combined_user) ** 2
        eaves_power = np.abs(combined_eaves) ** 2
        
        if user_power + eaves_power > 0:
            weight = eaves_power / (user_power + eaves_power)
        else:
            weight = 0.5
        
        optimal_phase = desired_phase - weight * undesired_phase
        
        # Normalize to [0, 2π)
        optimal_phase = optimal_phase % (2 * np.pi)
        
        # Create a simple phase matrix (1x1 for now)
        phase_value = np.exp(1j * optimal_phase)
        phase_matrix = np.array([[phase_value]], dtype=complex)
        
        # Apply momentum if enabled
        if self.use_acceleration and self.previous_phase_update is not None:
            # Average with previous update
            phase_matrix = (1 - self.momentum_factor) * phase_matrix + \
                          self.momentum_factor * self.previous_phase_update
        
        # Store for next iteration
        self.previous_phase_update = phase_matrix.copy()
        
        return phase_matrix
    
    def check_convergence(self, current_rate, previous_rate, iteration):
        """
        Check if optimization has converged
        """
        # Ensure rates are scalars
        if hasattr(current_rate, '__len__'):
            current_rate = current_rate[0] if len(current_rate) > 0 else current_rate
        if hasattr(previous_rate, '__len__'):
            previous_rate = previous_rate[0] if len(previous_rate) > 0 else previous_rate
        
        # Convert to float if needed
        current_rate = float(current_rate) if not np.isscalar(current_rate) else current_rate
        previous_rate = float(previous_rate) if not np.isscalar(previous_rate) else previous_rate
        
        # Calculate rate difference
        if previous_rate == 0:
            rate_diff = abs(current_rate)
        else:
            rate_diff = abs(current_rate - previous_rate) / abs(previous_rate)
        
        # Ensure rate_diff is a scalar
        rate_diff = float(rate_diff) if not np.isscalar(rate_diff) else rate_diff
        
        # Check absolute threshold
        if rate_diff < self.convergence_threshold:
            print(f"AO Module: Convergence achieved at iteration {iteration}")
            print(f"  Rate difference: {rate_diff:.6f} < {self.convergence_threshold}")
            return True
        
        # Check maximum iterations
        if iteration >= self.max_iterations:
            print(f"AO Module: Maximum iterations reached ({iteration})")
            return True
        
        return False