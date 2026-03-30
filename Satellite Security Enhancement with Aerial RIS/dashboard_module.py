#!/usr/bin/env python3
"""
Dashboard/UI Module
"""

import numpy as np
from datetime import datetime

class DashboardModule:
    """Dashboard for visualization and user interaction (console-based simulation)"""
    
    def __init__(self):
        """Initialize dashboard"""
        self.is_initialized = False
        self.plots_enabled = False  # Set to True for actual GUI
        self.refresh_rate = 1.0  # Hz
        
        # Data storage for visualization
        self.secrecy_rate_data = []
        self.iteration_data = []
        self.phase_data = []
        self.geometry_data = {}
        
        print("Dashboard Module: Initialized (Console Mode)")
    
    def initialize_dashboard(self):
        """Initialize dashboard components"""
        print("\n=== DASHBOARD INITIALIZATION ===")
        print("Initializing visualization components...")
        
        self.is_initialized = True
        
        print("Dashboard components:")
        print("  1. Secrecy Rate Convergence Plot")
        print("  2. RIS Phase Distribution Heatmap")
        print("  3. 3D Geometry Viewer (Satellite, UAV, Users)")
        print("  4. Real-time Metrics Panel")
        print("\nDashboard ready for simulation.")
    
    def update_iteration_results(self, iteration, secrecy_rate, beamforming_vector,
                                phase_matrix, satellite_position, ris_position,
                                user_position, eavesdropper_position):
        """
        Update dashboard with iteration results
        """
        # Store data for visualization
        self.secrecy_rate_data.append(secrecy_rate)
        self.iteration_data.append(iteration)
        
        # Extract phase information
        if phase_matrix is not None:
            if phase_matrix.ndim == 2:
                phases = np.angle(np.diag(phase_matrix))
            else:
                phases = np.angle(phase_matrix)
            self.phase_data.append(phases)
        
        # Store geometry data
        self.geometry_data = {
            "satellite": satellite_position,
            "ris": ris_position,
            "user": user_position,
            "eavesdropper": eavesdropper_position
        }
        
        # Update console display
        self.update_console_display(iteration, secrecy_rate, beamforming_vector)
    
    def update_console_display(self, iteration, secrecy_rate, beamforming_vector):
        """Update console display with current information"""
        print(f"\n--- Dashboard Update (Iteration {iteration}) ---")
        
        # Secrecy rate gauge
        self.display_gauge("Secrecy Rate", secrecy_rate, 0, 10, unit="bps/Hz")
        
        # Progress bar
        progress = min(iteration / 50, 1.0)  # Assuming 50 max iterations
        self.display_progress_bar(progress)
        
        # Phase statistics
        if self.phase_data:
            current_phases = self.phase_data[-1]
            mean_phase = np.mean(current_phases)
            print(f"RIS Phase: Mean={mean_phase:.3f} rad")
    
    def display_gauge(self, label, value, min_val, max_val, unit=""):
        """Display a simple text gauge"""
        # Normalize value for display
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        normalized = max(0, min(1, normalized))
        
        # Create gauge bar
        bar_length = 20
        filled = int(normalized * bar_length)
        bar = "[" + "=" * filled + " " * (bar_length - filled) + "]"
        
        print(f"{label:20} {bar} {value:.4f} {unit}")
    
    def display_progress_bar(self, progress):
        """Display a progress bar"""
        bar_length = 30
        filled = int(progress * bar_length)
        bar = "[" + "█" * filled + "░" * (bar_length - filled) + "]"
        percentage = progress * 100
        
        print(f"Optimization Progress: {bar} {percentage:.1f}%")
    
    def generate_final_report(self):
        """Generate final simulation report"""
        print("\n=== GENERATING FINAL REPORT ===")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not self.secrecy_rate_data:
            print("No data available for report generation.")
            return
        
        report = f"""
        SATELLITE-RIS SECURE COMMUNICATION SIMULATION REPORT
        ===================================================
        
        Report Generated: {timestamp}
        
        1. EXECUTIVE SUMMARY
        -------------------
        Simulation completed successfully with {len(self.secrecy_rate_data)} iterations.
        Final secrecy rate: {self.secrecy_rate_data[-1]:.4f} bps/Hz
        
        2. CONVERGENCE ANALYSIS
        -----------------------
        Initial rate: {self.secrecy_rate_data[0]:.4f} bps/Hz
        Final rate: {self.secrecy_rate_data[-1]:.4f} bps/Hz
        Improvement: {self.secrecy_rate_data[-1] - self.secrecy_rate_data[0]:.4f} bps/Hz
        
        3. PERFORMANCE METRICS
        ----------------------
        Average secrecy rate: {np.mean(self.secrecy_rate_data):.4f} bps/Hz
        Maximum secrecy rate: {np.max(self.secrecy_rate_data):.4f} bps/Hz
        Minimum secrecy rate: {np.min(self.secrecy_rate_data):.4f} bps/Hz
        
        ===================================================
        END OF REPORT
        """
        
        print(report)
        
        # Save report to file
        filename = f"simulation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {filename}")