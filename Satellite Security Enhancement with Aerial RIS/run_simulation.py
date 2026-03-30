#!/usr/bin/env python3
"""
Simplified Simulation Runner
Run this file to start the simulation
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import SecureSatCommSystem

def main():
    """Main simulation runner"""
    print("\n" + "="*60)
    print("SATELLITE-RIS SECURE COMMUNICATION SIMULATOR")
    print("="*60)
    
    # Create and run system
    system = SecureSatCommSystem()
    
    try:
        # Run main system
        system.run_system()
        
        # Optional: Run parameter sweep
        run_sweep = input("\nRun parameter sweep? (y/n): ").lower().strip()
        if run_sweep == 'y':
            parameter = input("Parameter to sweep (sat_power/ris_elements/uav_altitude): ").strip()
            min_val = float(input("Minimum value: "))
            max_val = float(input("Maximum value: "))
            steps = int(input("Number of steps: "))
            
            results = system.run_parameter_sweep(parameter, min_val, max_val, steps)
            print(f"\nParameter sweep completed. Results saved.")
        
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Simulation Complete")
    print("="*60)

if __name__ == "__main__":
    main()