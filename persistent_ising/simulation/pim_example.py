#!/usr/bin/env python3
"""
Toy Example: Persistent Ising Model
-----------------------------------
A minimal, laptop-friendly demonstration of the non-equilibrium 
persistent Ising model simulation. This script avoids MPI and HPC 
sharding, focusing purely on local execution.
"""

# Import high-level functions from the core module
from pim_core import run_multi_simulation, get_mean_var

def main():
    # 1. Toy Parameters (scaled down for fast local execution)
    L = 20                # 20x20 lattice (scaled down from 50)
    time_steps = 10_000   # 10k steps (scaled down from 40.4 million)
    J = 1.0               # Coupling strength
    E0 = 0.0              # Energy offset (0.0 represents equilibrium)
    h = 0.0               # External magnetic field
    temperature = 1.0     # System temperature
    
    # Simulation control
    numSims = 4           # Number of independent simulations to run
    sampleSize = 5_000    # Keep the last 5k steps to ensure steady-state
    subSample = L * L     # Sample rate (once per sweep)
    
    print("==================================================")
    print(" Persistent Ising Model - Minimum Working Example")
    print("==================================================")
    print(f"Parameters: L={L}x{L}, J={J}, E0={E0}, T={temperature}")
    print(f"Running {numSims} parallel simulations for {time_steps} steps...")
    
    # 2. Run the simulations
    # We use Glauber dynamics as defined in the manuscript's main execution script
    results = run_multi_simulation(
        L=L, 
        steps=time_steps, 
        J=J, 
        numSims=numSims, 
        temperature=temperature, 
        E0=E0, 
        h=h, 
        algorithm='Glauber',   
        time_series=False,     # Keep false unless you want the full time series
        sampleSize=sampleSize, 
        n_jobs=-1              # Use all available local CPU cores
    )
    
    print("Simulations complete! Computing thermodynamic quantities...")
    
    # 3. Analyze the results
    # Calculate the pooled mean and variance of magnetization and interaction energy
    mean_M, mean_R, var_M, var_R, cov_MR = get_mean_var(
        results, 
        L=L, 
        subSample=subSample, 
        absolute_m=True,       # Use absolute magnetization
        raw=False              # Pool all simulations together
    )

    # 4. Display Results
    print(f"Mean Magnetization (|M|):{mean_M:.4f}")
    print(f"Variance of |M|:{var_M:.4f}")
    print(f"Mean Interaction Energy:{mean_R:.4f}")
    print(f"Variance of Energy:{var_R:.4f}")
    print(f"Covariance (M, Energy):{cov_MR:.4f}")
    print("==================================================")

if __name__ == '__main__':
    main()