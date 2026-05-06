#!/usr/bin/env python3
"""
Toy Example: Active Ising Model (AIM)
-------------------------------------
This script provides a minimal, laptop-friendly demonstration of the 
core functions used in the Active Ising Model simulations. 
It operates on a small lattice with a reduced number of sweeps.
"""

import os
import numpy as np
# Import high-level wrappers from the core module
from aim_core import run_single_ness_with_profiles, run_evolution_and_save

def main():
    # 1. Define toy thermodynamic parameters
    Lx, Ly = 20, 20       # Scaled down from 100x100 for fast local execution
    rho0 = 3.0            # Average density
    beta = 1.0            # Inverse temperature
    epsilon = 1.0         # Activity parameter
    D = 1.0               # Diffusion coefficient
    J = 1.5               # Spin alignment interaction
    h = 0.0               # External magnetic field
    seed = 42             # Fixed seed for reproducibility

    print("==================================================")
    print(" Active Ising Model - Minimum Working Example")
    print("==================================================")
    print(f"Parameters: L={Lx}x{Ly}, rho0={rho0}, beta={beta}, eps={epsilon}, J={J}, h={h}\n")


    # ---------------------------------------------------------
    # Example A: NESS Thermodynamics & Spatial Profiles
    # ---------------------------------------------------------
    print("--- Example A: Single NESS Calculation with Profiles ---")
    print("Relaxing system and computing entropy, energy, and profiles on-the-fly...")
    
    n_relax_sweeps = 1000
    n_sample_sweeps = 100
    max_particles_per_cell = 50 

    # run_single_ness_with_profiles aggregates entropy histograms, 
    # computes final energy, and extracts 1D spatial profiles (shifted to center)[cite: 1]
    results = run_single_ness_with_profiles(
        Lx, Ly, rho0, beta, epsilon, D, J, h,
        n_relax_sweeps, n_sample_sweeps, max_particles_per_cell, seed
    )
    
    # Unpack the 8 return values[cite: 1]
    total_entropy, mean_energy, EJ_list, Eh_list, mean_rho_x, mean_m_x, final_rho_x, final_m_x = results
    
    print(f"Total Entropy:  {total_entropy:.4f}")
    print(f"Mean Energy:    {mean_energy:.4f}")
    # Print just the first 5 cells to keep the console output clean
    print(f"Mean Density Profile (first 5 cells):        {np.round(mean_rho_x[:5], 2)}")
    print(f"Mean Magnetization Profile (first 5 cells):  {np.round(mean_m_x[:5], 2)}\n")


    # ---------------------------------------------------------
    # Example B: Time Evolution and Snapshots
    # ---------------------------------------------------------
    print("--- Example B: Evolution and Snapshots ---")
    print("Running system evolution and saving lattice snapshots...")
    
    n_sweeps = 2000
    # Save snapshots at the start, middle, and end of the simulation[cite: 1, 2]
    snapshot_sweeps = np.array([0, 1000, 1999], dtype=np.int32) 
    
    # Save to the current directory for the toy example
    output_dir = "./"
    
    # run_evolution_and_save keeps snapshots at specified sweeps[cite: 1]
    output_path = run_evolution_and_save(
        Lx, Ly, rho0, beta, epsilon, D, J, h,
        n_sweeps, snapshot_sweeps, output_dir, seed
    )
    
    print(f"Snapshot data successfully saved to: \n{output_path}")
    print("==================================================")

if __name__ == "__main__":
    main()