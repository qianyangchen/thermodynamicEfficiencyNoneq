#!/usr/bin/env python3
"""
Toy Example: Active Ising Model (AIM)
-------------------------------------
This script provides a minimal demonstration of the core functions used in the Active Ising Model simulations. It generates the combined .npz file with the exact same data structure as the HPC pipeline.
"""

import time
import datetime
import numpy as np
from aim_core import run_multiple_ness, generate_simulation_seeds

def main():
    # 1. Define toy thermodynamic parameters (scaled down for fast local execution)
    Lx, Ly = 20, 20
    rho0 = 3.0
    beta = 1.0
    epsilon = 0.9 # epsilon must be strictly less than 1, otherwise will encounter division by zero error for computing entropy production rate 
    D = 1.0
    h = 0.0
    
    # Reduced parameter sweep and simulation counts
    J_list = np.linspace(0.5, 1.5, 3) # Just 3 J points
    n_relax_sweeps = 500              # Sweeps to reach NESS
    n_sample_sweeps = 100             # Sweeps for sampling at NESS
    max_particles_per_cell = 50       # Histogram bin cap
    n_sims = 2                        # Independent simulations per J
    master_seed = 12345
    n_jobs = -1                       # Use all local CPU cores

    n_J = len(J_list)

    print("==================================================")
    print(" Active Ising Model - Minimum Working Example")
    print("==================================================")
    print(f"Lattice: {Lx}x{Ly}, rho0={rho0}, Sims per J: {n_sims}")
    print(f"Relaxation Sweeps: {n_relax_sweeps}, Sample Sweeps: {n_sample_sweeps}")
    print(f"Scanning {n_J} J values...")

    # 2. Allocate final data structures (matching aim_merge.py exact structure)
    entropies = np.zeros((n_J, n_sims), dtype=np.float64)
    energies = np.zeros((n_J, n_sims), dtype=np.float64)
    E_Js = np.zeros((n_J, n_sims, n_sample_sweeps), dtype=np.float64)
    E_hs = np.zeros((n_J, n_sims, n_sample_sweeps), dtype=np.float64)
    mean_rho_x = np.zeros((n_J, n_sims, Lx), dtype=np.float64)
    mean_m_x = np.zeros((n_J, n_sims, Lx), dtype=np.float64)
    final_rho_x = np.zeros((n_J, n_sims, Lx), dtype=np.float64)
    final_m_x = np.zeros((n_J, n_sims, Lx), dtype=np.float64)
    entp_prod_rate = np.zeros((n_J, n_sims), dtype=np.float64)
    all_seeds = np.zeros((n_J, n_sims), dtype=np.int64)

    start_time = time.time()

    # 3. Loop over the J values and run parallel simulations locally
    for j_idx, J in enumerate(J_list):
        J = float(J)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Computing J = {J:.3f} ({j_idx + 1}/{n_J}) ...")

        params = {
            'Lx': Lx, 'Ly': Ly, 'rho0': rho0, 'beta': beta,
            'epsilon': epsilon, 'D': D, 'J': J, 'h': h
        }

        # Generate seeds for this J row
        row_master_seed = master_seed + j_idx
        row_seeds = generate_simulation_seeds(n_sims, master_seed=row_master_seed)
        all_seeds[j_idx, :] = row_seeds

        # Run parallel simulations for this specific J value
        results = run_multiple_ness(
            params,
            n_relax_sweeps,
            n_sample_sweeps,
            max_particles_per_cell,
            n_sims=n_sims,
            n_jobs=n_jobs,
            master_seed=row_master_seed
        )

        # Unpack the 9 outputs per simulation into the pre-allocated arrays
        for sim_idx, res in enumerate(results):
            entropy, energy, EJ_list, Eh_list, m_rho, m_m, f_rho, f_m, epr = res
            
            entropies[j_idx, sim_idx] = entropy
            energies[j_idx, sim_idx] = energy
            E_Js[j_idx, sim_idx, :] = np.asarray(EJ_list, dtype=np.float64)
            E_hs[j_idx, sim_idx, :] = np.asarray(Eh_list, dtype=np.float64)
            mean_rho_x[j_idx, sim_idx, :] = m_rho
            mean_m_x[j_idx, sim_idx, :] = m_m
            final_rho_x[j_idx, sim_idx, :] = f_rho
            final_m_x[j_idx, sim_idx, :] = f_m
            entp_prod_rate[j_idx, sim_idx] = epr

    # 4. Save compiled output mimicking the output of aim_merge.py
    compiled_file = f"./active_ising/data/aim_ness_TOY_L{Lx}x{Ly}_r{rho0}_b{beta}_e{epsilon}_D{D}.npz"
    
    np.savez_compressed(
        compiled_file,
        Lx=Lx, Ly=Ly,
        rho0=rho0, beta=beta, epsilon=epsilon, D=D, h=h,
        J_list=np.array(J_list, dtype=np.float64),
        n_relax_sweeps=n_relax_sweeps,
        n_sample_sweeps=n_sample_sweeps,
        max_particles_per_cell=max_particles_per_cell,
        n_sims=n_sims,
        entropies=entropies,
        energies=energies,
        E_Js=E_Js,
        E_hs=E_hs,
        all_seeds=all_seeds,
        master_seed=master_seed,
        mean_rho_x=mean_rho_x,
        mean_m_x=mean_m_x,
        final_rho_x=final_rho_x,
        final_m_x=final_m_x,
        entp_prod_rate=entp_prod_rate
    )

    duration = (time.time() - start_time) / 60.0
    
    print("==================================================")
    print(f"Simulations complete! ({duration:.2f} minutes)")
    print(f"Data successfully written to:")
    print(f" -> {compiled_file}")
    print("==================================================")

if __name__ == '__main__':
    main()