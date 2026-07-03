#!/usr/bin/env python3
"""
Toy Example: Persistent Ising Model
-----------------------------------
A demonstration of the persistent Ising model simulation for local execution. It reproduces the 
exact same JSON data structure as the HPC pipeline.
"""

import json
import numpy as np
from pim_core import run_multi_ness, GLAUBER

def main():
    # 1. Toy Parameters (scaled down for fast local execution)
    L = 10                    # 10x10 lattice
    n_relax_sweeps = 1000     # Sweeps to reach NESS
    n_samples = 1000          # Sweeps for sampling at NESS
    n_sims = 4                # Number of independent simulations per (E0, J) pair
    Js = np.linspace(0.5, 1.5, 4) # Just 4 points for J to keep it fast
    E0s = np.array([-2, 0, 2])    # Just 3 points for E0
    h = 0.0
    temperature = 1.0
    bias = 1.0
    algorithm = 'Glauber'
    n_jobs = -1               # Use all available local CPU cores

    print("==================================================")
    print(" Persistent Ising Model - Minimum Working Example")
    print("==================================================")
    print(f"Lattice: {L}x{L}, Sims per point: {n_sims}")
    print(f"Relaxation Sweeps: {n_relax_sweeps}, Sample Sweeps: {n_samples}")
    print(f"Scanning {len(E0s)} E0 values and {len(Js)} J values...")
    
    # 2. Initialize 3D arrays to hold the results
    E, K = len(E0s), len(Js)
    shape_3D = (E, K, n_sims)
    
    cnfg_entp_3D       = np.full(shape_3D, np.nan, dtype=float)
    entp_prod_rate_3D  = np.full(shape_3D, np.nan, dtype=float)
    tau_obs_3D         = np.full(shape_3D, np.nan, dtype=float)
    mean_sum_s_3D      = np.full(shape_3D, np.nan, dtype=float)
    mean_sum_ss_3D     = np.full(shape_3D, np.nan, dtype=float)
    var_sum_s_3D       = np.full(shape_3D, np.nan, dtype=float)
    var_sum_ss_3D      = np.full(shape_3D, np.nan, dtype=float)
    cov_s_ss_3D        = np.full(shape_3D, np.nan, dtype=float)
    binder_cumulant_3D = np.full(shape_3D, np.nan, dtype=float)

    # 3. Run the parameter sweep locally
    for ei, E0 in enumerate(E0s):
        for kj, J in enumerate(Js):
            print(f" -> Computing E0 = {E0:>2}, J = {J:.3f} ...")
            
            results = run_multi_ness(
                n_sims=n_sims, 
                L=L, 
                n_relax_sweeps=n_relax_sweeps, 
                n_samples=n_samples, 
                J=J, 
                bias=bias, 
                temperature=temperature, 
                E0=E0, 
                h=h, 
                n_jobs=n_jobs
            )
            
            # Unpack the returned list of tuples directly into the 3D grid
            for sim_idx, res in enumerate(results):
                cnfg_entp_3D[ei, kj, sim_idx]       = res[0]
                entp_prod_rate_3D[ei, kj, sim_idx]  = res[1]
                tau_obs_3D[ei, kj, sim_idx]         = res[2]
                mean_sum_s_3D[ei, kj, sim_idx]      = res[3]
                mean_sum_ss_3D[ei, kj, sim_idx]     = res[4]
                var_sum_s_3D[ei, kj, sim_idx]       = res[5]
                var_sum_ss_3D[ei, kj, sim_idx]      = res[6]
                cov_s_ss_3D[ei, kj, sim_idx]        = res[7]
                binder_cumulant_3D[ei, kj, sim_idx] = res[8]

    # 4. Package into the exact same JSON structure
    combined_data = {
        'L': L,
        'n_relax_sweeps': n_relax_sweeps,
        'n_samples': n_samples,
        'n_sims': n_sims,
        'Js': Js.tolist(),
        'E0s': E0s.tolist(),
        'h': h,
        'temperature': temperature,
        'bias': bias,
        'algorithm': algorithm,

        # 3D arrays converted to lists for JSON serialization
        'cnfg_entp': cnfg_entp_3D.tolist(),
        'entp_prod_rate': entp_prod_rate_3D.tolist(),
        'tau_obs': tau_obs_3D.tolist(),
        'mean_sum_s': mean_sum_s_3D.tolist(),
        'mean_sum_ss': mean_sum_ss_3D.tolist(),
        'var_sum_s': var_sum_s_3D.tolist(),
        'var_sum_ss': var_sum_ss_3D.tolist(),
        'cov_s_ss': cov_s_ss_3D.tolist(),
        'binder_cumulant': binder_cumulant_3D.tolist(),
    }

    # 5. Save the output
    filename = f"./persistent_ising/data/PIM_TOY_{(n_relax_sweeps+n_samples)//1000}kswps_{n_sims}sim_L{L}.json"
    
    with open(filename, 'w') as f:
        json.dump(combined_data, f)
        
    print("==================================================")
    print(f"Simulations complete! Data successfully written to:")
    print(f" -> {filename}")
    print("==================================================")

if __name__ == '__main__':
    main()