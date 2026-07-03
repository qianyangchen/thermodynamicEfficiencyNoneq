# THIS IS THE NEW ONE, ONLY MAKE EDITS HERE

#!/usr/bin/env python
import json
import numpy as np
from numba import njit
import os
from joblib import Parallel, delayed
import time
import datetime
from numpy.linalg import eigvals
# core
#--------------------------------------------------------------------------
# Low-level (Numba-accelerated) functions
#--------------------------------------------------------------------------
@njit(cache=True)
def seed_numba(seed):
    """Explicitly seed Numba's internal PRNG."""
    np.random.seed(seed)

@njit(cache=True)
def initialize_system(Lx, Ly, rho0, beta, J, h, D):

    N = int(rho0 * Lx * Ly)

    # Create N spins and allocate them randomly on the lattice
    spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=N)
    pos_x = np.random.randint(0, Lx, size=N).astype(np.int32)
    pos_y = np.random.randint(0, Ly, size=N).astype(np.int32)

    n_plus = np.zeros((Lx, Ly), dtype=np.int32)
    n_minus = np.zeros((Lx, Ly), dtype=np.int32)

    for i in range(N):
        if spins[i] == 1:
            n_plus[pos_x[i], pos_y[i]] += 1
        else:
            n_minus[pos_x[i], pos_y[i]] += 1

    # modified dt to account for non-zero J & h
    dt = 1.0 / (4.0 * D + np.exp(beta * (abs(J) + abs(h))))

    return pos_x, pos_y, spins, n_plus, n_minus, N, dt

@njit(cache=True)
def step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, 
               flip_table, p_R_const, p_L_const, p_UD_const, max_rho):
    Lx, Ly = n_plus.shape

    idx = np.random.randint(N)
    s = spins[idx]
    x = pos_x[idx]
    y = pos_y[idx]
    
    rho = n_plus[x, y] + n_minus[x, y]
    m = n_plus[x, y] - n_minus[x, y]
    
    s_idx = 0 if s == -1 else 1 # for lookup table indexing

    # 1. EXPONENTIAL LOOKUP
    if rho <= max_rho:
        p_flip = flip_table[s_idx, rho, m + max_rho]
    else:
        # Fallback just in case density spikes above cap
        p_flip = np.exp(-beta * s * (J * m / rho + h)) * dt

    # 2. PRECOMPUTED CONSTANTS
    p_R = p_flip + p_R_const[s_idx]
    p_L = p_R + p_L_const[s_idx]
    p_U = p_L + p_UD_const
    p_D = p_U + p_UD_const

    r = np.random.rand()

    if r < p_flip:
        spins[idx] = -s
        if s == 1:
            n_plus[x, y] -= 1
            n_minus[x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_plus[x, y] += 1

    elif r < p_R:
        new_x = x + 1
        if new_x == Lx:
            new_x = 0 #periodic boundary
            
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1

    elif r < p_L:
        new_x = x - 1
        if new_x < 0:
            new_x = Lx - 1 #periodic boundary
            
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1

    elif r < p_U:
        new_y = y + 1
        if new_y == Ly:
            new_y = 0 #periodic boundary
            
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1

    elif r < p_D:
        new_y = y - 1
        if new_y < 0:
            new_y = Ly - 1 #periodic boundary
            
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1

@njit(cache=True)
def step_numba_epr(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, flip_table, p_R_const, p_L_const, p_UD_const, max_rho):
    Lx, Ly = n_plus.shape

    # 1. choose one particle
    idx = np.random.randint(N)
    s = spins[idx]
    x = pos_x[idx]
    y = pos_y[idx]
    
    rho = n_plus[x, y] + n_minus[x, y]
    m = n_plus[x, y] - n_minus[x, y]
    
    s_idx = 0 if s == -1 else 1 # for lookup table indexing

    # 2. check events (precomputed probabilities)
    if rho <= max_rho:
        p_flip = flip_table[s_idx, rho, m + max_rho]
    else:
        # Fallback just in case density spikes above cap
        p_flip = np.exp(-beta * s * (J * m / rho + h)) * dt

    p_R = p_flip + p_R_const[s_idx]
    p_L = p_R + p_L_const[s_idx]
    p_U = p_L + p_UD_const
    p_D = p_U + p_UD_const
    log_ratio = 0.0  # Default log ratio for no-op or up/down moves (no time-reversal asymmetry there)

    r = np.random.rand()

    if r < p_flip: # flip
        spins[idx] = -s
        if s == 1:
            n_plus[x, y] -= 1
            n_minus[x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_plus[x, y] += 1
        m_after = m - 2 * s
        if rho <= max_rho:
            p_flip_rev = flip_table[s_idx ^ 1, rho, m_after + max_rho] # flip spin index for lookup
        else:
            # Fallback: use +beta instead because the spin is flipped (-s)
            p_flip_rev = np.exp(beta * s * (J * m_after / rho + h)) * dt
        log_ratio = np.log(p_flip / p_flip_rev)

    elif r < p_R: # move right
        new_x = x + 1
        if new_x == Lx:
            new_x = 0 #periodic boundary
            
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1
        log_ratio = np.log((1.0 + s * epsilon) / (1.0 - s * epsilon))
    
    elif r < p_L: # move left
        new_x = x - 1
        if new_x < 0:
            new_x = Lx - 1 #periodic boundary
            
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1
        log_ratio = np.log((1.0 - s * epsilon) / (1.0 + s * epsilon))
    
    elif r < p_U: # move up
        new_y = y + 1
        if new_y == Ly:
            new_y = 0 #periodic boundary
            
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1
    
    elif r < p_D: # move down
        new_y = y - 1
        if new_y < 0:
            new_y = Ly - 1 #periodic boundary
            
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1
    return log_ratio

@njit(cache=True)
def run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_sweeps, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell):
    """
    Each sweep = N updates. N is the total number of particles. For relaxing the system to NESS.
    """
    for _ in range(n_sweeps):
        for _ in range(N):
            step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell)

@njit(cache=True)
def run_sweeps_numba_epr(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_sweeps, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell):
    """
    Each sweep = N updates. N is the total number of particles.
    """
    # record log ratio for all attempt updates in the last sweep only
    log_ratios = np.zeros(N)
    for _ in range(n_sweeps):
        for i in range(N):
            log_ratios[i] = step_numba_epr(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell)
    return log_ratios

@njit(cache=True)
def run_sweeps_with_snapshots(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_sweeps, snapshot_sweeps, snapshots_plus, snapshots_minus, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell):
    """
    Runs the simulation for a total of n_sweeps, and saves snapshots of n_plus and n_minus at specified sweeps.
    snapshot_sweeps: a 1D Numpy array of sweeps at which to save snapshots (e.g., np.array([0, 100, 500, 1000]))
    """
    n_snaps = len(snapshot_sweeps)
    snap_idx = 0

    # 1. Capture the initial state (sweep 0) BEFORE any simulation runs
    if snap_idx < n_snaps and snapshot_sweeps[snap_idx] == 0:
        snapshots_plus[snap_idx] = n_plus.copy()
        snapshots_minus[snap_idx] = n_minus.copy()
        snap_idx += 1

    # 2. Run sweeps from 1 to n_sweeps (inclusive) to reach the 5000 target
    for sweep in range(1, n_sweeps + 1):
        # run one sweep (N updates)
        for _ in range(N):
            step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell)

        # 3. Check if current sweep is a snapshot target
        if snap_idx < n_snaps and sweep == snapshot_sweeps[snap_idx]:
            snapshots_plus[snap_idx] = n_plus.copy()
            snapshots_minus[snap_idx] = n_minus.copy()
            snap_idx += 1

@njit(cache=True)
def compute_energy_components(n_plus, n_minus):
    """
    Computes the separate components of the Hamiltonian to allow O(1) evaluation
    for multiple J and h values.
    Returns: E_J, E_h
    where H = -J * E_J - h * E_h
    """
    Lx, Ly = n_plus.shape
    E_J = 0.0
    E_h = 0.0
    for x in range(Lx):
        for y in range(Ly):
            rho = n_plus[x, y] + n_minus[x, y]
            if rho > 0:
                m = n_plus[x, y] - n_minus[x, y]
                E_J += ((m * m) / (2.0 * rho) - 0.5)
                E_h += m
    return E_J, E_h

@njit(cache=True)
def compute_total_energy(n_plus, n_minus, J, h):
    """
    Computes the total conservative Hamiltonian of the system.
    H = -J * sum(m^2 / 2*rho - 1/2) - h * sum(m)
    """
    E_J, E_h = compute_energy_components(n_plus, n_minus)
    energy = -J * E_J - h * E_h
    return energy

@njit(cache=True)
def accumulate_ness_histogram(n_plus, n_minus, hist):
    """
    Shifts the system to the comoving frame (centered at Lx/2) using the
    circular center of mass, and tallies the state into the 3D histogram array.
    """
    Lx, Ly = n_plus.shape
    
    # 1. Calculate unshifted 1D density
    raw_rho_x = np.zeros(Lx, dtype=np.float64)
    for x in range(Lx):
        for y in range(Ly):
            raw_rho_x[x] += n_plus[x, y] + n_minus[x, y]
            
    # 2. Robust Circular Center of Mass
    sum_cos = 0.0
    sum_sin = 0.0
    for x in range(Lx):
        theta = 2.0 * np.pi * x / Lx
        sum_cos += raw_rho_x[x] * np.cos(theta)
        sum_sin += raw_rho_x[x] * np.sin(theta)

    avg_theta = np.arctan2(sum_sin, sum_cos)
    if avg_theta < 0:
        avg_theta += 2.0 * np.pi

    center_x = int(np.round(avg_theta * Lx / (2.0 * np.pi))) % Lx
    shift = (Lx // 2) - center_x
    
    # 3. Accumulate counts in the shifted coordinate system
    max_n = hist.shape[1] - 1  # Cap to prevent array out-of-bounds
    
    for x in range(Lx):
        shifted_x = (x + shift) % Lx
        for y in range(Ly):
            np_val = min(n_plus[x, y], max_n)
            nm_val = min(n_minus[x, y], max_n)
            hist[shifted_x, np_val, nm_val] += 1


#--------------------------------------------------------------------------
# High-level sweeping functions (for evolution, keep snapshots)
#--------------------------------------------------------------------------
def ensure_list(x):
    if np.isscalar(x):
        return [x]
    return list(x)

def generate_simulation_seeds(n_sims, master_seed=12345):
    rng = np.random.default_rng(master_seed)
    return rng.integers(0, 2**31, size=n_sims, dtype=np.int64)

def run_evolution_and_save(Lx, Ly, rho0, beta, epsilon, D, J, h,
                           n_sweeps, snapshot_sweeps, output_dir, seed, max_particles_per_cell):
    """
    MODE 1: Run the system for n_sweeps, keep snapshots at specified sweeps.
    """
    
    # ---- Logging ----
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    params_str = f"rho={rho0}, beta={beta}, eps={epsilon}, J={J}, h={h}"
    print(f"[{timestamp}] START: {params_str}")
    # ---- End logging ----

    # 1. initialize system with an explicit per-simulation seed
    seed_numba(seed)
    pos_x, pos_y, spins, n_plus, n_minus, N, dt = \
        initialize_system(Lx, Ly, rho0, beta, J, h, D=D)

    n_snaps = len(snapshot_sweeps)
    snapshots_plus = np.zeros((n_snaps, Lx, Ly), dtype=np.int32)
    snapshots_minus = np.zeros((n_snaps, Lx, Ly), dtype=np.int32)

    # 2. precompute rates to save time in the inner loop
    # Calculate hopping probability constants [s=-1, s=+1]
    p_R_const = np.array([D * (1.0 - epsilon) * dt, D * (1.0 + epsilon) * dt])
    p_L_const = np.array([D * (1.0 + epsilon) * dt, D * (1.0 - epsilon) * dt])
    p_UD_const = D * dt

    # Build the np.exp() lookup table
    # Shape: (2 spins, max_rho + 1, max_rho*2 + 1 to handle negative m)
    flip_table = np.zeros((2, max_particles_per_cell + 1, max_particles_per_cell * 2 + 1))
    for s_val in (-1, 1):
        s_idx = 0 if s_val == -1 else 1
        for r_val in range(1, max_particles_per_cell + 1):
            for m_val in range(-r_val, r_val + 1):
                m_idx = m_val + max_particles_per_cell # Shift index to avoid negative arrays
                flip_rate = np.exp(-beta * s_val * (J * m_val / r_val + h))
                flip_table[s_idx, r_val, m_idx] = flip_rate * dt

    # 3. run sweeps
    run_sweeps_with_snapshots(pos_x, pos_y, spins, n_plus, n_minus, N, dt,
        beta, D, epsilon, J, h,
        n_sweeps, snapshot_sweeps, snapshots_plus, snapshots_minus, 
        flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell
    )

    # make filename unique and readable
    filename = f"aim_L{Lx}x{Ly}_rho{rho0:.0f}_beta{beta:.1f}_eps{epsilon:.2f}_J{J:.2f}_h{h:.3f}.npz"
    path = os.path.join(output_dir, filename)

    np.savez_compressed(
        path,
        Lx=Lx, Ly=Ly, N=N,
        rho0=rho0, beta=beta, epsilon=epsilon, D=D, J=J, h=h,
        snapshot_sweeps=snapshot_sweeps,
        snapshots_plus=snapshots_plus, snapshots_minus=snapshots_minus,
        seed=int(seed)
    )

    # ---- Logging ----
    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] FINISH: {params_str} (Duration: {duration:.2f} hrs)")
    # ---- End logging ----

    return path


#--------------------------------------------------------------------------
# High-level sweeping functions (for ness, no snapshots)
# returns profile and entropy production rate
#--------------------------------------------------------------------------
@njit(cache=True)
def compute_shifted_profiles(n_plus, n_minus):
    """
    Computes the y-averaged density and magnetization profiles for a single snapshot,
    shifted so the band center (calculated via circular center of mass) is perfectly aligned at Lx // 2.
    """
    Lx, Ly = n_plus.shape
    rho_x = np.zeros(Lx, dtype=np.float64)
    m_x = np.zeros(Lx, dtype=np.float64)

    # 1. Calculate unshifted 1D density
    raw_rho_x = np.zeros(Lx, dtype=np.float64)
    for x in range(Lx):
        for y in range(Ly):
            raw_rho_x[x] += n_plus[x, y] + n_minus[x, y]

    # 2. Robust Circular Center of Mass
    sum_cos = 0.0
    sum_sin = 0.0
    for x in range(Lx):
        theta = 2.0 * np.pi * x / Lx
        sum_cos += raw_rho_x[x] * np.cos(theta)
        sum_sin += raw_rho_x[x] * np.sin(theta)

    avg_theta = np.arctan2(sum_sin, sum_cos)
    if avg_theta < 0:
        avg_theta += 2.0 * np.pi

    center_x = int(np.round(avg_theta * Lx / (2.0 * np.pi))) % Lx
    shift = (Lx // 2) - center_x
    
    # 3. Accumulate shifted profiles
    for x in range(Lx):
        shifted_x = (x + shift) % Lx
        for y in range(Ly):
            rho_x[shifted_x] += n_plus[x, y] + n_minus[x, y]
            m_x[shifted_x] += n_plus[x, y] - n_minus[x, y]
            
    # 4. Average over the y-axis
    for x in range(Lx):
        rho_x[x] /= Ly
        m_x[x] /= Ly
        
    return rho_x, m_x

def run_single_ness(Lx, Ly, rho0, beta, epsilon, D, J, h,
                    n_relax_sweeps, n_samples, max_particles_per_cell, seed):
    """
    MODE 2: Run the system until it relaxes to NESS, then sample snapshots to compute final profiles and EPR.
     - n_relax_sweeps: number of sweeps to run for relaxation before sampling
     - n_samples: number of snapshots to sample after relaxation for computing averages
     - max_particles_per_cell: cap for histogram binning to prevent out-of-bounds errors
     Returns: 
     total_entropy (float), mean_energy (float), 
     EJ_list (list), Eh_list (list): lists of the separate energy components for each sampled snapshot
     mean_rho_x (array), mean_m_x (array): time-averaged profiles across all sampled snapshots
     final_rho_x (array), final_m_x (array): profile from from the last sampled snapshot
     entropy_production_rate (float): computed as the average log ratio of forward/reverse path probabilities across all attempted updates in the sampled snapshots. At steady state this tis the total (i.e., system + bath) EPR.
     Note: The profiles returned here are already shifted to the comoving frame using the circular center of mass method.
    """
    # 0. initialize system
    seed_numba(seed)
    pos_x, pos_y, spins, n_plus, n_minus, N, dt = \
        initialize_system(Lx, Ly, rho0, beta, J, h, D=D)
    
    # 1. precompute rates to save time in the inner loop
    # Calculate hopping probability constants [s=-1, s=+1]
    p_R_const = np.array([D * (1.0 - epsilon) * dt, D * (1.0 + epsilon) * dt])
    p_L_const = np.array([D * (1.0 + epsilon) * dt, D * (1.0 - epsilon) * dt])
    p_UD_const = D * dt

    # Build the np.exp() lookup table
    # Shape: (2 spins, max_rho + 1, max_rho*2 + 1 to handle negative m)
    flip_table = np.zeros((2, max_particles_per_cell + 1, max_particles_per_cell * 2 + 1))
    for s_val in (-1, 1):
        s_idx = 0 if s_val == -1 else 1
        for r_val in range(1, max_particles_per_cell + 1):
            for m_val in range(-r_val, r_val + 1):
                m_idx = m_val + max_particles_per_cell # Shift index to avoid negative arrays
                flip_rate = np.exp(-beta * s_val * (J * m_val / r_val + h))
                flip_table[s_idx, r_val, m_idx] = flip_rate * dt
    
    # 2. relax to NESS
    run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_relax_sweeps, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell)

    hist = np.zeros((Lx, max_particles_per_cell, max_particles_per_cell), dtype=np.int64)
    accumulate_ness_energy = 0.0
    accumulate_entropy_production = 0.0
    EJ_list = []
    Eh_list = []
    
    # Accumulators for the new profiles
    sum_rho_x = np.zeros(Lx, dtype=np.float64)
    sum_m_x = np.zeros(Lx, dtype=np.float64)

    # 3. sampling sweeps
    for _ in range(n_samples):
        log_ratios = run_sweeps_numba_epr(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, 1, flip_table, p_R_const, p_L_const, p_UD_const, max_particles_per_cell)
        accumulate_entropy_production += np.sum(log_ratios) # log_ratios is an array of all attempted updates in the sweep, including no-ops.
        accumulate_ness_histogram(n_plus, n_minus, hist)
        accumulate_ness_energy += compute_total_energy(n_plus, n_minus, J, h)
        EJ, Eh = compute_energy_components(n_plus, n_minus)
        EJ_list.append(EJ)
        Eh_list.append(Eh)
        
        # Compute shifted profiles for this snapshot and add to sum
        snap_rho_x, snap_m_x = compute_shifted_profiles(n_plus, n_minus)
        sum_rho_x += snap_rho_x
        sum_m_x += snap_m_x
        
    # 4. compute final energy, final entropy and EPR
    mean_energy = accumulate_ness_energy / n_samples
    total_entropy = 0.0
    for x in range(Lx):
        counts = hist[x].flatten()
        total_samples = np.sum(counts)
        if total_samples > 0:
            probs = counts[counts > 0] / total_samples
            S_x = -np.sum(probs * np.log(probs))
            total_entropy += S_x * Ly
    entropy_production_rate = accumulate_entropy_production / (n_samples * dt) # each sweep corresponds to physical time dt. 

    # 5. Compute time-averaged and final profiles
    mean_rho_x = sum_rho_x / n_samples
    mean_m_x = sum_m_x / n_samples
    final_rho_x = snap_rho_x   # State of the last snapshot taken
    final_m_x = snap_m_x
    
    return total_entropy, mean_energy, EJ_list, Eh_list, mean_rho_x, mean_m_x, final_rho_x, final_m_x, entropy_production_rate

def run_multiple_ness(params, n_relax_sweeps, n_sample_sweeps,
                      max_particles_per_cell, n_sims, n_jobs=1, master_seed=12345):
    """
    Parallel wrapper for run_single_ness.
    """
    seeds = generate_simulation_seeds(n_sims, master_seed=master_seed) # use same master seed for different (J, h) pairs so that simulation with the same ID across pairs have the same initial conditions.
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_ness)(
            params['Lx'], params['Ly'], params['rho0'], params['beta'],
            params['epsilon'], params['D'], params['J'], params['h'],
            n_relax_sweeps, n_sample_sweeps, max_particles_per_cell, int(seeds[i])
        )
        for i in range(n_sims)
    )
    return results

#--------------------------------------------------------------------------
# Refined mean-field model
#--------------------------------------------------------------------------

def coeffs(beta, h, J, alpha_m):
    C2 = beta * J * (beta * J - 2.0) * np.sinh(beta * h)
    C3 = beta**2 * J**2 * (1.0 - beta * J / 3.0) * np.cosh(beta * h)
    r_tilde = 1.5 * alpha_m * C3
    return C2, C3, r_tilde

def C0(rho0, beta, h, C2, alpha_m):
    return 2.0 * rho0 * np.sinh(beta * h) + alpha_m * C2

def C1(rho0, beta, h, J, r_tilde):
    return 2.0 * ((beta * J - 1.0) * np.cosh(beta * h) - r_tilde / rho0)

def homogeneous_roots(rho0, beta, h, J, alpha_m):
    C2, C3, r_tilde = coeffs(beta, h, J, alpha_m)
    c0 = C0(rho0, beta, h, C2, alpha_m)
    c1 = C1(rho0, beta, h, J, r_tilde)

    # cubic in m:
    # 0 = c0 + c1 m + (C2/rho0) m^2 - (C3/rho0^2) m^3
    poly = np.array([
        -C3 / rho0**2,
        C2 / rho0,
        c1,
        c0
    ], dtype=float)

    roots = np.roots(poly)
    real_roots = roots[np.isclose(roots.imag, 0.0, atol=1e-9)].real
    real_roots = np.unique(np.round(real_roots, 12))
    return real_roots, C2, C3, r_tilde

def F_derivatives(rho0, m0, beta, h, J, alpha_m, C2, C3, r_tilde):
    Fm = C1(rho0, beta, h, J, r_tilde) + 2.0 * C2 * m0 / rho0 - 3.0 * C3 * m0**2 / rho0**2
    Frho = (
        2.0 * np.sinh(beta * h)
        + 2.0 * r_tilde * m0 / rho0**2
        - C2 * m0**2 / rho0**2
        + 2.0 * C3 * m0**3 / rho0**3
    )
    return Frho, Fm

def max_growth_rate(rho0, m0, beta, h, J, v, D, alpha_m, q_min =0.0, qmax=10.0, nq=400):
    roots, C2, C3, r_tilde = homogeneous_roots(rho0, beta, h, J, alpha_m)
    Frho, Fm = F_derivatives(rho0, m0, beta, h, J, alpha_m, C2, C3, r_tilde)

    qxs = np.linspace(q_min, qmax, nq) # avoid q=0 to prevent singularity in the matrix
    max_real = -np.inf

    for qx in qxs:
        q2 = qx**2
        M = np.array([
            [-D * q2, -1j * qx * v],
            [Frho - 1j * qx * v, -D * q2 + Fm]
        ], dtype=complex)
        lam = eigvals(M)
        max_real = max(max_real, np.max(lam.real))

    return max_real

def classify_point(rho0, beta, h, J, v, D, alpha_m):
    roots, C2, C3, r_tilde = homogeneous_roots(rho0, beta, h, J, alpha_m)
    TOLERANCE = 1e-9

    if len(roots) == 0:
        return {"n_roots": 0, "stable_roots": 0, "label": "no real root"}

    stable_count = 0
    growths = []

    for m0 in roots:
        sigma = max_growth_rate(rho0, m0, beta, h, J, v, D, alpha_m)
        growths.append((m0, sigma))
        if sigma < TOLERANCE:
            stable_count += 1

    if stable_count == 0:
        label = "all homogeneous roots unstable"
    elif stable_count == 1:
        label = "one stable homogeneous root"
    else:
        label = "multistable homogeneous roots"

    return {
        "n_roots": len(roots),
        "stable_roots": stable_count,
        "growths": growths,
        "label": label
    }

def compute_h_J_phase(hs, Js, rho0, D, alpha_m, epsilon, beta=1.0, output_path="phase_diagram.json"):
    phase = np.zeros((len(hs), len(Js)), dtype=int)

    for i, h in enumerate(hs):
        for j, J in enumerate(Js):
            out = classify_point(rho0=rho0, beta=beta, h=h, J=J, v=2*D*epsilon, D=D, alpha_m=alpha_m)
            if out["label"] == "all homogeneous roots unstable":
                phase[i, j] = 1
            elif out["label"] == "one stable homogeneous root":
                if np.isclose(out["growths"][0][0], 0.0, atol=1e-3):
                    phase[i, j] = 0
                else:
                    phase[i, j] = 2
            elif out["label"] == "multistable homogeneous roots":
                phase[i, j] = 3

    # save data to json file
    with open(output_path, 'w') as f:
        json.dump({
            "hs": hs.tolist(),
            "Js": Js.tolist(),
            "phase": phase.tolist(),
            "rho0": rho0,
            "beta": beta,
            "v": 2*D*epsilon,
            "D": D,
            "alpha_m": alpha_m
        }, f)


#--------------------------------------------------------------------------
# Analysis functions
#--------------------------------------------------------------------------
def get_profiles_multi_sim(data, idxs=[-1]):
    rho_x = data['final_rho_x'] # shape (n_Js, n_sims, Lx)
    m_x = data['final_m_x'] # shape (n_Js, n_sims, Lx)
    # idxs are the indexes of J to plot
    rho_x_multi = rho_x[idxs]
    m_x_multi = m_x[idxs]
    return rho_x_multi, m_x_multi

def compute_eta_first_principles(data):
    entropies = data['entropies']
    E_Js = data['E_Js']
    J_list = data['J_list']
    mean_entropy = np.mean(entropies, axis=1)
    dS_dJ = np.gradient(mean_entropy, J_list) # shape (n_Js,)
    dW_dJ = E_Js.mean(axis=(1,2)) # shape (n_Js, n_sims, n_sample_sweeps), J->J+, 
    eta = -dS_dJ / dW_dJ
    return eta, -dS_dJ, dW_dJ

def compute_eta_inferential(data):
    E_Js = data['E_Js']
    J_list = data['J_list']
    num_etaInf = -J_list * np.var(E_Js, axis=(1,2)) # shape (n_Js, n_sims, n_sample_sweeps) -> (n_Js,)
    den_etaInf = np.mean(-E_Js, axis=(1,2)) # shape (n_Js,), E_J does not include the negative sign, need to add that to match the definition of conjugate observable
    etaInf = num_etaInf / den_etaInf
    return etaInf, num_etaInf, den_etaInf