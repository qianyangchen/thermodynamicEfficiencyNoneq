#!/usr/bin/env python
import json
import numpy as np
from numba import njit
import os
from joblib import Parallel, delayed
import time
import datetime
from numpy.linalg import eigvals

TOLERANCE = 1e-9 # for root classification

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
def step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h):
    """
    Take one step in simulation for one randomly chosen particle.
    """
    Lx, Ly = n_plus.shape
    N = len(pos_x)

    # randomly choose a particle to act on
    idx = np.random.randint(N)
    s = spins[idx]
    x = pos_x[idx]
    y = pos_y[idx]
    rho = n_plus[x, y] + n_minus[x, y]
    m = n_plus[x, y] - n_minus[x, y]
    flip_rate = np.exp(-beta * s * (J * m / rho + h))

    # Partition probability space with a SINGLE random number
    r = np.random.rand()

    # Cumulative probability thresholds, actions are mutually exclusive.
    p_flip = flip_rate * dt
    p_R = p_flip + D * (1.0 + s * epsilon) * dt
    p_L = p_R + D * (1.0 - s * epsilon) * dt
    p_U = p_L + D * dt
    p_D = p_U + D * dt

    # Execute event
    if r < p_flip:
        spins[idx] = -s
        if s == 1:
            n_plus[x, y] -= 1
            n_minus[x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_plus[x, y] += 1

    elif r < p_R:
        new_x = (x + 1) % Lx
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1

    elif r < p_L:
        new_x = (x - 1) % Lx
        pos_x[idx] = new_x
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[new_x, y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[new_x, y] += 1

    elif r < p_U:
        new_y = (y + 1) % Ly
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1

    elif r < p_D:
        new_y = (y - 1) % Ly
        pos_y[idx] = new_y
        if s == 1:
            n_plus[x, y] -= 1
            n_plus[x, new_y] += 1
        else:
            n_minus[x, y] -= 1
            n_minus[x, new_y] += 1

    # If r > p_D, the particle does nothing.


@njit(cache=True)
def run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_sweeps):
    """
    Each sweep = N updates. N is the total number of particles.
    """
    N = len(pos_x)
    for _ in range(n_sweeps):
        for _ in range(N):
            step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h)


@njit(cache=True)
def run_sweeps_with_snapshots(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_sweeps, snapshot_sweeps, snapshots_plus, snapshots_minus):
    """
    Runs the simulation for a total of n_sweeps, and saves snapshots of n_plus and n_minus at specified sweeps.
    snapshot_sweeps: list of sweep numbers at which to save snapshots (e.g. [0, 10, 100, 1000])
    snapshots_plus/minus: pre-allocated arrays of shape (len(snapshot_sweeps), Lx, Ly) to store the snapshots.
    """
    N = len(pos_x)
    n_snaps = len(snapshot_sweeps)
    snap_idx = 0

    for sweep in range(n_sweeps):
        # run one sweep (N updates)
        for _ in range(N):
            step_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h)

        # check if current sweep is a snapshot
        if snap_idx < n_snaps and sweep == snapshot_sweeps[snap_idx]:
            snapshots_plus[snap_idx] = n_plus
            snapshots_minus[snap_idx] = n_minus
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
                           n_sweeps, snapshot_sweeps, output_dir, seed):
    """
    MODE 1: Run the system for n_sweeps, keep snapshots at specified sweeps.
    """
    
    # ---- Logging ----
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    params_str = f"rho={rho0}, beta={beta}, eps={epsilon}, J={J}, h={h}"
    print(f"[{timestamp}] START: {params_str}")
    # ---- End logging ----

    # initialize system with an explicit per-simulation seed
    seed_numba(seed)
    pos_x, pos_y, spins, n_plus, n_minus, N, dt = \
        initialize_system(Lx, Ly, rho0, beta, J, h, D=D)

    n_snaps = len(snapshot_sweeps)
    snapshots_plus = np.zeros((n_snaps, Lx, Ly), dtype=np.int16)
    snapshots_minus = np.zeros((n_snaps, Lx, Ly), dtype=np.int16)

    run_sweeps_with_snapshots(pos_x, pos_y, spins, n_plus, n_minus, N, dt,
        beta, D, epsilon, J, h,
        n_sweeps, snapshot_sweeps, snapshots_plus, snapshots_minus
    )

    filename = f"snapshots_L{Lx}x{Ly}_rho{rho0:.0f}_beta{beta:.1f}_eps{epsilon:.1f}_J{J:.1f}_h{h:.1f}.npz"
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
#--------------------------------------------------------------------------

def run_single_ness(Lx, Ly, rho0, beta, epsilon, D, J, h,
                    n_relax_sweeps, n_samples, max_particles_per_cell, seed):
    """
    MODE 2: Run one simulation to NESS, aggregate entropy histograms on-the-fly, compute final energy.
    n_relax_sweeps: number of sweeps to relax the system to NESS before sampling.
    n_samples: number of snapshots to sample after relaxation, for histogram and energy estimation.
    """

    # 0. initialize system with an explicit per-simulation seed
    seed_numba(seed)
    pos_x, pos_y, spins, n_plus, n_minus, N, dt = \
        initialize_system(Lx, Ly, rho0, beta, J, h, D=D)
    
    # 1. relax to NESS
    run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_relax_sweeps)

    # 2. allocate histogram array for entropy, and a variable to accumulate energy
    # Each element: [x_coord, num_plus, num_minus], counts how many times we see (ni+,ni-) at that x-coordinate in the comoving frame.
    hist = np.zeros((Lx, max_particles_per_cell, max_particles_per_cell), dtype=np.int64)
    accumulate_ness_energy = 0.0
    EJ_list = []
    Eh_list = []

    # 3. sampling snapshots, accumulate histogram and energy on-the-fly
    for _ in range(n_samples):
        # make one sweep, then accumulate histogram for this snapshot
        run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, 1)
        accumulate_ness_histogram(n_plus, n_minus, hist)
        accumulate_ness_energy += compute_total_energy(n_plus, n_minus, J, h)
        EJ, Eh = compute_energy_components(n_plus, n_minus)
        EJ_list.append(EJ)
        Eh_list.append(Eh)
    
    # 4. compute final energy
    mean_energy = accumulate_ness_energy / n_samples

    # 5. compute total entropy from the histogram
    total_entropy = 0.0
    for x in range(Lx):
        # For location x, flatten the 2D grid of (n+, n-) counts for this x-coordinate
        counts = hist[x].flatten()
        total_samples = np.sum(counts)
        
        if total_samples > 0:
            # Filter out zero-counts to avoid log(0)
            probs = counts[counts > 0] / total_samples
            S_x = -np.sum(probs * np.log(probs))
            
            # Multiply by Ly (total entropy for Ly cells at this x-coordinate)
            total_entropy += S_x * Ly
    
    return total_entropy, mean_energy, EJ_list, Eh_list


def run_multiple_ness(params, n_relax_sweeps, n_sample_sweeps,
                      max_particles_per_cell, n_sims, n_jobs=1, master_seed=12345):
    """
    Run multiple NESS simulations in parallel and collect results.
    params: dictionary of parameters (Lx, Ly, rho0, beta, epsilon, D, J, h)
    Returns a list of tuples: [(entropy1, energy1, EJ1, Eh1), (entropy2, energy2, EJ2, Eh2), ...]
    """
    seeds = generate_simulation_seeds(n_sims, master_seed=master_seed)
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
# Profile Extraction and Extended NESS Functions (for ness, no snapshots)
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


def run_single_ness_with_profiles(Lx, Ly, rho0, beta, epsilon, D, J, h,
                    n_relax_sweeps, n_samples, max_particles_per_cell, seed):
    """
    Extended version of run_single_ness to include profiles rho_x, m_x.
    """
    # 0. initialize system
    seed_numba(seed)
    pos_x, pos_y, spins, n_plus, n_minus, N, dt = \
        initialize_system(Lx, Ly, rho0, beta, J, h, D=D)
    
    # 1. relax to NESS
    run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, n_relax_sweeps)

    hist = np.zeros((Lx, max_particles_per_cell, max_particles_per_cell), dtype=np.int64)
    accumulate_ness_energy = 0.0
    EJ_list = []
    Eh_list = []
    
    # Accumulators for the new profiles
    sum_rho_x = np.zeros(Lx, dtype=np.float64)
    sum_m_x = np.zeros(Lx, dtype=np.float64)

    # 3. sampling snapshots
    for _ in range(n_samples):
        run_sweeps_numba(pos_x, pos_y, spins, n_plus, n_minus, N, dt, beta, D, epsilon, J, h, 1)
        accumulate_ness_histogram(n_plus, n_minus, hist)
        accumulate_ness_energy += compute_total_energy(n_plus, n_minus, J, h)
        EJ, Eh = compute_energy_components(n_plus, n_minus)
        EJ_list.append(EJ)
        Eh_list.append(Eh)
        
        # Compute shifted profiles for this snapshot and add to sum
        snap_rho_x, snap_m_x = compute_shifted_profiles(n_plus, n_minus)
        sum_rho_x += snap_rho_x
        sum_m_x += snap_m_x
        
    # 4. compute final energy and entropy
    mean_energy = accumulate_ness_energy / n_samples
    total_entropy = 0.0
    for x in range(Lx):
        counts = hist[x].flatten()
        total_samples = np.sum(counts)
        if total_samples > 0:
            probs = counts[counts > 0] / total_samples
            S_x = -np.sum(probs * np.log(probs))
            total_entropy += S_x * Ly

    # 5. Compute time-averaged and final profiles
    mean_rho_x = sum_rho_x / n_samples
    mean_m_x = sum_m_x / n_samples
    final_rho_x = snap_rho_x   # State of the last snapshot taken
    final_m_x = snap_m_x
    
    return total_entropy, mean_energy, EJ_list, Eh_list, mean_rho_x, mean_m_x, final_rho_x, final_m_x


def run_multiple_ness_with_profiles(params, n_relax_sweeps, n_sample_sweeps,
                      max_particles_per_cell, n_sims, n_jobs=1, master_seed=12345):
    """
    Parallel wrapper for run_single_ness_with_profiles.
    """
    seeds = generate_simulation_seeds(n_sims, master_seed=master_seed)
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_ness_with_profiles)(
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

def compute_h_J_phase(hs, Js, rho0, D, alpha_m, epsilon, beta=1.0):
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

    return phase


#--------------------------------------------------------------------------
# Data visualisation and analysis functions
#--------------------------------------------------------------------------
# get snapshot and profiles
def get_snapshot_and_profiles(snapdata, idx=-1):
    snapshots_plus = snapdata['snapshots_plus']
    snapshots_minus = snapdata['snapshots_minus']
    snapshot = snapshots_plus[idx] - snapshots_minus[idx]
    rho_x = (snapshots_plus[idx] + snapshots_minus[idx]).mean(axis=1)
    m_x = (snapshots_plus[idx] - snapshots_minus[idx]).mean(axis=1)
    return snapshot, rho_x, m_x

def get_profiles_multi_sim(data, idxs=[-1]):
    rho_x = data['final_rho_x'] # shape (n_Js, n_sims, Lx)
    m_x = data['final_m_x'] # shape (n_Js, n_sims, Lx)
    # idxs are the indexes of J to plot
    rho_x_multi = rho_x[idxs]
    m_x_multi = m_x[idxs]
    return rho_x_multi, m_x_multi

# compute eta
def compute_eta_first_principles(data):
    entropies = data['entropies']
    E_Js = data['E_Js']
    J_list = data['J_list']
    mean_entropy = np.mean(entropies, axis=1)
    dS_dJ = np.gradient(mean_entropy, J_list)[:-1] # shape (n_Js-1,)
    dJs = np.diff(J_list) # shape (n_Js-1,)
    w_fwd = dJs[:, np.newaxis, np.newaxis] * E_Js[:-1] # shape (n_Js-1, n_sims, n_sample_sweeps), J->J+, 
    w_fwd_mean = w_fwd.mean(axis=(1,2)) # shape (n_Js-1,)
    w_fwd_rate = w_fwd_mean / dJs # shape (n_Js-1,)
    eta = -dS_dJ / w_fwd_rate
    return eta, -dS_dJ, w_fwd_rate

def compute_eta_inferential(data):
    E_Js = data['E_Js']
    J_list = data['J_list']
    num_etaInf = -J_list * np.var(E_Js, axis=(1,2)) # shape (n_Js, n_sims, n_sample_sweeps) -> (n_Js,)
    den_etaInf = np.mean(-E_Js, axis=(1,2)) # shape (n_Js,), E_J does not include the negative sign, need to add that to match the definition of conjugate observable
    etaInf = num_etaInf / den_etaInf
    return etaInf, num_etaInf, den_etaInf
