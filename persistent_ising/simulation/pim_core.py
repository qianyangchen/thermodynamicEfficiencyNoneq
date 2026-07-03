#!/usr/bin/env python
import numpy as np
from numba import njit
from joblib import Parallel, delayed

# ==== Constants ====
EPSILON = 1e-6 # to avoid log(p) getting too large due to random fluctuation
DENO_EPS = 0.05 # to avoid division by zero
GLAUBER = 0 
METROPOLIS = 1 # for faster numba compilation, use integers to represent algorithms instead of strings
#--------------------------------------------------------------------------
# Low-level (Numba-accelerated) functions
#--------------------------------------------------------------------------

@njit(nogil=True)
def initialise(L, bias, seed_val):
    """
    Initialize an LxL lattice with values in {1, -1}.
    Each site is set to 1 with probability `bias`, otherwise -1.
    Uses vectorized operations.
    """
    np.random.seed(seed_val)
    rand_matrix = np.random.rand(L, L)
    lattice = np.where(rand_matrix < bias, 1, -1)
    return lattice

@njit(nogil=True)
def get_mu(lattice):
    """
    Compute the net interaction energy (-sum_{m<n} s_m s_n) given lattice configuration. Periodic boundary condition applied.
    """
    L = lattice.shape[0]
    mu = 0.0
    for i in range(L):
        for j in range(L):
            down  = lattice[(i + 1) % L, j]
            right = lattice[i, (j + 1) % L]
            mu += -lattice[i, j] * (down + right)
    return mu
    
@njit(nogil=True)
def step_glauber(lattice, J, temperature, mu, E0, h):
    """
    Take one Monte Carlo step of the Glauber algorithm for persistent Ising model.
    
    Parameters:
      lattice (2D array of int64): The current lattice configuration.
      J (float): Coupling strength.
      temperature (float): Temperature of the system.
      mu (float): Current net interaction energy.
      E0 (float): Additional energy offset for this time step.
      h (float): External field at this time step.

    Returns:
      mu (float): Updated net interaction energy after the step.
      spin_i (int): The initial spin at the chosen site before the flip.
      spin_f (int): The final spin at the chosen site after the flip (may be the same as spin_i if no flip).
      sum_nb (int): The sum of neighboring spins at the chosen site.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    beta = 1.0 / temperature
    L = lattice.shape[0]
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)
    spin_i = lattice[x, y]
    spin_f = -spin_i
    sum_nb = (lattice[(x - 1) % L, y] +
                lattice[(x + 1) % L, y] +
                lattice[x, (y - 1) % L] +
                lattice[x, (y + 1) % L])
    # Compute local energy contribution at the chosen site.
    mu_i = -spin_i * sum_nb
    mu_f = -spin_f * sum_nb
    # H(s) = -J*sum(si*sj) - h*si
    # dE = H(s_f) - H(s_i)
    dE = J * (mu_f - mu_i) + h * (-spin_f + spin_i) #negative sign included in mu_i and mu_f
    dE_eff = dE + E0
    if np.random.rand() < 1.0/(1.0 + np.exp(beta * dE_eff)):
        # flip spin
        lattice[x, y] = spin_f
        mu += (mu_f - mu_i)
    return mu, spin_i, lattice[x, y], sum_nb

@njit(nogil=True)
def run_sweeps(lattice, n_sweeps, J, temperature, mu, E0, h):
    """
    Run one or multiple sweeps of the chosen algorithm for persistent Ising model. One sweep = lattice.size steps.
    
    Parameters:
      lattice (2D array of int64): The initial lattice configuration.
      n_sweeps (int): Number of sweeps to perform.
      J (float): Coupling strength.
      temperature (float): Temperature of the system.
      mu (float): Initial net interaction energy.
      E0 (float): Energy offset for persistent Ising model.
      h (float): External field.

    Returns:
      lattice (2D array of int64): The final lattice configuration after the sweeps.
      total_magnetisation (float): total magnetisation at the end of the sweep.
      mu (float): [-sum_{m<n} s_m s_n] at the end of the sweep.
      spins_i (1D array of floats, size=lattice.size): initial spin at each recorded time step.
      spins_f (1D array of floats, size=lattice.size): final spin at each recorded time step.
      sum_neighbors (1D array of floats, size=lattice.size): sum of neighboring spins at each recorded time step.
    """
    L = lattice.shape[0]
    N = L * L
    beta = 1.0 / temperature

    # Lookup table for transition probabilities to speed up computation, indexed by current spin and sum of neighbors.
    prob_table = np.zeros((2, 9))
    for s in (-1, 1):
        for snb in (-4, -2, 0, 2, 4):
            dE = 2.0 * J * s * snb + 2.0 * h * s
            dE_eff = dE + E0
            s_idx = 0 if s == -1 else 1
            snb_idx = snb + 4

            # GLAUBER
            prob_table[s_idx, snb_idx] = 1.0 / (1.0 + np.exp(beta * dE_eff))

    spins_i = np.zeros(N)
    spins_f = np.zeros(N)
    sum_neighbors = np.zeros(N)
    
    for sweep in range(n_sweeps):
        is_last_sweep = (sweep == n_sweeps - 1)
        
        for step in range(N):
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)
            spin_i = lattice[x, y]
            
            sum_nb = (lattice[(x - 1) % L, y] +
                      lattice[(x + 1) % L, y] +
                      lattice[x, (y - 1) % L] +
                      lattice[x, (y + 1) % L])
            
            # Use the lookup table
            s_idx = 0 if spin_i == -1 else 1
            snb_idx = sum_nb + 4
            flip_prob = prob_table[s_idx, snb_idx]
            
            spin_f = spin_i
            if np.random.rand() < flip_prob:
                spin_f = -spin_i
                lattice[x, y] = spin_f
                # Update mu directly
                mu += -spin_f * sum_nb - (-spin_i * sum_nb)

            if is_last_sweep:
                spins_i[step] = spin_i
                spins_f[step] = spin_f
                sum_neighbors[step] = sum_nb
                
    total_magnetisation = np.sum(lattice)
    return lattice, total_magnetisation, mu, spins_i, spins_f, sum_neighbors

@njit(nogil=True)
def compute_entropy_production_rate_glauber(spins_i, spins_f, sum_neighbors, J, E0, h, temperature):
    """
    Compute the entropy production rate at NESS using the time series data of spins and their neighbors. EPR = sum_{tau_obs} log[W(s_i->s_f)/W(s_f->s_i)]/tau_obs, where W(s_i->s_f) is the transition rate from s_i to s_f, and tau_obs is the number of Monte Carlo steps. W(s_i->s_f) = 1/(1+exp(beta*dE_eff)) for Glauber dynamics, and min(1, exp(-beta*dE_eff)) for Metropolis dynamics, where dE_eff = dE + E0 is the effective energy change including the persistent energy offset E0.
    
    Parameters:
      spins_i (1D array of floats): initial spin at each recorded time step.
      spins_f (1D array of floats): final spin at each recorded time step.
      sum_neighbors (1D array of floats): sum of neighboring spins at each recorded time step.
      J (float): Coupling strength.
      E0 (float): Energy offset for persistent Ising model.
      h (float): External field.
      temperature (float): Temperature of the system.   
    Returns:
      entropy_production_rate (float): Estimated entropy production rate at NESS.
      tau_obs (int): Number of transition observations used in the estimation.
    """
    beta = 1.0 / temperature
    dE = J * (2 * spins_i * sum_neighbors) + h * (2 * spins_i) # energy change if the spin flips
    log_W_fwd = -np.log(1.0 + np.exp(beta * (dE + E0))) # log W(s_i->s_f) for Glauber dynamics
    log_W_rev = -np.log(1.0 + np.exp(beta * (-dE + E0))) # log W(s_f->s_i) for Glauber dynamics
    log_ratio = np.where(spins_i != spins_f, log_W_fwd - log_W_rev, 0.0) # log[W(s_i->s_f)/W(s_f->s_i)], only nonzero when a transition is observed
    
    tau_obs = len(spins_i) # total number of Monte Carlo steps
    entropy_production_rate = np.sum(log_ratio) / tau_obs
    return entropy_production_rate, tau_obs

@njit(nogil=True)
def compute_entropy_kikuchi(lattice):
    """ 
    Compute configuration entropy using kikuchi approximation S = S1 - 2*S2 + S4.
    Optimized for Numba using bitwise state mapping and a single lattice pass.
    """
    L = lattice.shape[0]
    N = L * L
    
    # Pre-allocate arrays to act as our frequency histograms.
    # 2 states for 1x1, 4 states for 1x2, 16 states for 2x2.
    counts_1 = np.zeros(2, dtype=np.int64)
    counts_2 = np.zeros(4, dtype=np.int64)
    counts_4 = np.zeros(16, dtype=np.int64)
    
    # Single pass over the lattice
    for i in range(L):
        for j in range(L):
            # Map spins from {-1, 1} to {0, 1}
            # Bitwise right-shift '>> 1' turns 0 into 0, and 2 into 1.
            s00 = (lattice[i, j] + 1) >> 1
            s01 = (lattice[i, (j + 1) % L] + 1) >> 1
            s10 = (lattice[(i + 1) % L, j] + 1) >> 1
            s11 = (lattice[(i + 1) % L, (j + 1) % L] + 1) >> 1
            
            # --- 1x1 configuration (1 spin) ---
            idx_1 = s00
            counts_1[idx_1] += 1
            
            # --- 1x2 configuration (horizontal pair) ---
            # Shift s00 left by 1 bit, combine with s01 using bitwise OR
            idx_2 = (s00 << 1) | s01
            counts_2[idx_2] += 1
            
            # --- 2x2 configuration (square cluster) ---
            idx_4 = (s00 << 3) | (s01 << 2) | (s10 << 1) | s11
            counts_4[idx_4] += 1
            
    # Calculate Entropies: S = - sum(p * ln(p))
    S1 = 0.0
    for i in range(2):
        if counts_1[i] > 0:
            p = counts_1[i] / N
            S1 -= p * np.log(p)
            
    S2 = 0.0
    for i in range(4):
        if counts_2[i] > 0:
            p = counts_2[i] / N
            S2 -= p * np.log(p)
            
    S4 = 0.0
    for i in range(16):
        if counts_4[i] > 0:
            p = counts_4[i] / N
            S4 -= p * np.log(p)
            
    return S1 - 2.0 * S2 + S4

@njit(nogil=True)
def compute_binder_cumulant(magnetisations):
    """
    Compute the Binder cumulant given a list of magnetisation values.

    Parameters:
    magnetisations (list or np.array): A list or array of magnetisation values from different samples.
    
    Returns:
    float: The Binder cumulant value.
    """
    m2 = np.mean(magnetisations**2)
    m4 = np.mean(magnetisations**4)
    if m2 == 0:
        return 0.0 # to avoid division by zero, return zero if variance is zero
    else:
        return 1 - (m4 / (3 * m2**2))

@njit(nogil=True)
def run_diagnostic_relaxation(L, max_sweeps, J, temperature, E0, h, bias=0.5):
    """
    Runs a simulation to determine NESS relaxation time.
    """
    # 1. Initialize the system
    # (Using an arbitrary seed just for diagnostic reproducibility)
    lattice = initialise(L, bias, 123)
    mu = get_mu(lattice)
    
    # 2. Pre-allocate tracking arrays
    mus_over_time = np.zeros(max_sweeps)
    mags_over_time = np.zeros(max_sweeps)
    
    # 3. Run sweeps and track observables
    for sweep in range(max_sweeps):
        # Run exactly 1 sweep
        lattice, mag, mu, _, _, _ = run_sweeps(
            lattice, 1, J, temperature, mu, E0, h
        )
        mus_over_time[sweep] = mu
        mags_over_time[sweep] = mag
        
    return mus_over_time, mags_over_time

#--------------------------------------------------------------------------
# Mid-level simulation functions (for ness, no transient snapshots)
#--------------------------------------------------------------------------
def run_single_ness(seed_val, L, n_relax_sweeps, n_samples, J, bias=0.5, temperature=1.0, E0=0.0, h=0.0):
    """
    Run a single simulation of the persistent Ising model until it reaches NESS, compute entropy, entropy production rate and other statistics at NESS. No transient snapshots are saved, only the final NESS data is returned.

    Parameters:
        seed_val (int): Random seed for numba PRNG.
        L (int): Linear size of the lattice (LxL).
        n_relax_sweeps (int): Number of sweeps to allow the system to relax to NESS before sampling.
        n_samples (int): Number of sweeps to sample at NESS for computing statistics.
        J (float): Coupling strength.
        bias (float): Probability of initializing each spin to 1 (default 0.5 for random initialization).
        temperature (float): Temperature of the system.
        E0 (float): Energy offset for persistent Ising model.
        h (float): External field.
    Returns:
        cnfg_entp (float): Configuration entropy at NESS (average over sweeps).
        entp_prod_rate (float): Entropy production rate at NESS (averaged over total transitions).
        tau_obs (int): Total number of observed transitions used in the entropy production estimation.
        mean_sum_s (float): Mean of sum of spins at NESS.
        mean_sum_ss (float): Mean of sum of spin products at NESS.
        var_sum_s (float): Variance of sum of spins at NESS.
        var_sum_ss (float): Variance of sum of spin products at NESS.
        cov_s_ss (float): Covariance of sum of spins and sum of spin products at NESS.
        binder_cumulant (float): Binder cumulant computed from the magnetisation samples at NESS.
    """
    lattice = initialise(L, bias, seed_val)
    mu = get_mu(lattice)

    # Relaxation phase to reach NESS
    lattice, _, mu, _, _, _ = run_sweeps(lattice, n_relax_sweeps, J, temperature, mu, E0, h)

    # Sampling phase at NESS
    cnfg_entp_per_sweep = np.zeros(n_samples)
    sum_log_ratios_per_sweep = np.zeros(n_samples)
    tau_obs_per_sweep = np.zeros(n_samples)
    sum_s_per_sweep = np.zeros(n_samples)
    sum_ss_per_sweep = np.zeros(n_samples)
    
    for sweep in range(n_samples):
        lat, sum_s, mu, spins_i, spins_f, sum_neighbors = run_sweeps(lattice, 1, J, temperature, mu, E0, h)
        sweep_epr, sweep_tau_obs = compute_entropy_production_rate_glauber(spins_i, spins_f, sum_neighbors, J, E0, h, temperature)
        
        cnfg_entp_per_sweep[sweep] = compute_entropy_kikuchi(lat)
        sum_log_ratios_per_sweep[sweep] = sweep_epr * sweep_tau_obs # convert back to total log ratio for this sweep
        tau_obs_per_sweep[sweep] = sweep_tau_obs
        sum_s_per_sweep[sweep] = sum_s
        sum_ss_per_sweep[sweep] = -mu # mu = -sum(si*sj)

    # Compute statistics at NESS
    cnfg_entp = np.mean(cnfg_entp_per_sweep)
    entp_prod_rate = np.sum(sum_log_ratios_per_sweep) / np.sum(tau_obs_per_sweep) if np.sum(tau_obs_per_sweep) > 0 else 0.0
    tau_obs = np.sum(tau_obs_per_sweep)
    mean_sum_s = np.mean(sum_s_per_sweep)
    mean_sum_ss = np.mean(sum_ss_per_sweep)
    var_sum_s = np.var(sum_s_per_sweep)
    var_sum_ss = np.var(sum_ss_per_sweep)
    cov_s_ss = np.cov(sum_s_per_sweep, sum_ss_per_sweep)[0][1] # covariance of sum of spins and sum of spin products
    binder_cumulant = compute_binder_cumulant(sum_s_per_sweep/(L*L)) # compute binder cumulant using magnetisation per spin
    return cnfg_entp, entp_prod_rate, tau_obs, mean_sum_s, mean_sum_ss, var_sum_s, var_sum_ss, cov_s_ss, binder_cumulant

def run_multi_ness(n_sims, L, n_relax_sweeps, n_samples, J, bias=0.5, temperature=1.0, E0=0.0, h=0.0, n_jobs=-1):
    """
    Run multiple simulations of the persistent Ising model until they reach NESS, compute statistics at NESS for each simulation, and return the results as a list.

    Parameters:
        n_sims (int): Number of independent simulations to run.
        n_relax_sweeps (int): Number of sweeps to allow the system to relax to NESS before sampling.
        n_samples (int): Number of sweeps to sample at NESS for computing statistics.
        J (float): Coupling strength.
        bias (float): Probability of initializing each spin to 1 (default 0.5 for random initialization).
        temperature (float): Temperature of the system.
        E0 (float): Energy offset for persistent Ising model.
        h (float): External field.
        n_jobs (int): Number of parallel jobs to run. If -1, use all available cores. (default -1 for all available cores). 
    Returns:
        list of tuples: Each tuple contains the results from a single simulation, in the order of (cnfg_entp, entp_prod_rate, tau_obs, mean_sum_s, mean_sum_ss, var_sum_s, var_sum_ss, cov_s_ss, binder_cumulant).
    """
    # use same seed for each simulation so that J and J+dJ can be matched when computing derivatives for each simulation.
    results = Parallel(n_jobs=n_jobs)(delayed(run_single_ness)(sim_id, L, n_relax_sweeps, n_samples, J, bias, temperature, E0, h) for sim_id in range(n_sims))

    return results

#--------------------------------------------------------------------------
# High-level functions for thermodynamic efficiency
#-------------------------------------------------------------------
def pool_variance(mean_per_sim, var_per_sim, n_per_sim):
    """
    Compute the pooled variance across multiple simulations.
    
    Parameters:
        mean_per_sim (array-like, len=n_sims): An array of means from each simulation.
        var_per_sim (array-like, len=n_sims): An array of variances from each simulation.
        n_per_sim (int): constant sample size across all simulations.
        
    Returns:
        float: The pooled variance.
    """
    mean_per_sim = np.array(mean_per_sim)
    var_per_sim = np.array(var_per_sim)
    total_n = n_per_sim * len(mean_per_sim) # total sample size across all simulations
    
    # Compute the pooled variance using the formula:
    # pooled_var = (sum((n_i - 1) * var_i) - total_n * overall_mean^2) / (total_n - 1)
    sum_x2 = np.sum((n_per_sim - 1) * var_per_sim) + np.sum(n_per_sim * mean_per_sim**2)
    overall_mean = np.mean(mean_per_sim)
    pooled_var = (sum_x2 - total_n * overall_mean**2) / (total_n - 1)
    
    return pooled_var

def compute_eta_first_principle(data, e0_idx=0):
    """
    Compute the thermodynamic efficiency using the first principle definition: eta = -(dS/dJ) / (dW/dJ).
    Assumes the variable is always J.
    """
    L = data['L']
    Js = np.array(data['Js'])
    
    # Extract the 2D slice for the specific E0: Shape becomes (len(Js), n_sims)
    cnfg_entp = np.array(data['cnfg_entp'])[e0_idx]
    mean_sum_ss = np.array(data['mean_sum_ss'])[e0_idx]
    
    # 1. Compute PER SIMULATION quantities shape = (len(Js), n_sims)
    dW_dJ_per_sim = mean_sum_ss / (L**2)
    # Gradient along the J axis (axis=0)
    dS_dJ_per_sim = np.gradient(cnfg_entp, Js, axis=0) 
    
    eta_num_per_sim = -dS_dJ_per_sim
    eta_den_per_sim = dW_dJ_per_sim
    
    # Avoid division by zero
    mask_per_sim = eta_den_per_sim > DENO_EPS
    eta_per_sim = np.where(mask_per_sim, eta_num_per_sim / eta_den_per_sim, np.nan)

    # 2. Compute AVERAGED quantities shape = (len(Js),)
    # Average over the n_sims axis (axis=-1)
    cnfg_entp_avg = np.mean(cnfg_entp, axis=-1)
    mean_sum_ss_avg = np.mean(mean_sum_ss, axis=-1)
    
    dW_dJ_avg = mean_sum_ss_avg / (L**2)
    dS_dJ_avg = np.gradient(cnfg_entp_avg, Js)
    
    eta_num = -dS_dJ_avg
    eta_den = dW_dJ_avg
    
    mask_avg = eta_den > DENO_EPS
    eta = np.where(mask_avg, eta_num / eta_den, np.nan)
    
    return eta, eta_num, eta_den, eta_per_sim, eta_num_per_sim, eta_den_per_sim

def compute_eta_inferential(data, e0_idx=0):
    """
    Compute the thermodynamic efficiency using the covariance form.
    Assumes the variable is always J.
    """
    L = data['L']
    Js = np.array(data['Js'])
    h = data['h'] # Constant h
    n_samples = data['n_samples'] # Constant n_samples each simulation
    
    # Extract the 2D slice for the specific E0: Shape becomes (len(Js), n_sims)
    cov_s_ss = np.array(data['cov_s_ss'])[e0_idx]
    var_sum_ss = np.array(data['var_sum_ss'])[e0_idx]
    mean_sum_ss = np.array(data['mean_sum_ss'])[e0_idx]
    
    # 1. Compute PER SIMULATION quantities
    # Js[:, None] broadcasts the 1D J array against the 2D variance array
    eta_num_per_sim = (h * cov_s_ss + Js[:, None] * var_sum_ss)
    eta_den_per_sim = mean_sum_ss
    
    mask_per_sim = eta_den_per_sim > DENO_EPS
    eta_per_sim = np.where(mask_per_sim, eta_num_per_sim / eta_den_per_sim, np.nan)
    

    # 2. Compute AVERAGED quantities
    # Average over the n_sims axis (axis=-1)
    cov_s_ss_avg = np.mean(cov_s_ss, axis=-1)
    var_sum_ss_avg = np.zeros_like(cov_s_ss_avg)
    for i in range(cov_s_ss_avg.shape[0]):
        var_sum_ss_avg[i] = pool_variance(mean_sum_ss[i], var_sum_ss[i], n_samples)
    mean_sum_ss_avg = np.mean(mean_sum_ss, axis=-1)
    
    eta_num = (h * cov_s_ss_avg + Js * var_sum_ss_avg)
    eta_den = mean_sum_ss_avg
    
    mask_avg = eta_den/(L*L) > DENO_EPS # avoid dividing by really small numbers
    eta = np.where(mask_avg, eta_num / eta_den, np.nan)
    
    return eta, eta_num, eta_den, eta_per_sim, eta_num_per_sim, eta_den_per_sim