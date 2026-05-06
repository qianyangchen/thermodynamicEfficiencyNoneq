#!/usr/bin/env python
import numpy as np
from numba import njit
from joblib import Parallel, delayed
from collections import defaultdict # to compute pdf
import itertools # for entropy production and fisher information

# Maximum number of steps to keep for truncated output.
MAX_STEPS = 200000
EPSILON = 1e-6 # to avoid log(p) getting too large due to random fluctuation
DENO_EPS = 0.05 # to avoid division by zero

#--------------------------------------------------------------------------
# Low-level (Numba-accelerated) functions
#--------------------------------------------------------------------------

@njit(nogil=True)
def initialise(L, bias):
    """
    Initialize an LxL lattice with values in {1, -1}.
    Each site is set to 1 with probability `bias`, otherwise -1.
    Uses vectorized operations.
    """
    rand_matrix = np.random.rand(L, L)
    lattice = np.where(rand_matrix < bias, 1, -1)
    return lattice

@njit(nogil=True)
def get_mu(lattice):
    """
    Compute the net interaction energy (mu) using periodic boundary conditions.
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
def metropolis(lattice, time, J, temperature, mu, E0, h, sampleSize, truncate):
    """
    Run the Metropolis algorithm for a given number of time steps in non-equilibrium ising model.
    
    Parameters:
      lattice : 2D array of int64
          The initial lattice.
      time : int
          Total number of time steps.
      J : float
          Coupling strength.
      temperature : float
          Temperature of the system.
      mu : float
          Initial net interaction energy.
      E0 : 1D array of float
          Additional energy offset for each time step. If constant pass np.full(time, E0).
      h : 1D array of float
            External field at each time step.
      sampleSize : int
          Number of samples to keep.
      truncate : bool
          If True, only the last min(time, sampleSize) observations are kept.

    Returns:
      magnetisations, mus
      Each is a 1D array of floats containing lattice magnetisation and net energy (mu) respectively.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if time <= 0:
        raise ValueError("Time must be greater than 0.")
    beta = 1.0 / temperature
    L = lattice.shape[0]
    if E0 is None:
        E0 = np.zeros(time, dtype=np.float64)
    if h is None:
        h = np.zeros(time, dtype=np.float64)
    
    if truncate:
        # truncate=True returns the last min(time, sampleSize) observations
        sample = min(time, sampleSize)
        magnetisations = np.zeros(sample)
        mus = np.zeros(sample)
        spins_i = np.zeros(sample)
        spins_f = np.zeros(sample)
        sum_neighbors = np.zeros(sample)
    else:
        # keep results for every time step.
        magnetisations = np.zeros(time)
        mus = np.zeros(time)
        spins_i = np.zeros(time)
        spins_f = np.zeros(time)
        sum_neighbors = np.zeros(time)

    for t in range(time):
        x = np.random.randint(L)
        y = np.random.randint(L)
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
        dE = J * (mu_f - mu_i) + h[t] * (-spin_f + spin_i) #negative sign included in mu_i and mu_f
        dE_eff = dE + E0[t]
        if dE_eff < 0 or np.random.rand() < np.exp(-beta * dE_eff):
            # flip spin
            lattice[x, y] = spin_f
            mu += (mu_f - mu_i)
        if truncate and t >= time - sample:
            idx = t - (time - sample)
            mus[idx] = mu
            magnetisations[idx] = lattice.sum() / lattice.size
            spins_i[idx] = spin_i
            spins_f[idx] = lattice[x, y] # may or may not have flipped
            sum_neighbors[idx] = sum_nb
        elif truncate and t < time - sample:
            # do not store the first time - sample steps
            pass
        else:
            mus[t] = mu
            magnetisations[t] = lattice.sum() / lattice.size
            spins_i[t] = spin_i
            spins_f[t] = lattice[x, y] # may or may not have flipped
            sum_neighbors[t] = sum_nb
    return magnetisations, mus, spins_i, spins_f, sum_neighbors

@njit(nogil=True)
def glauber(lattice, time, J, temperature, mu, E0, h, sampleSize,truncate):
    """
    Run the Glauber algorithm for a given number of time steps in non-equilibrium ising model.
    
    Parameters:
      lattice : 2D array of int64
          The initial lattice.
      time : int
          Total number of time steps.
      J : float
          Coupling strength.
      temperature : float
          Temperature of the system.
      mu : float
          Initial net interaction energy.
      E0 : 1D array of float
          Additional energy offset for each time step. If constant pass np.full(time, E0).
      h : 1D array of float
            External field at each time step.
      sampleSize : int
          Number of samples to keep.
      truncate : bool
          If True, only the last min(time, sampleSize) observations are kept.
    
    Returns:
      magnetisations, mus
      Each is a 1D array of floats containing lattice magnetisation and net energy (mu) respectively.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if time <= 0:
        raise ValueError("Time must be greater than 0.")
    beta = 1.0 / temperature
    L = lattice.shape[0]
    if E0 is None:
        E0 = np.zeros(time, dtype=np.float64)
    if h is None:
        h = np.zeros(time, dtype=np.float64)
    
    if truncate:
        # truncate=True returns the last min(time, sampleSize) observations
        sample = min(time, sampleSize)
        magnetisations = np.zeros(sample)
        mus = np.zeros(sample)
        spins_i = np.zeros(sample)
        spins_f = np.zeros(sample)
        sum_neighbors = np.zeros(sample)
    else:
        # keep results for every time step.
        magnetisations = np.zeros(time)
        mus = np.zeros(time)
        spins_i = np.zeros(time)
        spins_f = np.zeros(time)
        sum_neighbors = np.zeros(time)

    for t in range(time):
        x = np.random.randint(L)
        y = np.random.randint(L)
        spin_i = lattice[x, y]
        spin_f = -spin_i
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
        dE = J * (mu_f - mu_i) + h[t] * (-spin_f + spin_i) #negative sign included in mu_i and mu_f
        dE_eff = dE + E0[t]
        if np.random.rand() < 1.0/(1.0 + np.exp(beta * dE_eff)):
            # flip spin
            lattice[x, y] = spin_f
            mu += (mu_f - mu_i)
        if truncate and t >= time - sample:
            idx = t - (time - sample)
            mus[idx] = mu
            magnetisations[idx] = lattice.sum() / lattice.size
            spins_i[idx] = spin_i
            spins_f[idx] = lattice[x, y] # may or may not have flipped
            sum_neighbors[idx] = sum_nb
        elif truncate and t < time - sample:
            # do not store the first time - sample steps
            pass
        else:
            mus[t] = mu
            magnetisations[t] = lattice.sum() / lattice.size
            spins_i[t] = spin_i
            spins_f[t] = lattice[x, y] # may or may not have flipped
            sum_neighbors[t] = sum_nb
    return magnetisations, mus, spins_i, spins_f, sum_neighbors

#--------------------------------------------------------------------------
# High-level simulation driver functions
#--------------------------------------------------------------------------

def run_single_simulation(L, steps, J, bias=0.5, temperature=1.0, E0=None, h=None, algorithm='Metropolis', truncate=True, time_series=False, sampleSize=MAX_STEPS):
    """ 
    Run a single simulation of the chosen algorithm.
    
    Parameters:
      L (int): Lattice size (LxL).
      steps (int): Number of time steps to run the simulation.
      J (float): Coupling strength.
      sampleSize (int): Number of final steps to return.
      bias (float): Bias for lattice initialization.
      temperature (float): Temperature of the simulation.
      E0 (1D array of float): Energy offset per time step.
      algorithm (str): 'Metropolis' or 'Glauber'.
    
    Returns:
      tuple: (magnetisations_last, lattice)
             Each array is of length sampleSize.
    """
    lattice = initialise(L, bias=bias)
    mu_initial = get_mu(lattice)
    if E0 is not None:
        if np.isscalar(E0):
            E0_arr = np.full(steps, E0, dtype=np.float64)
        elif len(E0) != steps:
            print("Warning: E0 array length does not match the number of steps. Using constant E0[0].")
            E0_arr = np.full(steps, E0[0], dtype=np.float64)
        else:
            E0_arr = np.asarray(E0, dtype=np.float64)
    else:
        E0_arr = E0
    if h is not None:
        if np.isscalar(h):
            h_arr = np.full(steps, h, dtype=np.float64)
        elif len(h) != steps:
            print("Warning: h array length does not match the number of steps. Using constant h[0].")
            h_arr = np.full(steps, h[0], dtype=np.float64)
        else:
            h_arr = np.asarray(h, dtype=np.float64)
    else:
        h_arr = h
    if time_series:
        if algorithm == 'Glauber':
            magnetisations, mus, spins_i, spins_f, sum_neighbors = glauber(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, sampleSize=sampleSize, truncate=truncate)
        else:
            magnetisations, mus, spins_i, spins_f, sum_neighbors = metropolis(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, sampleSize=sampleSize, truncate=truncate)
        return magnetisations, mus, lattice, spins_i, spins_f, sum_neighbors
    else:
        if algorithm == 'Glauber':
            magnetisations, mus, _, _, _ = glauber(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, sampleSize=sampleSize, truncate=truncate)
        else:
            magnetisations, mus, _, _, _ = metropolis(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, sampleSize=sampleSize, truncate=truncate)
        return magnetisations, mus, lattice

def run_multi_simulation(L, steps, J, numSims, bias=0.5, temperature=1.0, E0=None, h=None, algorithm='Metropolis', truncate=True, time_series=False, sampleSize=MAX_STEPS, n_jobs=-1):
    """
    Run multiple single simulations in parallel using joblib.
    
    Parameters:
      L (int): Lattice size.
      steps (int): Number of time steps per simulation.
      J (float): Coupling strength.
      numSims (int): Number of simulations.
      bias (float): Lattice initialization bias.
      temperature (float): Temperature of the simulation.
      E0 (1D array of float): Energy offset per time step.
      algorithm (str): 'Metropolis' or 'Glauber'.
      n_jobs (int): Number of parallel jobs (-1 uses all cores).
      
    Returns:
      List of numSims tuples (magnetisation, sum_interaction, lattice), each tuple contains the results of a single simulation. magnetisation and sum_interaction are numpy arrays of length MAX_STEPS (if truncate=True) or time (if truncate=False), lattice is a numpy array of shape (L, L).
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_simulation)(L, steps, J, bias, temperature, E0, h, algorithm, truncate, time_series, sampleSize=sampleSize) for _ in range(numSims)
    )
    return results

#--------------------------------------------------------------------------
# High-level analysis functions
#--------------------------------------------------------------------------

# Configuration entropy related ====================================================
def compute_entropy(prob_distribution):
    """
    Compute the entropy of a given probability distribution.
    
    Parameters:
    prob_distribution (dict): A dictionary with probabilities.
    
    Returns:
    float: The entropy value. Uses base-2 logarithm.
    """
    entropy = -sum(p * np.log2(p) for p in prob_distribution.values() if p > EPSILON)
    return entropy

def compute_probability_distribution(lattice, n, m):
    """
    Compute the probability distribution of configurations in a given nxm area
    in a wrapped around square lattice. Modified from v3 to include all possible configurations.
    
    Parameters:
    lattice (np.ndarray): A LxL numpy array with values -1 or 1.
    n (int): The number of rows in the area.
    m (int): The number of columns in the area.
    
    Returns:
    dict: A dictionary with configurations as keys and their probabilities as values.
    """
    L = lattice.shape[0]
    count_dict = defaultdict(int)

    # Iterate over all possible configurations in the sub-lattice
    for config in itertools.product([-1, 1], repeat=n*m):
        config = tuple(tuple(config[x*m:(x+1)*m]) for x in range(n))
        count_dict[config] = 0  # initialize all possible configurations
    
    for i in range(L):
        for j in range(L):
            # Extract the nxm sub-lattice starting at (i, j)
            sub_lattice = tuple(tuple(lattice[(i + x) % L, (j + y) % L] for y in range(m)) for x in range(n))
            count_dict[sub_lattice] += 1

    # Compute the probability distribution
    total_counts = sum(count_dict.values())
    prob_distribution = {k: v / total_counts for k, v in count_dict.items()}
    
    return prob_distribution

def get_entropy_kikuchi(lattice):
    """ 
    Compute configuration entropy using kikuchi approximation S = S1-2*S2+S4.

    Parameters:
    lattice (np.array): A 2D integer array of the LxL squre lattice. Values +/-1.
    
    Returns:
    float: The entropy value.
    """
    # compute kikuchi approx 
    entp1 = compute_entropy(compute_probability_distribution(lattice, 1, 1))
    entp2 = compute_entropy(compute_probability_distribution(lattice, 1, 2))
    entp4 = compute_entropy(compute_probability_distribution(lattice, 2, 2))
    return entp1 - 2 * entp2 + entp4

def get_entropy_meanfield(pdf):
    """ 
    Compute configuration entropy using meanfield approximation S = sum(-1,1) -p*log(p).

    Parameters:
    pdf (dict): Proability distribution {spin:p(spin)}. This can be an average distribution computed over 
                a period of time and (or) mutltiple simulations.
    
    Returns:
    float: The entropy value.
    """
    # compute mean-field approx 
    return -sum(p * np.log2(p) for p in pdf.values() if p > EPSILON)

def estimate_conditional_prob(variable, condition):
        joint_counts = defaultdict(int)
        condition_counts = defaultdict(int)

        for v, c in zip(variable, condition):
            joint_counts[(v, c)] += 1
            condition_counts[c] += 1

        conditional_prob = {}
        for (v, c), joint_count in joint_counts.items():
            conditional_prob[(v, c)] = joint_count / condition_counts[c]

        return conditional_prob

def estimate_joint_prob(variable1, variable2, variable3):
        joint_counts = defaultdict(int)

        for v1, v2, v3 in zip(variable1, variable2, variable3):
            joint_counts[(v1, v2, v3)] += 1

        total_count = len(variable1)
        joint_prob = {k: v / total_count for k, v in joint_counts.items()}

        return joint_prob

# Intermediate results ==========================================
def unpack_results(results):
    magnetisations_array = np.array([result[0] for result in results])
    interactions_array = np.array([result[1] for result in results])
    lattice_array = np.array([result[2].astype(int) for result in results])
    return magnetisations_array, interactions_array, lattice_array
 
def get_mean_var(results, L, subSample, absolute_m=True, raw=False):
    magnetisations_array, interactions_array, _= unpack_results(results)
    numSims = len(magnetisations_array) # each element in the array is a list of samples collected for one simulation. Length of the array is the number of simulations.
    if raw:
        # keep each simulation separate
        mean_sum_s  = np.zeros(numSims)
        mean_sum_ss = np.zeros(numSims)
        var_sum_s   = np.zeros(numSims)
        var_sum_ss  = np.zeros(numSims)
        cov_s_ss    = np.zeros(numSims)
        for i in range(numSims):
            # average magnetisation each simulation
            # magnetisation is sampled every subSample time steps because it is highly correlated
            if absolute_m:
                M = abs(magnetisations_array[i,:][::subSample]) * L * L # convert to total magnetisation
            else:
                M = magnetisations_array[i,:][::subSample] * L * L # convert to total magnetisation
            # interaction energy
            R = -interactions_array[i,:][::subSample] # interaction terms includes negative sign from computing Hamiltonian, remove it here
            mean_sum_s[i] = np.mean(M)
            mean_sum_ss[i] = np.mean(R)
            var_sum_s[i] = np.var(M)
            var_sum_ss[i] = np.var(R)
            cov_s_ss[i] = np.cov(M, R)[0, 1]
        return mean_sum_s, mean_sum_ss, var_sum_s, var_sum_ss, cov_s_ss
    else:
        pooled_sum_s = []
        pooled_sum_ss = []
            
        for i in range(numSims):
            # average magnetisation each simulation
            # magnetisation is sampled every subSample time steps because it is highly correlated
            if absolute_m:
                pooled_sum_s.append(abs(magnetisations_array[i][::subSample]))
            else:
                pooled_sum_s.append(magnetisations_array[i][::subSample])
            # interaction energy
            pooled_sum_ss.append(interactions_array[i][::subSample])
        pooled_sum_s = np.array(pooled_sum_s).flatten()
        pooled_sum_ss = np.array(pooled_sum_ss).flatten()
        pooled_sum_s = pooled_sum_s * L * L # convert to total magnetisation
        pooled_sum_ss = -pooled_sum_ss # interaction terms includes negative sign from computing Hamiltonian, remove it here

        # compute mean and variance
        mean_sum_s = np.mean(pooled_sum_s)
        mean_sum_ss = np.mean(pooled_sum_ss)
        var_sum_s = np.var(pooled_sum_s)
        var_sum_ss = np.var(pooled_sum_ss)
        cov_s_ss = np.cov(pooled_sum_s, pooled_sum_ss)[0, 1]
        return mean_sum_s, mean_sum_ss, var_sum_s, var_sum_ss, cov_s_ss

# Efficiencies ==========================================   
def mask_array(input, mask, output):
    return output[eval(f'input{mask}')]
 
def compute_eta_inferential(data, subsample=1, variable='J', mask=None):
    """
    Compute the thermodynamic efficiency using the covariance form for 1D input, e.g. input in the form of (E0, hs) or (E0, Js), and thermodynamic efficiency computed over h or J.

    Parameters:
    mask: string input such as '>0'.
    """
    E0s = np.array(data['E0s']) # dimension 0
    L = data['L']
    if variable == 'J': # each column is a different J
        theta = np.array(data['Js'])[::subsample] 
        other_theta = data['h'] # constant h
        cov = np.array(data['cov_s_ss'])[:, ::subsample] # shape (len(E0s), len(Js))
        mean = np.array(data['mean_sum_ss'])[:, ::subsample] # shape (len(E0s), len(Js)), negative sign incorporated
        var = np.array(data['var_sum_ss'])[:, ::subsample] # shape (len(E0s), len(Js))
    elif variable == 'h': # each column is a different h
        theta = np.array(data['hs'])[::subsample] 
        other_theta = data['J'] # constant J
        cov = np.array(data['cov_s_ss'])[:, ::subsample] # shape (len(E0s), len(hs))
        mean = np.array(data['mean_sum_s'])[:, ::subsample] # shape (len(E0s), len(hs)), negative sign incorporated
        var = np.array(data['var_sum_s'])[:, ::subsample] # shape (len(E0s), len(hs))
    else:
        raise ValueError("variable must be either 'J' or 'h'")

    # thermodynamic efficiency
    numerator_2D = (other_theta*cov + theta[None, :]*var)/(L*L)
    denominator_2D = mean/(L*L)
    
    if mask is not None:
        theta = mask_array(theta, mask, theta)
        for i in range(len(E0s)):
            numerator_2D[i, :] = mask_array(theta, mask, numerator_2D[i, :])
            denominator_2D[i, :] = mask_array(theta, mask, denominator_2D[i, :])
    eta_cov = numerator_2D / denominator_2D  

    # discard points where denominator is too small to avoid numerical instability
    mask_out = denominator_2D > DENO_EPS
    eta_cov = np.where(mask_out, eta_cov, np.nan)
    return eta_cov, theta, numerator_2D, denominator_2D

def compute_eta_first_principle(data, variable='J'):
    """
    Compute the thermodynamic efficiency using the first principle definition for 1D input, e.g. input in the form of (E0, hs) or (E0, Js), and thermodynamic efficiency computed over h or J.

    Parameters:
    mask: string input such as '>0'.
    """
    L = data['L']
    if variable == 'J': # each column is a different J
        theta = np.array(data['Js']) 
        dW_dtheta = np.array(data['mean_sum_ss'])/L**2 # shape (len(E0s), len(Js)), negative sign incorporated
    elif variable == 'h': # each column is a different h
        theta = np.array(data['hs']) 
        dW_dtheta = abs(np.array(data['mean_sum_s']))/L**2 # shape (len(E0s), len(hs)), negative sign incorporated
    else:
        raise ValueError("variable must be either 'J' or 'h'")
    
    cnfg_entp = np.array(data['cnfg_entp']) * np.log(2) # convert to base e

    dW_dtheta = dW_dtheta # normalized work rate
    dS_dtheta = np.gradient(cnfg_entp, theta, axis=1) # entropy rate
    eta = -dS_dtheta / dW_dtheta # thermodynamic efficiency from first principles

    # discard points where denominator is too small to avoid numerical instability
    mask = dW_dtheta > DENO_EPS
    eta = np.where(mask, eta, np.nan)
    return eta