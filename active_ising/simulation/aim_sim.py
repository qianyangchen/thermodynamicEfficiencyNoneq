import numpy as np
import os
import time
import datetime

from aim_core import run_multiple_ness, generate_simulation_seeds

def shard_indices(indices, world_size, rank):
    """
    Split a list of indices across ranks as evenly as possible.
    """
    indices = list(indices)
    n = len(indices)
    counts = [n // world_size + (1 if r < (n % world_size) else 0) for r in range(world_size)]
    starts = np.cumsum([0] + counts[:-1])
    return indices[starts[rank]: starts[rank] + counts[rank]]


def get_rank_info():
    """
    Support OpenMPI / mpirun style environment variables.
    Fallback to single-rank mode for local testing.
    """
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", 0)))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", 1)))
    return rank, world_size


def shard_filename(output_dir, rank):
    return os.path.join(output_dir, f"aim_ness_rank{rank:03d}.npz")


if __name__ == "__main__":

    # --- 1. System & Run Parameters ---
    output_dir = os.environ.get("OUT_DIR", os.environ.get("PBS_O_WORKDIR", "."))

    # Read dynamic parameters from bash environment
    h = float(os.environ.get("H_VAL", 0.0))
    epsilon = float(os.environ.get("EPS_VAL", 1.0))
    lat_x = int(os.environ.get("L_VAL", 100))
    lat_y = int(os.environ.get("L_VAL", 100)) * int(os.environ.get("Y_TO_X_VAL", 1))
    D = float(os.environ.get("D_VAL", 1.0))
    rho0 = float(os.environ.get("RHO0_VAL", 3.0))
    beta = float(os.environ.get("BETA_VAL", 1.0))
    batch = int(os.environ.get("BATCH", 1))
    n_relax_sweeps = int(os.environ.get("N_RELX", 2_000_000))

    # Only J changes
    J_list = np.linspace(0.5, 2.5, 41)

    Lx, Ly = lat_x, lat_y # 200, 200
    n_sample_sweeps = 5000
    max_particles_per_cell = 150
    n_sims = 48

    # Use one node only. Cap at n_sims so never spawn more workers than tasks.
    requested_jobs = int(os.environ.get("AIM_N_JOBS", os.environ.get("PBS_NCPUS", 1)))
    n_jobs = min(requested_jobs, n_sims)

    # Offset the master seed by the batch number so Batch 2 generates completely new paths
    master_seed = int(os.environ.get("AIM_MASTER_SEED", 12345)) + (batch * 1000)

    # Get rank from the PBS array, fallback to MPI logic if testing locally
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))
    
    my_j_indices = shard_indices(range(len(J_list)), world_size, rank)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting AIM NESS rank-shard job")
    print(f"rank={rank}, world_size={world_size}, n_jobs={n_jobs}, n_sims={n_sims}")
    print(f"my_j_indices={my_j_indices}")

    shard_file = shard_filename(output_dir, rank)

    # Resume-friendly: skip if this rank shard already exists
    if os.path.exists(shard_file):
        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"Shard already exists, skipping: {os.path.basename(shard_file)}"
        )
        raise SystemExit(0)

    start_time = time.time()

    # Save a meta file so later stitching is easy and reproducible
    meta_file = os.path.join(output_dir, "aim_ness_meta.npz")
    if not os.path.exists(meta_file):
        np.savez_compressed(
            meta_file,
            Lx=Lx, Ly=Ly,
            rho0=rho0, beta=beta, epsilon=epsilon, D=D, h=h,
            J_list=np.array(J_list, dtype=np.float64),
            n_relax_sweeps=n_relax_sweeps,
            n_sample_sweeps=n_sample_sweeps,
            max_particles_per_cell=max_particles_per_cell,
            n_sims=n_sims,
            master_seed=master_seed
        )

    # --- 2. Allocate arrays ---
    local_n_J = len(my_j_indices)

    entropies_local = np.zeros((local_n_J, n_sims), dtype=np.float64)
    energies_local = np.zeros((local_n_J, n_sims), dtype=np.float64)
    E_Js_local = np.zeros((local_n_J, n_sims, n_sample_sweeps), dtype=np.float64)
    E_hs_local = np.zeros((local_n_J, n_sims, n_sample_sweeps), dtype=np.float64)
    seeds_local = np.zeros((local_n_J, n_sims), dtype=np.int64)
    
    # For the 1D profiles along Lx
    mean_rho_x_local = np.zeros((local_n_J, n_sims, Lx), dtype=np.float64)
    mean_m_x_local = np.zeros((local_n_J, n_sims, Lx), dtype=np.float64)
    final_rho_x_local = np.zeros((local_n_J, n_sims, Lx), dtype=np.float64)
    final_m_x_local = np.zeros((local_n_J, n_sims, Lx), dtype=np.float64)

    # For entropy production rate
    entp_prod_rate_local = np.zeros((local_n_J, n_sims), dtype=np.float64)

    j_indices_local = np.array(my_j_indices, dtype=np.int64)
    J_values_local = np.zeros(local_n_J, dtype=np.float64)

    # --- 3. Loop over this rank's assigned J values ---
    for local_idx, j_idx in enumerate(my_j_indices):
        J = float(J_list[j_idx])
        J_values_local[local_idx] = J

        print(
            f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
            f"rank={rank}: running global j_idx={j_idx}, J={J:.6f} "
            f"({local_idx+1}/{local_n_J} on this rank)"
        )

        params = {
            'Lx': Lx,
            'Ly': Ly,
            'rho0': rho0,
            'beta': beta,
            'epsilon': epsilon,
            'D': D,
            'J': J,
            'h': h
        }

        # Give each J-row its own reproducible seed family
        row_master_seed = master_seed + j_idx
        row_seeds = generate_simulation_seeds(n_sims, master_seed=row_master_seed)
        seeds_local[local_idx, :] = row_seeds

        # Use the newly defined parallel wrapper
        results = run_multiple_ness(
            params,
            n_relax_sweeps,
            n_sample_sweeps,
            max_particles_per_cell,
            n_sims=n_sims,
            n_jobs=n_jobs,
            master_seed=row_master_seed
        )

        for sim_idx, (entropy, energy, EJ_list, Eh_list, m_rho, m_m, f_rho, f_m, epr) in enumerate(results):
            entropies_local[local_idx, sim_idx] = entropy
            energies_local[local_idx, sim_idx] = energy
            E_Js_local[local_idx, sim_idx, :] = np.asarray(EJ_list, dtype=np.float64)
            E_hs_local[local_idx, sim_idx, :] = np.asarray(Eh_list, dtype=np.float64)
            entp_prod_rate_local[local_idx, sim_idx] = epr

            # Record the new profile data
            mean_rho_x_local[local_idx, sim_idx, :] = m_rho
            mean_m_x_local[local_idx, sim_idx, :] = m_m
            final_rho_x_local[local_idx, sim_idx, :] = f_rho
            final_m_x_local[local_idx, sim_idx, :] = f_m

    # --- 4. Save one file per rank ---
    np.savez_compressed(
        shard_file,
        rank=rank,
        world_size=world_size,
        j_indices=j_indices_local,
        J_values=J_values_local,
        entropies=entropies_local,
        energies=energies_local,
        E_Js=E_Js_local,
        E_hs=E_hs_local,
        mean_rho_x=mean_rho_x_local,
        mean_m_x=mean_m_x_local,
        final_rho_x=final_rho_x_local,
        final_m_x=final_m_x_local,
        entp_prod_rate=entp_prod_rate_local,
        seeds=seeds_local,
        Lx=Lx, Ly=Ly,
        rho0=rho0, beta=beta, epsilon=epsilon, D=D, h=h,
        n_relax_sweeps=n_relax_sweeps,
        n_sample_sweeps=n_sample_sweeps,
        max_particles_per_cell=max_particles_per_cell,
        n_sims=n_sims,
        master_seed=master_seed
    )

    end_time = time.time()
    duration = (end_time - start_time) / 60.0

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Finished rank {rank}")
    print(f"Saved {os.path.basename(shard_file)}")
    print(f"Elapsed time: {duration:.2f} minutes")