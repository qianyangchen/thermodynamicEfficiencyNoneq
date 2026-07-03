import os
import numpy as np
from pim_core import run_multi_ness

if __name__ == '__main__':
    # parallel processing scaffolding
    def shard(lst, w, r):
        n = len(lst)
        base, extra = divmod(n, w)
        start = r * base + min(r, extra)
        end   = start + base + (1 if r < extra else 0)
        return lst[start:end]
    
    # parameters for simulation
    L = int(os.environ.get("L_VAL", 50))
    n_relax_sweeps = int(os.environ.get("N_RELX", 50_000))
    n_samples = 5_000
    n_sims = 48 
    Js = np.linspace(0.01, 2, 40)
    E0s = np.array([-4, -2, 0, 2, 4])
    h = 0.0
    temperature = 1.0
    n_jobs = 48 
    algorithm = 'Glauber'  # Use the constant imported from pim_core
    bias = 1.0

    # parallel processing scaffolding
    cells = [(ei, kj, E0, J) for ei, E0 in enumerate(E0s) for kj, J in enumerate(Js)]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank       = int(os.environ.get("RANK", "0"))
    my_cells = shard(cells, world_size, rank)

    # initialize 3D arrays to hold every simulation
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

    # run simulations for different E0 and J values
    for ei, kj, E0, J in my_cells:
        results = run_multi_ness(
            n_sims=n_sims, L=L, n_relax_sweeps=n_relax_sweeps, 
            n_samples=n_samples, J=J, bias=bias, temperature=temperature, 
            E0=E0, h=h, n_jobs=n_jobs
        )
        
        # unpack the returned list of tuples directly into the 3D grid
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

    # save to shared path
    out_dir = os.environ.get("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, f"PIM_grid_rank{rank:03d}.npz"),
        cnfg_entp_3D=cnfg_entp_3D,
        entp_prod_rate_3D=entp_prod_rate_3D,
        tau_obs_3D=tau_obs_3D,
        mean_sum_s_3D=mean_sum_s_3D,
        mean_sum_ss_3D=mean_sum_ss_3D,
        var_sum_s_3D=var_sum_s_3D,
        var_sum_ss_3D=var_sum_ss_3D,
        cov_s_ss_3D=cov_s_ss_3D,
        binder_cumulant_3D=binder_cumulant_3D,
        # metadata
        h=float(h),
        Js=np.asarray(Js),
        E0s=np.asarray(E0s),
        L=np.int64(L),
        n_relax_sweeps=np.int64(n_relax_sweeps),
        n_samples=np.int64(n_samples),
        n_sims=np.int64(n_sims),
        temperature=float(temperature),
        bias=float(bias),
        algorithm='Glauber'
    )