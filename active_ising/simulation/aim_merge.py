import numpy as np
import os
import glob
import re

SHARD_PATTERN = re.compile(r"aim_ness_rank(\d{3})\.npz")


def parse_rank_file(filename):
    base = os.path.basename(filename)
    match = SHARD_PATTERN.fullmatch(base)
    if match is None:
        raise ValueError(f"Unrecognized shard filename format: {base}")
    return int(match.group(1))


if __name__ == "__main__":

    output_dir = os.environ.get("OUT_DIR", os.environ.get("PBS_O_WORKDIR", "."))
    meta_file = os.path.join(output_dir, "aim_ness_meta.npz")

    if not os.path.exists(meta_file):
        raise FileNotFoundError(
            f"meta not found: {meta_file}\n"
            "Run aim_ness.py first."
        )

    meta = np.load(meta_file)

    Lx = int(meta["Lx"])
    Ly = int(meta["Ly"])
    rho0 = float(meta["rho0"])
    beta = float(meta["beta"])
    epsilon = float(meta["epsilon"])
    D = float(meta["D"])
    h = float(meta["h"])
    J_list = meta["J_list"]
    n_relax_sweeps = int(meta["n_relax_sweeps"])
    n_sample_sweeps = int(meta["n_sample_sweeps"])
    max_particles_per_cell = int(meta["max_particles_per_cell"])
    n_sims = int(meta["n_sims"])
    master_seed = int(meta["master_seed"])

    n_J = len(J_list)

    shard_files = glob.glob(os.path.join(output_dir, "aim_ness_rank*.npz"))
    if len(shard_files) == 0:
        raise FileNotFoundError(
            f"No shard files found in {output_dir}. "
            "Expected files like aim_ness_rank000.npz"
        )

    shard_files = sorted(shard_files, key=parse_rank_file)

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
    completed_rows = np.zeros(n_J, dtype=bool)

    for shard_file in shard_files:
        data = np.load(shard_file)

        j_indices = data["j_indices"]
        J_values = data["J_values"]
        entropies_local = data["entropies"]
        energies_local = data["energies"]
        E_Js_local = data["E_Js"]
        E_hs_local = data["E_hs"]
        seeds_local = data["seeds"]
        mean_rho_x_local = data["mean_rho_x"]
        mean_m_x_local = data["mean_m_x"]
        final_rho_x_local = data["final_rho_x"]
        final_m_x_local = data["final_m_x"]
        entp_prod_rate_local = data["entp_prod_rate"]

        local_n_J = len(j_indices)

        if entropies_local.shape != (local_n_J, n_sims):
            raise ValueError(f"Bad entropies shape in {shard_file}: {entropies_local.shape}, expected ({local_n_J}, {n_sims})")
        if energies_local.shape != (local_n_J, n_sims):
            raise ValueError(f"Bad energies shape in {shard_file}: {energies_local.shape}, expected ({local_n_J}, {n_sims})")
        if E_Js_local.shape != (local_n_J, n_sims, n_sample_sweeps):
            raise ValueError(f"Bad E_Js shape in {shard_file}: {E_Js_local.shape}, expected ({local_n_J}, {n_sims}, {n_sample_sweeps})")
        if E_hs_local.shape != (local_n_J, n_sims, n_sample_sweeps):
            raise ValueError(f"Bad E_hs shape in {shard_file}: {E_hs_local.shape}, expected ({local_n_J}, {n_sims}, {n_sample_sweeps})")
        if seeds_local.shape != (local_n_J, n_sims):
            raise ValueError(f"Bad seeds shape in {shard_file}: {seeds_local.shape}, expected ({local_n_J}, {n_sims})")
        if mean_rho_x_local.shape != (local_n_J, n_sims, Lx):
            raise ValueError(f"Bad mean_rho_x shape in {shard_file}: {mean_rho_x_local.shape}, expected ({local_n_J}, {n_sims}, {Lx})")
        if mean_m_x_local.shape != (local_n_J, n_sims, Lx):
            raise ValueError(f"Bad mean_m_x shape in {shard_file}: {mean_m_x_local.shape}, expected ({local_n_J}, {n_sims}, {Lx})")
        if final_rho_x_local.shape != (local_n_J, n_sims, Lx):
            raise ValueError(f"Bad final_rho_x shape in {shard_file}: {final_rho_x_local.shape}, expected ({local_n_J}, {n_sims}, {Lx})")
        if final_m_x_local.shape != (local_n_J, n_sims, Lx):
            raise ValueError(f"Bad final_m_x shape in {shard_file}: {final_m_x_local.shape}, expected ({local_n_J}, {n_sims}, {Lx})")
        if entp_prod_rate_local.shape != (local_n_J, n_sims):
            raise ValueError(f"Bad entp_prod_rate shape in {shard_file}: {entp_prod_rate_local.shape}, expected ({local_n_J}, {n_sims})")


        for local_idx, j_idx in enumerate(j_indices):
            j_idx = int(j_idx)
            J = float(J_values[local_idx])

            if j_idx < 0 or j_idx >= n_J:
                raise ValueError(f"Row index out of range in {shard_file}: {j_idx}")

            if not np.isclose(J, J_list[j_idx]):
                raise ValueError(
                    f"J mismatch in {shard_file}: file has J={J}, "
                    f"but meta expects J_list[{j_idx}]={J_list[j_idx]}"
                )

            if completed_rows[j_idx]:
                raise RuntimeError(f"Duplicate data for j_idx={j_idx} found in {shard_file}")

            entropies[j_idx, :] = entropies_local[local_idx, :]
            energies[j_idx, :] = energies_local[local_idx, :]
            E_Js[j_idx, :, :] = E_Js_local[local_idx, :, :]
            E_hs[j_idx, :, :] = E_hs_local[local_idx, :, :]
            all_seeds[j_idx, :] = seeds_local[local_idx, :]
            mean_rho_x[j_idx, :, :] = mean_rho_x_local[local_idx, :, :]
            mean_m_x[j_idx, :, :] = mean_m_x_local[local_idx, :, :]
            final_rho_x[j_idx, :, :] = final_rho_x_local[local_idx, :, :]
            final_m_x[j_idx, :, :] = final_m_x_local[local_idx, :, :]
            entp_prod_rate[j_idx, :] = entp_prod_rate_local[local_idx, :]
            completed_rows[j_idx] = True

    if not np.all(completed_rows):
        missing = np.where(~completed_rows)[0]
        raise RuntimeError(
            f"Cannot stitch: missing data for j_idx = {missing.tolist()}"
        )

    compiled_file = os.path.join(
        output_dir,
        f"aim_ness_L{Lx}x{Ly}_r{rho0}_b{beta}_e{epsilon}_D{D}.npz"
    )

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

    print(f"Compiled file saved to {compiled_file}")
    print(f"entropies shape = {entropies.shape}")
    print(f"energies shape  = {energies.shape}")
    print(f"E_Js shape      = {E_Js.shape}")
    print(f"E_hs shape      = {E_hs.shape}")
    print(f"entp_prod_rate shape = {entp_prod_rate.shape}")