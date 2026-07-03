import json, glob, os
import numpy as np

# find shards
out_dir = os.environ.get("OUT_DIR", os.environ.get("PBS_O_WORKDIR", "."))
shards = sorted(glob.glob(os.path.join(out_dir, "PIM_grid_rank*.npz")))
if not shards:
    raise SystemExit(f"{out_dir}: No shard files found (PIM_grid_rank*.npz).")

# load first shard to get shapes + metadata
ref = np.load(shards[0], allow_pickle=True)
E0s = ref["E0s"]; Js = ref["Js"]
E, K, n_sims = len(E0s), len(Js), int(ref["n_sims"])

# allocate final 3D arrays (start as NaN)
def nan_array_3D():
    return np.full((E, K, n_sims), np.nan, dtype=float)

cnfg_entp_3D       = nan_array_3D()
entp_prod_rate_3D  = nan_array_3D()
tau_obs_3D         = nan_array_3D()
mean_sum_s_3D      = nan_array_3D()
mean_sum_ss_3D     = nan_array_3D()
var_sum_s_3D       = nan_array_3D()
var_sum_ss_3D      = nan_array_3D()
cov_s_ss_3D        = nan_array_3D()
binder_cumulant_3D = nan_array_3D()

# helper: merge one array from a shard into the final
def merge_into(final_arr, shard_arr):
    mask = ~np.isnan(shard_arr)
    overwrite_mask = mask & ~np.isnan(final_arr)
    if np.any(overwrite_mask):
        raise RuntimeError("Overlap detected between shards.")
    final_arr[mask] = shard_arr[mask]

# loop shards
for path in shards:
    z = np.load(path, allow_pickle=True)
    if not (np.array_equal(z["E0s"], E0s) and np.array_equal(z["Js"], Js)):
        raise RuntimeError(f"Inconsistent E0s/Js in {path}")
    
    merge_into(cnfg_entp_3D, z["cnfg_entp_3D"])
    merge_into(entp_prod_rate_3D, z["entp_prod_rate_3D"])
    merge_into(tau_obs_3D, z["tau_obs_3D"])
    merge_into(mean_sum_s_3D, z["mean_sum_s_3D"])
    merge_into(mean_sum_ss_3D, z["mean_sum_ss_3D"])
    merge_into(var_sum_s_3D, z["var_sum_s_3D"])
    merge_into(var_sum_ss_3D, z["var_sum_ss_3D"])
    merge_into(cov_s_ss_3D, z["cov_s_ss_3D"])
    merge_into(binder_cumulant_3D, z["binder_cumulant_3D"])

# verify completeness
for name, arr in {
    "cnfg_entp": cnfg_entp_3D,
    "entp_prod_rate": entp_prod_rate_3D,
    "tau_obs": tau_obs_3D,
    "mean_sum_s": mean_sum_s_3D,
    "mean_sum_ss": mean_sum_ss_3D,
    "var_sum_s": var_sum_s_3D,
    "var_sum_ss": var_sum_ss_3D,
    "cov_s_ss": cov_s_ss_3D,
    "binder_cumulant": binder_cumulant_3D,
}.items():
    if np.isnan(arr).any():
        missing = np.argwhere(np.isnan(arr))
        raise RuntimeError(f"Missing cells after merge in {name}, e.g. {missing[:5]}")

# pull common metadata from first shard
L = int(ref["L"])
n_relax_sweeps = int(ref["n_relax_sweeps"])
n_samples = int(ref["n_samples"])
temperature = float(ref["temperature"])
bias = float(ref["bias"])
algorithm = 'Glauber'
h = float(ref["h"])

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

    # 3D arrays
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

filename = os.path.join(out_dir, f"PIM_{(n_relax_sweeps+n_samples)//1000}kswps_{n_sims}sim_L{L}.json")
with open(filename, 'w') as f:
    json.dump(combined_data, f)
print(f"Wrote {filename}", flush=True)