# thermodynamicEfficiencyNoneq

Simulation codes for computing the thermodynamic efficiency of two nonequilibrium spin models: the **persistent Ising model** and the **active Ising model**. This repository accompanies the manuscript: Q. Chen and M. Prokopenko, *Thermodynamic efficiency of self-organisation in nonequilibrium steady states* 	[arXiv:2605.04508](https://arxiv.org/abs/2605.04508) (2026).

---

## Models

### Persistent Ising Model
A nonequilibrium Ising model in which detailed balance is broken by introducing a constant bias to the spin-flip dynamics.

> Reference: 
> - [1] M. Kumar and C. Dasgupta, *Nonequilibrium phase transition in an Ising model without detailed balance*, Phys. Rev. E **102**, 052111 (2020).

### Active Ising Model
A two-dimensional active-spin model combining Ising-like alignment with self-propulsion, exhibiting a flocking transition.

> References:
> - [2] A. P. Solon and J. Tailleur, *Revisiting the Flocking Transition Using Active Spins*, Phys. Rev. Lett. **111**, 078101 (2013).
> - [3] A. P. Solon and J. Tailleur, *Flocking with discrete symmetry: The two-dimensional active Ising model*, Phys. Rev. E **92**, 042119 (2015).

---

## Repository Structure

```
thermodynamicEfficiencyNoneq/
├── persistent_ising/
│   ├── simulation/        # Python scripts for running persistent Ising simulations
│   ├── data/              # Output data from simulations
│   └── notebooks/         # Jupyter notebooks for data analysis and figures
├── active_ising/
│   ├── simulation/        # Python scripts for running active Ising simulations
│   ├── data/              # Output data from simulations
│   └── notebooks/         # Jupyter notebooks for data analysis and figures
├── requirements.txt       # Python package dependencies
└── README.md
```

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Running Simulations
All core simulation functions are contained in pim_core.py (for the persistent Ising model [1]) and aim_core.py (for the active Ising model[2,3]).

**Local Examples:**
Example usage of the core functions is shown in pim_sim.py and aim_sim.py. You can run these locally to test the simulations:

persistent Ising model:

```bash
cd persistent_ising/simulation
python pim_sim.py
```

active Ising model:
```bash
cd active_ising/simulation
python aim_sim.py
```

**HPC / Cluster Execution:**
Please note that the data generated for this manuscript was run on the National Computational Infrastructure (NCI).

### 2. Generating Figures & Data Analysis
We provide Jupyter notebooks to load, analyse, and visualise the sample data.

For the persistent Ising Model, run:

```bash
cd persistent_ising/notebooks
jupyter notebook pim.ipynb
```

For the active Ising Model, run:

```bash
cd active_ising/notebooks
jupyter notebook activeIsing.ipynb
```

---

## Citing


If you use this code, please cite:

> Qianyang Chen and Mikhail Prokopenko, *Thermodynamic efficiency of self-organisation in nonequilibrium steady states*, arXiv:2605.04508 [nlin.AO] (2026).
Available at: [https://arxiv.org/abs/2605.04508](https://arxiv.org/abs/2605.04508)

**Bibtex**

```bibtex
@article{chen2026therm,
  author  = {Chen, Qianyang and Prokopenko, Mikhail},
  title   = {Thermodynamic efficiency of self-organisation in nonequilibrium steady states},
  journal = {arXiv preprint arXiv:2605.04508},	
  year    = {2026},
  eprint  = {2605.04508},
  archivePrefix = {arXiv},
  primaryClass = {nlin.AO}
}
```
