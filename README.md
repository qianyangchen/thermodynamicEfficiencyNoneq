# thermodynamicEfficiencyNoneq

Simulation codes for computing the thermodynamic efficiency of two nonequilibrium spin models: the **persistent Ising model** and the **active Ising model**. This repository accompanies a manuscript submitted to *Physical Review Research*.

---

## Models

### Persistent Ising Model
A nonequilibrium Ising model in which detailed balance is broken by introducing a constant bias to the spin-flip dynamics.

> Reference: M. Kumar and C. Dasgupta, *Nonequilibrium phase transition in an Ising model without detailed balance*, Phys. Rev. E **102**, 052111 (2020).

### Active Ising Model
A two-dimensional active-spin model combining Ising-like alignment with self-propulsion, exhibiting a flocking transition.

> References:
> - A. P. Solon and J. Tailleur, *Revisiting the Flocking Transition Using Active Spins*, Phys. Rev. Lett. **111**, 078101 (2013).
> - A. P. Solon and J. Tailleur, *Flocking with discrete symmetry: The two-dimensional active Ising model*, Phys. Rev. E **92**, 042119 (2015).

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

- Python 3.x
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### Running Simulations

All core functions are contained in pim_core.py and aim_core.py for the persistent Ising model and active Ising model, respectively. Example usage of the core functions are shown in pim_sim.py and aim_sim.py. The data generated for this manuscript is run in batches on the National Computational Infrastructure (NCI):
```bash
cd persistent_ising/simulation
python pim_sim.py
```
-->

### Generating Figures

Run the notebook to load and visualise sample data:
```bash
cd persistent_ising/notebooks
jupyter notebook aim.ipynb
```


---

## Citation

<!-- TODO: add the full citation for manuscript once it is published. -->

If you use this code, please cite:

> [Author(s)], *[Title]*, *Physical Review Research* (submitted).

and the relevant model references listed above.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
