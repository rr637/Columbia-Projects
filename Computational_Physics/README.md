# Computational physics

Problem sets are from PHYSGR-6080 - Scientific Computing. Everything is numpy + matplotlib, and the shared functions (energies, forces, autocorrelation, jackknife, etc.) are in `Sci_comp_helpers.py`.

- **`Poisson_SE_lanczos.ipynb`**: 2D Poisson equation solved with conjugate gradient, then the ground state of a particle in a 2D box with an obstacle using restarted Lanczos. Both only use a function that applies the Laplacian, the matrix is never built.
- **`MD_Liquid_Argon.ipynb`**: molecular dynamics of 864 Lennard-Jones argon atoms with velocity Verlet. Checks energy conservation, the Maxwell-Boltzmann speed distribution, and gets the pressure from the virial.
- **`Monte_Carlo_Liquid_Argon.ipynb`**: Metropolis Monte Carlo of the same system, compared to the MD results.
- **`MD_stat_analysis.ipynb`**: error analysis for correlated data (blocking, autocorrelation time, jackknife), then applied to the MD run.
- **`Ising_Cluster.ipynb`**: 2D Ising model with the Wolff cluster algorithm, measuring the correlation length near the critical point.

The data files that some notebooks load (the MD `.npz` runs and `v1.txt` ... `v5.txt`) are too big to put here, so those cells won't run as is.
