# Coupled Peridynamics–SPH for Fluid–Structure Interaction

A two-dimensional meshfree solver coupling **smoothed particle hydrodynamics (SPH)** for free-surface flow with **non-ordinary state-based peridynamics (PD)** for deformable structures. The project investigates dam-break loading, large plate deformation, and tensile damage/fracture.

This is a computational engineering case-study implementation developed in Python using NumPy, SciPy, JAX and Matplotlib.

## Numerical methods

- **SPH fluid:** weakly compressible flow, Tait equation of state, Wendland C2 kernel, artificial viscosity and density diffusion.
- **PD solid:** correspondence formulation with nonlocal deformation gradient, St. Venant–Kirchhoff elasticity, and damage-consistent hourglass stabilization.
- **Fluid–solid interaction:** shared particle-neighbour interactions transferring pressure and reaction forces between the domains.
- **Damage:** initial notches, tensile-energy-based degradation, broken-bond tracking and shape-tensor rebuilding.
- **Time stepping and diagnostics:** explicit updates, tip displacement histories, particle snapshots and VTK output.

## Three simulation cases

| Script | Problem | Intended use |
| --- | --- | --- |
| [case_01_impacting_plate.py](case_01_impacting_plate.py) | Dam break impacting a bottom-clamped elastic plate | Elastic benchmark |
| [case_02_through_plate.py](case_02_through_plate.py) | Water flowing beneath a top-clamped elastic plate | Elastic benchmark |
| [case_03_notched_dam_break.py](case_03_notched_dam_break.py) | Dam-break loading on a notched plate | Fracture demonstration |
| [pdsph_functions.py](pdsph_functions.py) | Shared SPH, PD, coupling and output routines | Core implementation |

## Results from the accompanying case-study report

The independently documented report compares the first two elastic cases with results presented by Huang et al. The third case demonstrates progressive damage but has **no independent fracture-validation benchmark**.

| Case | Recorded response | Comparison/qualification |
| --- | --- | --- |
| 1 – impacting plate | Peak horizontal tip displacement **0.0474 m** at approximately **0.25 s** | Reference peak range **0.042–0.048 m**; report notes around 9% overprediction relative to the reference PD–SPH curve |
| 2 – through plate | Peak leftward displacement magnitude approximately **0.051 m** at **0.16 s** | Reference PD–SPH peak approximately **0.052 m** at **0.16 s** |
| 3 – notched plate | Additional bond failure from approximately **0.0756 s**; **238 fully broken bonds** by approximately **0.0964 s**, including initial notch bonds | Fracture demonstration only; not an independently validated fracture benchmark |

See [results/case_study_summary.md](results/case_study_summary.md) for provenance, limitations, and additional observations.

## Running a case

Install Python with NumPy, SciPy, JAX, and Matplotlib. Run from the repository root, for example:

```bash
python case_01_impacting_plate.py
```

The case scripts specify particle spacing, fluid/solid parameters, boundary conditions, time step, and output folders.

