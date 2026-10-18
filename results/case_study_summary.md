# PD–SPH case-study result summary

**Source:** *Peridynamics–Smoothed Particle Hydrodynamics Coupling for Fluid–Structure Interaction*, Computational Engineering case-study report, Ruhr University Bochum, August 2026, Chapter 3 (verification and results). These are **reported simulation results**, not re-executions of the current GitHub scripts.

## Case 1: dam-break impact on a bottom-clamped elastic plate

- Grid spacing: **4.0 mm**, compared with **2.5 mm** in the referenced simulation.
- Free-tip horizontal displacement peak: **4.74 × 10⁻² m** at approximately **0.25 s**.
- Published comparison range cited in the report: **4.2–4.8 × 10⁻² m** at approximately the same time.
- The report describes a roughly **9% larger peak** relative to Huang et al.'s PD–SPH prediction, while capturing subsequent elastic recovery and oscillation.

## Case 2: flow through a top-clamped plate

- Grid spacing: **1.25 mm**.
- Maximum leftward free-tip displacement magnitude: approximately **5.1 × 10⁻² m** at **0.16 s**.
- Reference PD–SPH result: approximately **5.2 × 10⁻² m** at **0.16 s**.
- The report notes an early fluctuation around 0.037–0.047 s absent in the smoother reference solution.

## Case 3: notched plate with tensile energy-based damage

- Grid spacing: **0.75 mm**; input fracture energy **10 J/m²**.
- Initially disconnected bonds due to the notch: **68**.
- First additional bond failure: approximately **0.0756 s**.
- Fully broken bonds reach approximately **238** by **0.0964 s**, remaining at that value through the end of the reported interval.
- Reported final horizontal tip displacement: approximately **−0.039 m** at **0.12 s**.
- The report explicitly treats this as a **fracture demonstration**, not a separately validated fracture benchmark; a ligament remains at the end.

## Interpreting the evidence

Case 1 and Case 2 are quantitative elastic-response comparisons. Case 3 illustrates the capability of the fracture extension. These should not be described as three independently validated benchmarks.

The original report contains tip-displacement figures and flow/damage snapshots (Chapter 3, Figures 3.1–3.7). Those images are **not present in this GitHub repository**, and no substitute curves or synthetic simulation snapshots have been created.
