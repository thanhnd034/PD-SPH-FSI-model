from pathlib import Path
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import pdsph_functions as ps

# 1. INPUT PARAMETERS

# GEOMETRY
dx = 0.004
thickness = 1.0
domain_width = 0.584
domain_height = 1.168
plate_x0 = 0.292
plate_y0 = 0.0
plate_width = 0.012
plate_height = 0.08
water_x0 = 0.002
water_y0 = 0.002
water_width = 0.146
water_height = 0.292
wall_layers = 4
wall_above_plate = False

# SPH FLUID
rho0_f = 1000.0
gamma = 7.0
c0 = 1.0
gravity = -9.8
alpha_visc = 1.0
beta_visc = 1.0
phi_visc = 0.1
delta_sph = 0.1

# PD SOLID / DAMAGE
rho_s = 2500.0
E_s = 1000000.0
nu_s = 0.0
use_energy_failure = False
fracture_energy_Gc = 0.0
damage_softening_kappa = 0.0
damage_relaxation_time = 0.003
bond_damage_max = 0.8
damage_disconnect_threshold = 0.995
F_singular_min = 0.7
F_singular_max = 1.5
hourglass_coeff = 0.25

# COUPLING / CONTACT
use_fluid_pd_interface_penalty = False
use_pd_fixed_wall_penalty = False
contact_stiffness = 5000.0
contact_damping = 5.0

# TIME / BOUNDARY / OUTPUT
final_time = 1.1
dt = 2e-05
clamp_edge = 'bottom'
tip_edge = 'top'
use_initial_notch = False
notch_depth_fraction = 0.0
notch_offset_rows = 0.0
case_title = 'Dam-break impacting plate'
vtk_dir = 'vtk_case1'
output_prefix = 'case1'
plot_every = 250
log_every = 250
vtk_every = 250
max_sph_pairs = 1000000

# DERIVED QUANTITIES
h = 1.5 * dx
horizon = 3.015 * dx
pd_dV = dx * dx * thickness
pd_mass = rho_s * pd_dV
fluid_mass = rho0_f * dx * dx * thickness
mu_s = E_s / (2.0 * (1.0 + nu_s))
lambda_s = E_s * nu_s / ((1.0 + nu_s) * (1.0 - 2.0 * nu_s))
bulk_s = E_s / (3.0 * (1.0 - 2.0 * nu_s))
B_tait = rho0_f * c0**2
g_vector = np.array([0.0, gravity])
contact_distance = dx
critical_energy_density = (3.0 * fracture_energy_Gc / (2.0 * horizon**3 * thickness)) if fracture_energy_Gc > 0.0 else 0.0

sph = ps.make_sph_parameters(
    h=h,
    c0=c0,
    delta_sph=delta_sph,
    alpha_visc=alpha_visc,
    beta_visc=beta_visc,
    phi_visc=phi_visc,
    contact_distance=contact_distance,
    contact_stiffness=contact_stiffness,
    contact_damping=contact_damping,
    use_fluid_pd_interface_penalty=use_fluid_pd_interface_penalty,
    use_pd_fixed_wall_penalty=use_pd_fixed_wall_penalty,
    g_vector=g_vector,
    max_sph_pairs=max_sph_pairs,
    rho_s=rho_s,
    fluid_mass=fluid_mass,
    pd_mass=pd_mass,
)

# 2. BUILD AND INITIALIZE MODEL

# Fluid
fluid_pos, fluid_vel, fluid_rho, fluid_p = ps.make_fluid_block(
    water_x0, water_y0, water_width, water_height, dx, rho0_f
)
fluid_p = ps.tait_pressure(fluid_rho, rho0_f, gamma, B_tait)

# Fixed walls
fixed_pos = ps.make_perimeter_walls(domain_width, domain_height, dx, wall_layers)
if wall_above_plate:
    wall_above = ps.make_wall_above_plate(
        plate_x0, plate_height, plate_width, domain_height, dx
    )
    fixed_pos = np.vstack([fixed_pos, wall_above])
fixed_vel = np.zeros_like(fixed_pos)
fixed_rho = rho0_f * np.ones(len(fixed_pos))
fixed_p = ps.tait_pressure(fixed_rho, rho0_f, gamma, B_tait)

# PD plate
pd_ref, _, _, _, _ = ps.build_grid(
    plate_x0, plate_y0, plate_width, plate_height, dx, dx
)
pd = ps.initialize_pd_solver(
    pd_ref,
    horizon=horizon,
    pd_dV=pd_dV,
    rho_s=rho_s,
    lambda_s=lambda_s,
    mu_s=mu_s,
    bulk_s=bulk_s,
    hourglass_coeff=hourglass_coeff,
    bond_damage_max=bond_damage_max,
    F_singular_min=F_singular_min,
    F_singular_max=F_singular_max,
    alpha_visc=alpha_visc,
    beta_visc=beta_visc,
    phi_visc=phi_visc,
    damage_disconnect_threshold=damage_disconnect_threshold,
    critical_energy_density=critical_energy_density,
    damage_softening_kappa=damage_softening_kappa,
    damage_relaxation_time=damage_relaxation_time,
)
n_pd = len(pd_ref)

# Boundary condition and tip probe
y_pd = pd_ref[:, 1]
if clamp_edge == "bottom":
    pd_fixed = np.isclose(y_pd, y_pd.min())
else:
    pd_fixed = np.isclose(y_pd, y_pd.max())

protected_band = 3.0 * dx
if clamp_edge == "top":
    pd["no_fracture_mask"] = y_pd > (y_pd.max() - protected_band)
else:
    pd["no_fracture_mask"] = y_pd < (y_pd.min() + protected_band)

edge_y = y_pd.max() if tip_edge == "top" else y_pd.min()
edge_ids = np.where(np.isclose(y_pd, edge_y))[0]
tip_probe = edge_ids[np.argmin(pd_ref[edge_ids, 0])]
tip_ref = pd_ref[tip_probe].copy()

# Notch and broken-bond tracking
crack_history = []
broken_pairs = np.empty((0, 2), dtype=np.int32)
y_notch = float("nan")
x_min = float(pd_ref[:, 0].min())
x_max = float(pd_ref[:, 0].max())
initial_notch_bonds = 0

if use_initial_notch:
    y_max = float(pd_ref[:, 1].max())
    plate_actual_width = x_max - x_min
    y_notch = y_max - (notch_offset_rows + 0.5) * dx
    notch_depth = notch_depth_fraction * plate_actual_width
    notch_x_from = x_max - notch_depth
    initial_notch_bonds = ps.seed_edge_notch(pd, y_notch, notch_x_from, x_max + horizon)


# State variables
pd_U = np.zeros_like(pd_ref)
pd_V = np.zeros_like(pd_ref)
pd_A = np.zeros_like(pd_ref)
fluid_acc = np.zeros_like(fluid_pos)
fluid_drho = np.zeros_like(fluid_rho)
sph_force_on_pd = np.zeros((n_pd, 2))
history = ps.create_diagnostic_history()

# Output
output_dir = Path(vtk_dir)
output_dir.mkdir(parents=True, exist_ok=True)
for pattern in ("coupling_*.vtk", "pd_crack_*.vtk", "broken_bonds_*.vtk"):
    for old_file in output_dir.glob(pattern):
        old_file.unlink()
if plot_every > 0:
    plt.figure(figsize=(10, 7))

# Initial acceleration
pd_acc_internal, _, pd_pressure = ps.compute_pd(pd, pd_U, pd_V)
pd_current = pd_ref + pd_U
sph_pos, sph_vel, sph_rho, sph_p, sph_mass, ptype, pd_map = ps.assemble_interphase_cloud(
    fluid_pos, fluid_vel, fluid_rho, fluid_p,
    fixed_pos, fixed_vel, fixed_rho, fixed_p,
    pd_current, pd_V, pd_pressure, sph,
)
try:
    initial_pair_count = len(cKDTree(sph_pos).query_pairs(2.0*h, output_type="ndarray"))
except TypeError:
    initial_pair_count = len(cKDTree(sph_pos).query_pairs(2.0*h))
if initial_pair_count > max_sph_pairs:
    raise RuntimeError(f"Initial SPH pair count {initial_pair_count} exceeds max_sph_pairs={max_sph_pairs}")

sph_acc, sph_drho, force_on_pd = ps.compute_sph_and_interphase_forces(
    sph_pos, sph_vel, sph_rho, sph_p, sph_mass, ptype, pd_map,
    sph
)
pd_A = pd_acc_internal + force_on_pd / pd_mass
pd_A[pd_fixed] = 0.0
fluid_acc = sph_acc[:len(fluid_pos)]
fluid_drho = sph_drho[:len(fluid_pos)]
sph_force_on_pd = force_on_pd

print(f"PD-SPH case: {output_prefix}")
print(f"  fluid particles      = {len(fluid_pos):,d}")
print(f"  fixed-wall particles = {len(fixed_pos):,d}")
print(f"  PD particles         = {n_pd:,d}")
print(f"  initial SPH pairs    = {initial_pair_count:,d}")
print(f"  dt                   = {dt:.8e} s")
print(f"  final time           = {final_time:.8e} s")
print(f"  hourglass coeff      = {hourglass_coeff}")
print(f"  energy damage        = {use_energy_failure}")
if use_initial_notch:
    print(f"  initial notch bonds  = {initial_notch_bonds:,d}")
    print(f"  notch depth fraction = {notch_depth_fraction:.3f}")
    print(f"  Gc                   = {fracture_energy_Gc:.6g} J/m^2")

# 3. MAIN SOLVER

step = 0
time = 0.0

while time <= final_time + 1.0e-15:
    dt_now = ps.current_time_step(time, dt, final_time)
    if dt_now <= 0.0:
        break

    # A. Current state
    pd_acc_internal, pd_F, pd_pressure = ps.compute_pd(pd, pd_U, pd_V)
    pd_current = pd_ref + pd_U

    pd_detF = np.linalg.det(pd_F)
    pd_speed = np.linalg.norm(pd_V, axis=1)
    pd_acc_coupling = sph_force_on_pd / pd_mass
    tip = pd_current[tip_probe]

    # Broken-bond tracking
    if use_initial_notch:
        broken_pairs = ps.unique_bond_pairs(pd, damage_disconnect_threshold)
    else:
        broken_pairs = np.empty((0, 2), dtype=np.int32)

    max_amat = float(np.linalg.norm(pd["last_acc_material"], axis=1).max())
    max_ahg = float(np.linalg.norm(pd["last_acc_hourglass"], axis=1).max())
    hg_ratio = max_ahg / max(max_amat, 1.0e-30)
    max_force = float(np.linalg.norm(sph_force_on_pd, axis=1).max())

    # B. Diagnostics and output
    ps.record_diagnostics(
        history,
        time=time, tip_x=tip[0], tip_y=tip[1],
        tip_deflection_x=tip[0]-tip_ref[0], tip_deflection_y=tip[1]-tip_ref[1],
        max_fluid_pressure=float(np.max(fluid_p)),
        max_pd_pressure=float(np.max(pd_pressure)),
        max_sph_force_on_pd=max_force,
        max_damage=float(pd["damage"].max()),
        max_bond_damage=float(pd["bond_damage"].max()),
        min_detF=float(np.min(pd_detF)), max_detF=float(np.max(pd_detF)),
        max_pd_velocity=float(np.max(pd_speed)),
        max_pd_acc_internal=float(np.linalg.norm(pd_acc_internal, axis=1).max()),
        max_pd_acc_material=max_amat,
        max_pd_acc_hourglass=max_ahg,
        max_pd_acc_coupling=float(np.linalg.norm(pd_acc_coupling, axis=1).max()),
        max_hourglass_Z_parallel=float(np.max(pd["last_Z_parallel"])),
        max_bond_energy=float(pd["bond_energy"].max()),
        min_axial_strain=float(np.min(pd["last_axial_strain"])),
        max_axial_strain=float(np.max(pd["last_axial_strain"])),
        max_tensile_damage_driver=float(np.max(pd["last_tensile_damage_driver"])),
        max_tensile_bond_fraction=float(np.max(pd["last_tensile_bond_fraction"])),
    )

    if use_initial_notch:
        crack_history.append([
            time, len(broken_pairs),
            tip[0]-tip_ref[0], float(pd["damage"].max()), float(pd["bond_damage"].max()),
            float(np.min(pd_detF)), float(np.max(pd_detF)), float(np.max(pd_speed)),
            max_amat, max_ahg, hg_ratio, max_force,
        ])

    if log_every > 0 and step % log_every == 0:
        if use_initial_notch:
            print(
                f"step={step:6d}, t={time:.6e}, "
                f"broken={len(broken_pairs):4d}, tip_dx={tip[0]-tip_ref[0]:.4e}, "
                f"max_d={pd['bond_damage'].max():.3f}, "
                f"detF=[{pd_detF.min():.3e},{pd_detF.max():.3e}], "
                f"max|Vpd|={pd_speed.max():.3e}, HG/MAT={hg_ratio:.3e}, "
                f"max|Fcouple|={max_force:.3e}"
            )
        else:
            print(
                f"step={step:6d}, t={time:.6e}, max fluid p={fluid_p.max():.3e}, "
                f"max PD p={pd_pressure.max():.3e}, max |F_SPH->PD|={max_force:.3e}, "
                f"detF=[{pd_detF.min():.3e},{pd_detF.max():.3e}], "
                f"max|Vpd|={pd_speed.max():.3e}, max damage={pd['damage'].max():.3f}, "
                f"|Amat|={max_amat:.3e}, |Ahg|={max_ahg:.3e}, "
                f"tip_dx={tip[0]-tip_ref[0]:.4e}, tip_dy={tip[1]-tip_ref[1]:.4e}"
            )

    if plot_every > 0 and step % plot_every == 0:
        if use_initial_notch:
            ps.plot_notched_state(
                step, time, fluid_pos, fixed_pos, pd_current, pd,
                broken_pairs, y_notch, tip, tip_ref,
                domain_width, domain_height,
            )
        else:
            ps.plot_state(
                step, time, fluid_pos, fixed_pos, pd_current, pd_pressure,
                sph_force_on_pd, domain_width, domain_height, case_title,
            )

    if vtk_every > 0 and step % vtk_every == 0:
        ps.write_vtk(
            step, time, fluid_pos, fluid_vel, fluid_rho, fluid_p,
            fixed_pos, fixed_vel, pd_current, pd_U, pd_V, pd_pressure,
            sph_force_on_pd, pd["damage"], pd_detF, rho0_f, rho_s, vtk_dir,
            pd_acc_material=pd["last_acc_material"],
            pd_acc_hourglass=pd["last_acc_hourglass"],
            pd_Z_parallel=pd["last_Z_parallel"],
            pd_axial_strain=pd["last_axial_strain"],
            pd_max_principal_strain=pd["last_max_principal_strain"],
            pd_tensile_damage_driver=pd["last_tensile_damage_driver"],
            pd_tensile_bond_fraction=pd["last_tensile_bond_fraction"],
        )
        if use_initial_notch:
            ps.write_pd_crack_vtk(
                output_dir / f"pd_crack_{step:07d}.vtk",
                pd_ref, pd_U, pd_V, pd_fixed, pd, pd_F, sph_force_on_pd,
            )
            ps.write_broken_bond_vtk(
                output_dir / f"broken_bonds_{step:07d}.vtk", pd_current, broken_pairs
            )

    if time >= final_time:
        break

    # C. Velocity-Verlet half kick and drift
    pd_Vhalf = pd_V + 0.5 * dt_now * pd_A
    fluid_Vhalf = fluid_vel + 0.5 * dt_now * fluid_acc
    pd_Vhalf[pd_fixed] = 0.0

    pd_U_new = pd_U + dt_now * pd_Vhalf
    fluid_pos_new = fluid_pos + dt_now * fluid_Vhalf
    pd_U_new[pd_fixed] = 0.0

    fluid_rho_new = fluid_rho + dt_now * fluid_drho
    fluid_p_new = ps.tait_pressure(
        fluid_rho_new, rho0_f, gamma, B_tait
    )

    # D. Damage update
    pd_current_new = pd_ref + pd_U_new

    if use_energy_failure:
        _, pd_F_trial, _ = ps.compute_pd(pd, pd_U_new, pd_Vhalf)
        ps.update_energy_damage(
            pd, pd_U_new, pd_U,
            F_old=pd_F, F_new=pd_F_trial, dt_step=dt_now,
        )

    # Recompute PD forces after damage
    pd_acc_internal_new, _, pd_pressure_new = ps.compute_pd(
        pd, pd_U_new, pd_Vhalf
    )

    if not (
        np.all(np.isfinite(pd_current_new))
        and np.all(np.isfinite(pd_Vhalf))
        and np.all(np.isfinite(pd_acc_internal_new))
    ):
        raise FloatingPointError(f"PD state became non-finite at t={time+dt_now:.8e}.")

    # E. New SPH and coupling forces
    sph_pos_new, sph_vel_new, sph_rho_new, sph_p_new, sph_mass_new, ptype_new, pd_map_new = ps.assemble_interphase_cloud(
        fluid_pos_new, fluid_Vhalf, fluid_rho_new, fluid_p_new,
        fixed_pos, fixed_vel, fixed_rho, fixed_p,
        pd_current_new, pd_Vhalf, pd_pressure_new, sph,
    )
    sph_acc_new, sph_drho_new, force_on_pd_new = ps.compute_sph_and_interphase_forces(
        sph_pos_new, sph_vel_new, sph_rho_new, sph_p_new,
        sph_mass_new, ptype_new, pd_map_new, sph,
    )
    pd_A_new = pd_acc_internal_new + force_on_pd_new / pd_mass
    pd_A_new[pd_fixed] = 0.0
    fluid_acc_new = sph_acc_new[:len(fluid_pos)]
    fluid_drho_new = sph_drho_new[:len(fluid_pos)]

    # F. Velocity-Verlet second half kick
    pd_V_new = pd_Vhalf + 0.5 * dt_now * pd_A_new
    fluid_vel_new = fluid_Vhalf + 0.5 * dt_now * fluid_acc_new
    pd_V_new[pd_fixed] = 0.0

    # Update state
    pd_U = pd_U_new
    pd_V = pd_V_new
    pd_A = pd_A_new
    fluid_pos = fluid_pos_new
    fluid_vel = fluid_vel_new
    fluid_rho = fluid_rho_new
    fluid_p = fluid_p_new
    fluid_acc = fluid_acc_new
    fluid_drho = fluid_drho_new
    sph_force_on_pd = force_on_pd_new

    time += dt_now
    step += 1

# 4. FINAL OUTPUT

_, pd_F, pd_pressure = ps.compute_pd(pd, pd_U, pd_V)
pd_current = pd_ref + pd_U
pd_detF = np.linalg.det(pd_F)

ps.write_vtk(
    step, time, fluid_pos, fluid_vel, fluid_rho, fluid_p,
    fixed_pos, fixed_vel, pd_current, pd_U, pd_V, pd_pressure,
    sph_force_on_pd, pd["damage"], pd_detF, rho0_f, rho_s, vtk_dir,
    pd_acc_material=pd["last_acc_material"],
    pd_acc_hourglass=pd["last_acc_hourglass"],
    pd_Z_parallel=pd["last_Z_parallel"],
    pd_axial_strain=pd["last_axial_strain"],
    pd_max_principal_strain=pd["last_max_principal_strain"],
    pd_tensile_damage_driver=pd["last_tensile_damage_driver"],
    pd_tensile_bond_fraction=pd["last_tensile_bond_fraction"],
)

if use_initial_notch:
    broken_pairs = ps.unique_bond_pairs(pd, damage_disconnect_threshold)
    ps.write_pd_crack_vtk(
        output_dir / f"pd_crack_{step:07d}_final.vtk",
        pd_ref, pd_U, pd_V, pd_fixed, pd, pd_F, sph_force_on_pd,
    )
    ps.write_broken_bond_vtk(
        output_dir / f"broken_bonds_{step:07d}_final.vtk", pd_current, broken_pairs
    )
    ps.save_crack_history(f"{output_prefix}_crack_history.csv", crack_history)

ps.save_diagnostics(history, output_prefix, F_singular_min, F_singular_max)

print("\nFinal summary")
print(f"  final time  = {time:.8e} s")
print(f"  VTK folder  = {vtk_dir}")
if use_initial_notch:
    print(f"  fully broken bonds = {len(broken_pairs)}")

if plot_every > 0:
    plt.show()
else:
    plt.close("all")
