from pathlib import Path
import csv
import numpy as np
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree


def tait_pressure(rho, rho0, gamma, B):
    """Pressure from density via the Tait EOS."""
    return np.minimum(B * ((rho / rho0) ** gamma - 1.0), 1.0e6)


def build_grid(x0, y0, length_x, length_y, dx, dy):
    """Rectangular lattice; returns (coords, x_axis, y_axis, nx, ny)."""
    nx = int(length_x / dx)
    ny = int(length_y / dy)
    x = x0 + np.arange(nx) * dx
    y = y0 + np.arange(ny) * dy
    coords = np.array([[x[i], y[j]] for i in range(nx) for j in range(ny)], dtype=float)
    return coords, x, y, nx, ny


def make_fluid_block(x0, y0, width, height, dx, rho0):
    """SPH fluid column; returns (pos, vel, rho, p) with vel=0, rho=rho0, p=0."""
    xs = np.arange(x0, x0 + width, dx)
    ys = np.arange(y0, y0 + height, dx)
    pos = np.array([[x, y] for y in ys for x in xs], dtype=float)
    vel = np.zeros_like(pos)
    rho = rho0 * np.ones(len(pos))
    p = np.zeros(len(pos))
    return pos, vel, rho, p


def make_perimeter_walls(domain_width, domain_height, dx, layers=3):
    """Floor, ceiling, left and right tank walls, `layers` rows thick."""
    pos = []
    x_vals = np.arange(0.0 - 2 * dx, domain_width + 2 * dx, dx)
    y_vals = np.arange(0.0, domain_height + dx, dx)
    for k in range(layers):
        y = -0.5 * dx - k * dx
        pos.extend([x, y] for x in x_vals)
    for k in range(layers):
        y = domain_height + 0.5 * dx + k * dx
        pos.extend([x, y] for x in x_vals)
    for k in range(layers):
        x = -0.5 * dx - k * dx
        pos.extend([x, y] for y in y_vals)
    for k in range(layers):
        x = domain_width + 0.5 * dx + k * dx
        pos.extend([x, y] for y in y_vals)
    return np.array(pos, dtype=float)


def make_wall_above_plate(plate_x0, plate_height, plate_width, domain_height, dx):
    """Fixed wall segment directly above the plate (through-plate case).

    Starts one row above the plate's top lattice line so plate + wall form one
    continuous boundary at uniform spacing, and runs up to the tank roof.
    """
    y0 = int(round(plate_height / dx)) * dx
    coords, _, _, _, _ = build_grid(plate_x0, y0, plate_width,
                                    domain_height - y0 + dx, dx, dx)
    return coords


def wendland_c2_gradient(rij, h):
    """Gradient of the 2D Wendland C2 kernel w.r.t. the first particle.

    rij = x_i - x_j (..., 2); returns grad_i W (..., 2), zero outside 0 < r < 2h.
    """
    r = jnp.linalg.norm(rij, axis=-1)
    q = r / h
    sigma = 7.0 / (4.0 * jnp.pi * (h ** 2))
    dW_dq = -5.0 * sigma * q * ((1.0 - 0.5 * q) ** 3)
    dW_dr = dW_dq / h
    valid = (r > 1.0e-12) & (r < 2.0 * h)
    safe_r = jnp.where(r > 1.0e-12, r, 1.0)
    gradW = dW_dr[..., None] * rij / safe_r[..., None]
    return jnp.where(valid[..., None], gradW, 0.0)


def monaghan_artificial_viscosity(vij, xij, smoothing_length, sound_speed, rho_ij,
                                  alpha, beta, phi):
    """Monaghan artificial viscosity Pi_ij (scalar per pair), active in compression."""
    vdotx = jnp.sum(vij * xij, axis=-1)
    x2 = jnp.sum(xij * xij, axis=-1)
    mu_ij = smoothing_length * vdotx / (x2 + (phi * smoothing_length) ** 2)
    Pi_ij = (-alpha * sound_speed * mu_ij + beta * mu_ij * mu_ij) / rho_ij
    return jnp.where(vdotx < 0.0, Pi_ij, 0.0)


def make_sph_parameters(h, c0, delta_sph, alpha_visc, beta_visc, phi_visc,
                        contact_distance, contact_stiffness, contact_damping,
                        use_fluid_pd_interface_penalty, use_pd_fixed_wall_penalty,
                        g_vector, max_sph_pairs, rho_s, fluid_mass, pd_mass):
    return {
        "h": h,
        "c0": c0,
        "delta_sph": delta_sph,
        "alpha_visc": alpha_visc,
        "beta_visc": beta_visc,
        "phi_visc": phi_visc,
        "contact_distance": contact_distance,
        "contact_stiffness": contact_stiffness,
        "contact_damping": contact_damping,
        "use_fluid_pd_interface_penalty": use_fluid_pd_interface_penalty,
        "use_pd_fixed_wall_penalty": use_pd_fixed_wall_penalty,
        "g_vector": np.asarray(g_vector, dtype=float),
        "max_sph_pairs": max_sph_pairs,
        "rho_s": rho_s,
        "fluid_mass": fluid_mass,
        "pd_mass": pd_mass,
    }


@jit
def compute_sph_forces_jax(pos, vel, rho, p, mass, ptype, pd_map,
                           pairs, pair_valid, force_on_pd,
                           h, c0, delta_sph, alpha_visc, beta_visc, phi_visc,
                           contact_distance, contact_stiffness, contact_damping,
                           use_fluid_pd_interface_penalty,
                           use_pd_fixed_wall_penalty, g_vector):
    n = pos.shape[0]
    i = pairs[:, 0]
    j = pairs[:, 1]
    pair_valid = pair_valid.astype(bool)

    rij = pos[i] - pos[j]
    vij = vel[i] - vel[j]
    r = jnp.linalg.norm(rij, axis=1)
    gradW = wendland_c2_gradient(rij, h)

    is_fluid = ptype == 0
    is_wall = ptype == 1
    is_pd = ptype == 2
    fluid_i = is_fluid[i] & pair_valid
    fluid_j = is_fluid[j] & pair_valid
    pd_i = is_pd[i] & pair_valid
    pd_j = is_pd[j] & pair_valid

    vij_dot_gradW = jnp.sum(vij * gradW, axis=1)

    drho = jnp.zeros(n)
    drho_i = rho[i] * mass[j] / rho[j] * vij_dot_gradW
    drho_j = rho[j] * mass[i] / rho[i] * vij_dot_gradW
    drho = drho.at[i].add(jnp.where(fluid_i, drho_i, 0.0))
    drho = drho.at[j].add(jnp.where(fluid_j, drho_j, 0.0))

    fluid_pair = fluid_i & fluid_j
    r2_eps = r * r + (0.01 * h) ** 2
    rij_dot_gradW = jnp.sum(rij * gradW, axis=1)
    psi = 2.0 * (rho[i] - rho[j]) * rij_dot_gradW / r2_eps
    diff_i = delta_sph * h * c0 * psi * (mass[j] / rho[j])
    diff_j = -delta_sph * h * c0 * psi * (mass[i] / rho[i])
    drho = drho.at[i].add(jnp.where(fluid_pair, diff_i, 0.0))
    drho = drho.at[j].add(jnp.where(fluid_pair, diff_j, 0.0))

    rho_ij = 0.5 * (rho[i] + rho[j])
    Pi_ij = jnp.where(
        pair_valid,
        monaghan_artificial_viscosity(
            vij, rij, h, c0, rho_ij, alpha_visc, beta_visc, phi_visc
        ),
        0.0,
    )
    pressure_term = p[i] / (rho[i] * rho[i]) + p[j] / (rho[j] * rho[j]) + Pi_ij

    a_i_from_j = -mass[j, None] * pressure_term[:, None] * gradW
    a_j_from_i = +mass[i, None] * pressure_term[:, None] * gradW

    acc = jnp.zeros_like(pos)
    acc = acc.at[i].add(jnp.where(fluid_i[:, None], a_i_from_j, 0.0))
    acc = acc.at[j].add(jnp.where(fluid_j[:, None], a_j_from_i, 0.0))

    force_on_fluid_i = mass[i, None] * a_i_from_j
    force_on_fluid_j = mass[j, None] * a_j_from_i
    pd_index_j = jnp.maximum(pd_map[j], 0)
    pd_index_i = jnp.maximum(pd_map[i], 0)
    force_on_pd = force_on_pd.at[pd_index_j].add(
        jnp.where((fluid_i & pd_j)[:, None], -force_on_fluid_i, 0.0)
    )
    force_on_pd = force_on_pd.at[pd_index_i].add(
        jnp.where((fluid_j & pd_i)[:, None], -force_on_fluid_j, 0.0)
    )

    contact_pair_pd = pair_valid & ((fluid_i & pd_j) | (fluid_j & pd_i))
    contact_pair_wall = pair_valid & (
        (fluid_i & (is_wall[j] & pair_valid))
        | (fluid_j & (is_wall[i] & pair_valid))
    )
    contact_pair_pd_wall = pair_valid & (
        (pd_i & (is_wall[j] & pair_valid))
        | (pd_j & (is_wall[i] & pair_valid))
    )
    contact_pair = contact_pair_wall | (
        use_fluid_pd_interface_penalty & contact_pair_pd
    )
    contact_pair = contact_pair | (
        use_pd_fixed_wall_penalty & contact_pair_pd_wall
    )
    contact_pair = contact_pair & (r > 1.0e-12) & (r < contact_distance)

    safe_r = jnp.where(r > 1.0e-12, r, 1.0)
    n_ij = rij / safe_r[:, None]
    normal_speed = jnp.sum(vij * n_ij, axis=1)
    penetration = contact_distance - r
    damping_force_mag = -contact_damping * jnp.minimum(normal_speed, 0.0)
    f_contact_i = (
        contact_stiffness * penetration + damping_force_mag
    )[:, None] * n_ij
    f_contact_i = jnp.where(contact_pair[:, None], f_contact_i, 0.0)
    f_contact_j = -f_contact_i

    contact_i_fluid = contact_pair & fluid_i
    contact_j_fluid = contact_pair & fluid_j
    acc = acc.at[i].add(
        jnp.where(contact_i_fluid[:, None], f_contact_i / mass[i, None], 0.0)
    )
    acc = acc.at[j].add(
        jnp.where(contact_j_fluid[:, None], f_contact_j / mass[j, None], 0.0)
    )

    force_on_pd = force_on_pd.at[pd_index_j].add(
        jnp.where((contact_i_fluid & pd_j)[:, None], -f_contact_i, 0.0)
    )
    force_on_pd = force_on_pd.at[pd_index_i].add(
        jnp.where((contact_j_fluid & pd_i)[:, None], -f_contact_j, 0.0)
    )
    force_on_pd = force_on_pd.at[pd_index_i].add(
        jnp.where((contact_pair_pd_wall & pd_i)[:, None], f_contact_i, 0.0)
    )
    force_on_pd = force_on_pd.at[pd_index_j].add(
        jnp.where((contact_pair_pd_wall & pd_j)[:, None], f_contact_j, 0.0)
    )

    acc = acc + jnp.where(is_fluid[:, None], g_vector, 0.0)
    return acc, drho, force_on_pd


def influence_function(ksi_norm, horizon):
    if ksi_norm < 1.0e-12 or ksi_norm > horizon:
        return 0.0
    return horizon / ksi_norm


def build_pd_family(ref_pos, horizon):
    tree = cKDTree(ref_pos)
    pairs = tree.query_pairs(horizon)
    family = [[] for _ in range(len(ref_pos))]
    for i, j in pairs:
        family[i].append(j)
        family[j].append(i)
    return family


def build_padded_family(family):
    max_family_size = max(len(neighbors) for neighbors in family)
    family_idx = np.zeros((len(family), max_family_size), dtype=np.int32)
    family_mask = np.zeros((len(family), max_family_size), dtype=bool)
    for i, neighbors in enumerate(family):
        if len(neighbors) == 0:
            continue
        family_idx[i, :len(neighbors)] = np.asarray(neighbors, dtype=np.int32)
        family_mask[i, :len(neighbors)] = True
    return family_idx, family_mask


def precompute_shape_tensor_inverse(ref_pos, family, horizon, pd_dV):
    Kinv = np.zeros((len(ref_pos), 2, 2))
    for i in range(len(ref_pos)):
        K = np.zeros((2, 2))
        for j in family[i]:
            ksi = ref_pos[j] - ref_pos[i]
            w = influence_function(np.linalg.norm(ksi), horizon)
            K += w * np.outer(ksi, ksi) * pd_dV
        eig = np.linalg.eigvalsh(K)
        if eig[-1] > 0.0 and eig[0] > 1.0e-10 * eig[-1]:
            Kinv[i] = np.linalg.inv(K)
        else:
            Kinv[i] = np.linalg.pinv(K, rcond=1.0e-12)
    return Kinv


def precompute_reverse_local_indices(family_idx, family_mask):
    """Locate the reverse directed entry j->i for every stored i->j bond."""
    n, max_family = family_idx.shape
    reverse_local_idx = np.zeros((n, max_family), dtype=np.int32)
    reverse_valid = np.zeros((n, max_family), dtype=bool)
    lookup = []
    for i in range(n):
        lookup.append({
            int(family_idx[i, m]): int(m)
            for m in np.where(family_mask[i])[0]
        })
    for i in range(n):
        for m in np.where(family_mask[i])[0]:
            j = int(family_idx[i, m])
            rev = lookup[j].get(i)
            if rev is not None:
                reverse_local_idx[i, m] = rev
                reverse_valid[i, m] = True
    return reverse_local_idx, reverse_valid


@jit
def compute_pd_forces_jax(ref_pos, U, V, family_idx, family_mask,
                          bond_active_mask, bond_weight, Kinv, Kinv_hg,
                          lambda_s, mu_s, bulk_s, horizon, rho_s, pd_dV,
                          hourglass_coeff,
                          F_singular_min, F_singular_max,
                          alpha_visc, beta_visc, phi_visc):
    current = ref_pos + U
    current_j = current[family_idx]
    ksi = ref_pos[family_idx] - ref_pos[:, None, :]
    Yij = current_j - current[:, None, :]
    etaij = U[family_idx] - U[:, None, :]
    ksi_norm = jnp.linalg.norm(ksi, axis=-1)

    w0 = jnp.where(
        family_mask & (ksi_norm > 1.0e-12) & (ksi_norm <= horizon),
        horizon / ksi_norm,
        0.0,
    )
    w = w0 * bond_weight
    ksi_ij = ksi
    ksi_ji = -ksi

    c_solid = jnp.sqrt((lambda_s + 2.0 * mu_s) / rho_s)
    V_j = V[family_idx]
    v_ij_current = V_j - V[:, None, :]
    x_ij_current = current_j - current[:, None, :]
    Pi_ij = monaghan_artificial_viscosity(
        v_ij_current, x_ij_current, horizon,
        c_solid, rho_s, alpha_visc, beta_visc, phi_visc,
    )
    Pi_ij = jnp.where(bond_active_mask, Pi_ij, 0.0)

    N = jnp.sum(
        w[:, :, None, None]
        * etaij[:, :, :, None]
        * ksi[:, :, None, :]
        * pd_dV,
        axis=1,
    )
    F = jnp.eye(2) + jnp.einsum("nab,nbc->nac", N, Kinv)

    detF_raw = jnp.linalg.det(F)
    Uw, sv, Vh = jnp.linalg.svd(F, full_matrices=False)
    sv = jnp.clip(sv, F_singular_min, F_singular_max)
    F_limited = jnp.einsum("nai,ni,nib->nab", Uw, sv, Vh)
    bad = (detF_raw <= 0.0) | (~jnp.isfinite(detF_raw))
    F = jnp.where(bad[:, None, None], jnp.eye(2), F_limited)

    detF = jnp.linalg.det(F)
    FinvT = jnp.swapaxes(jnp.linalg.inv(F), 1, 2)
    E = 0.5 * (
        jnp.einsum("nab,nbc->nac", jnp.swapaxes(F, 1, 2), F)
        - jnp.eye(2)
    )
    trace_E = jnp.trace(E, axis1=1, axis2=2)
    S = lambda_s * trace_E[:, None, None] * jnp.eye(2) + 2.0 * mu_s * E
    P = jnp.einsum("nab,nbc->nac", F, S)
    sigma = jnp.einsum("nab,ncb->nac", P, F) / detF[:, None, None]

    P_j = P[family_idx]
    Kinv_j = Kinv[family_idx]
    detF_j = detF[family_idx]
    FinvT_j = FinvT[family_idx]
    Ki_xi = jnp.einsum("nab,nmb->nma", Kinv, ksi_ij)
    Kj_xi = jnp.einsum("nmab,nmb->nma", Kinv_j, ksi_ji)

    Sigma_i = (
        detF[:, None, None, None]
        * Pi_ij[:, :, None, None]
        * FinvT[:, None, :, :]
    )
    Sigma_j = (
        detF_j[:, :, None, None]
        * Pi_ij[:, :, None, None]
        * FinvT_j
    )
    Piola_i = P[:, None, :, :] - Sigma_i
    Piola_j = P_j - Sigma_j
    Ti = w[:, :, None] * jnp.einsum("nmab,nmb->nma", Piola_i, Ki_xi)
    Tj = w[:, :, None] * jnp.einsum("nmab,nmb->nma", Piola_j, Kj_xi)

    pd_pressure = jnp.maximum(0.0, -0.5 * jnp.trace(sigma, axis1=1, axis2=2))
    pd_pressure = jnp.clip(pd_pressure, 0.0, 1.0e6)

    hg_connected = family_mask & bond_active_mask
    hg_weight = jnp.where(
        hg_connected,
        jnp.clip(bond_weight, 0.0, 1.0),
        0.0,
    )

    w_hg = w0 * hg_weight
    N_hg = jnp.sum(
        w_hg[:, :, None, None]
        * etaij[:, :, :, None]
        * ksi[:, :, None, :]
        * pd_dV,
        axis=1,
    )
    F_hg = jnp.eye(2) + jnp.einsum("nab,nbc->nac", N_hg, Kinv_hg)
    F_hg_j = F_hg[family_idx]
    F_ksi_i = jnp.einsum("nab,nmb->nma", F_hg, ksi_ij)
    F_ksi_j = jnp.einsum("nmab,nmb->nma", F_hg_j, ksi_ji)
    Z_i = Yij - F_ksi_i
    Z_j = -Yij - F_ksi_j
    hg_stiffness = hourglass_coeff * 18.0 * bulk_s / (jnp.pi * horizon ** 4)

    Yij_norm = jnp.linalg.norm(Yij, axis=-1)
    safe_norm = jnp.where(Yij_norm > 1.0e-12, Yij_norm, 1.0)
    n_hat = Yij / safe_norm[:, :, None]
    Zi_parallel = jnp.sum(Z_i * n_hat, axis=-1)[:, :, None] * n_hat
    Zj_parallel = jnp.sum(Z_j * n_hat, axis=-1)[:, :, None] * n_hat
    T_hg_i = hg_stiffness * w_hg[:, :, None] * Zi_parallel
    T_hg_j = hg_stiffness * w_hg[:, :, None] * Zj_parallel

    Z_parallel_node = jnp.max(
        jnp.where(family_mask, jnp.linalg.norm(Zi_parallel, axis=-1), 0.0),
        axis=1,
    )
    rhs_material = jnp.sum(
        (Ti - Tj) * pd_dV * bond_weight[:, :, None], axis=1
    )
    rhs_hourglass = jnp.sum((T_hg_i - T_hg_j) * pd_dV, axis=1)
    acc_material = rhs_material / rho_s
    acc_hourglass = rhs_hourglass / rho_s
    acc = acc_material + acc_hourglass

    return (
        acc, F, pd_pressure, acc_material, acc_hourglass, Z_parallel_node
    )


def rebuild_shape_tensors(pd):
    idx = pd["family_idx"]
    ksi = pd["ref_pos"][idx] - pd["ref_pos"][:, None, :]
    ksi_norm = np.linalg.norm(ksi, axis=-1)
    horizon = pd["horizon"]
    pd_dV = pd["pd_dV"]

    w = np.where(
        pd["family_mask"] & (ksi_norm > 1.0e-12),
        horizon / np.maximum(ksi_norm, 1.0e-12),
        0.0,
    ) * pd["bond_weight"]

    K_raw = np.einsum("nm,nma,nmb->nab", w, ksi, ksi, optimize=True) * pd_dV
    ev = np.linalg.eigvalsh(K_raw)
    ok = (ev[:, 1] > 0.0) & (ev[:, 0] > 1.0e-3 * ev[:, 1])
    K_safe = np.where(ok[:, None, None], K_raw, np.eye(2))

    pd["Kinv"] = np.where(ok[:, None, None], np.linalg.inv(K_safe), 0.0)
    pd["_Kinv"] = jnp.asarray(pd["Kinv"])

    hg_factor = np.where(
        pd["bond_active_mask"],
        np.clip(pd["bond_weight"], 0.0, 1.0),
        0.0,
    )

    w_hg = np.where(
        pd["family_mask"] & (ksi_norm > 1.0e-12),
        horizon / np.maximum(ksi_norm, 1.0e-12) * hg_factor,
        0.0,
    )
    K_hg_raw = np.einsum("nm,nma,nmb->nab", w_hg, ksi, ksi, optimize=True) * pd_dV
    ev_hg = np.linalg.eigvalsh(K_hg_raw)
    ok_hg = (ev_hg[:, 1] > 0.0) & (ev_hg[:, 0] > 1.0e-3 * ev_hg[:, 1])
    K_hg_safe = np.where(ok_hg[:, None, None], K_hg_raw, np.eye(2))
    Kinv_hg = np.where(ok_hg[:, None, None], np.linalg.inv(K_hg_safe), 0.0)
    pd["_Kinv_hg"] = jnp.asarray(Kinv_hg)


def seed_edge_notch(pd, y_notch, x_from, x_to):
    """Fully disconnect every bond crossing a prescribed edge-notch line.

    ``y_notch`` defines a crack plane perpendicular to the reference plate
    axis.  Only crossings whose line-intersection x coordinate lies in
    ``[x_from, x_to]`` are removed.  The directed bond storage is
    symmetrized before rebuilding the shape tensors.
    """
    idx = pd["family_idx"]
    xi = pd["ref_pos"][:, None, 0]
    yi = pd["ref_pos"][:, None, 1]
    xj = pd["ref_pos"][idx, 0]
    yj = pd["ref_pos"][idx, 1]

    dy = yj - yi
    crosses = (
        pd["family_mask"]
        & (np.abs(dy) > 1.0e-14)
        & ((yi - y_notch) * (yj - y_notch) < 0.0)
    )
    safe_dy = np.where(np.abs(dy) > 1.0e-14, dy, 1.0)
    t = np.where(crosses, (y_notch - yi) / safe_dy, 0.0)
    x_cross = xi + t * (xj - xi)
    notch = crosses & (x_cross >= x_from) & (x_cross <= x_to)

    new_damage = np.where(notch, 1.0, pd["bond_damage"])
    reverse_damage = new_damage[
        pd["family_idx"], pd["reverse_local_idx"]
    ]
    new_damage = np.where(
        pd["reverse_valid"],
        np.maximum(new_damage, reverse_damage),
        new_damage,
    )

    pd["bond_damage"] = new_damage
    pd["bond_weight"] = np.where(
        pd["family_mask"], 1.0 - pd["bond_damage"], 0.0
    )
    pd["bond_active_mask"] = (
        pd["family_mask"]
        & (pd["bond_damage"] < pd["damage_disconnect_threshold"])
    )
    family_count = np.maximum(np.sum(pd["family_mask"], axis=1), 1)
    pd["damage"] = np.sum(
        pd["bond_damage"] * pd["family_mask"], axis=1
    ) / family_count
    rebuild_shape_tensors(pd)

    # Each physical bond is stored twice.
    directed = int(np.count_nonzero(notch))
    return directed // 2


def dual_force_density(pd, P, w, ksi, idx):
    """Elastic dual force density T_i<xi_ij> - T_j<xi_ji>.

    P can be either the full elastic Piola stress or a tensile-only Piola
    stress. Keeping this reconstruction unchanged makes the damage split
    act on the thermodynamic driver rather than on the PD kinematics.
    """
    Ti = w[..., None] * np.einsum(
        "nab,nmb->nma", P,
        np.einsum("nab,nmb->nma", pd["Kinv"], ksi),
    )
    Tj = w[..., None] * np.einsum(
        "nmab,nmb->nma", P[idx],
        np.einsum("nmab,nmb->nma", pd["Kinv"][idx], -ksi),
    )
    return Ti - Tj


def strain_and_tensile_piola(pd, F):
    F = np.asarray(F, dtype=float)
    eye = np.eye(2)

    strain = 0.5 * (
        np.einsum("nka,nkb->nab", F, F, optimize=True) - eye
    )

    eigvals, eigvecs = np.linalg.eigh(strain)
    eigvals_pos = np.maximum(eigvals, 0.0)
    strain_pos = np.einsum(
        "nak,nk,nbk->nab", eigvecs, eigvals_pos, eigvecs, optimize=True
    )
    trace_pos = np.maximum(np.trace(strain, axis1=1, axis2=2), 0.0)

    positive_stress = (
        pd["lambda_s"] * trace_pos[:, None, None] * eye
        + 2.0 * pd["mu_s"] * strain_pos
    )
    P_plus = np.einsum("nab,nbc->nac", F, positive_stress, optimize=True)
    psi_plus = (
        0.5 * pd["lambda_s"] * trace_pos ** 2
        + pd["mu_s"] * np.sum(eigvals_pos ** 2, axis=1)
    )
    return strain, P_plus, eigvals[:, -1], psi_plus


def symmetrize_bond_scalar(pd, values):
    """Make the two directed storage entries of every bond identical."""
    reverse = values[pd["family_idx"], pd["reverse_local_idx"]]
    return np.where(
        pd["reverse_valid"],
        0.5 * (values + reverse),
        values,
    )


def update_energy_damage(pd, U_new, U_old, F_old, F_new, dt_step):
    idx = pd["family_idx"]
    ksi = pd["ref_pos"][idx] - pd["ref_pos"][:, None, :]
    ksi_norm = np.linalg.norm(ksi, axis=-1)
    w = np.where(
        pd["bond_active_mask"] & (ksi_norm > 1.0e-12),
        pd["horizon"] / np.maximum(ksi_norm, 1.0e-12),
        0.0,
    ) * pd["bond_weight"]

    eta_old = U_old[idx] - U_old[:, None, :]
    eta_new = U_new[idx] - U_new[:, None, :]
    d_eta = eta_new - eta_old

    strain_old, P_plus_old, _, _ = strain_and_tensile_piola(pd, F_old)
    strain_new, P_plus_new, max_principal_new, psi_plus_new = strain_and_tensile_piola(pd, F_new)

    axial_strain_new = strain_new[:, 1, 1]

    dual_old = dual_force_density(pd, P_plus_old, w, ksi, idx)
    dual_new = dual_force_density(pd, P_plus_new, w, ksi, idx)
    dual = 0.5 * (dual_old + dual_new)
    energy_increment = np.sum(dual * d_eta, axis=-1)
    updated_energy = np.maximum(pd["bond_energy"] + energy_increment, 0.0)

    eligible_count = np.maximum(np.sum(pd["family_mask"], axis=1), 1)
    pd["last_axial_strain"] = axial_strain_new
    pd["last_max_principal_strain"] = max_principal_new
    pd["last_tensile_damage_driver"] = psi_plus_new
    pd["last_tensile_bond_fraction"] = np.sum(pd["family_mask"], axis=1) / eligible_count

    updated_energy = symmetrize_bond_scalar(pd, updated_energy)
    pd["bond_energy"] = np.where(
        pd["bond_active_mask"], updated_energy, pd["bond_energy"]
    )

    if pd["no_fracture_mask"] is not None:
        protected_bond = (
            pd["no_fracture_mask"][:, None] | pd["no_fracture_mask"][idx]
        )
    else:
        protected_bond = np.zeros_like(pd["family_mask"])
    breakable = pd["family_mask"] & ~protected_bond

    wc = pd["critical_energy_density"]
    kappa = max(pd["damage_softening_kappa"], 1.0e-12)
    d_target = np.clip(
        (pd["bond_energy"] - wc) / (kappa * wc),
        0.0,
        pd["bond_damage_max"],
    )
    d_target = np.where(breakable, d_target, 0.0)

    tau = pd["damage_relaxation_time"]
    if tau > 0.0:
        alpha = min(1.0, dt_step / tau)
        new_damage = pd["bond_damage"] + alpha * np.maximum(
            d_target - pd["bond_damage"], 0.0
        )
    else:
        new_damage = np.maximum(pd["bond_damage"], d_target)

    reverse_damage = new_damage[pd["family_idx"], pd["reverse_local_idx"]]
    new_damage = np.where(
        pd["reverse_valid"], np.maximum(new_damage, reverse_damage), new_damage
    )

    disconnect = pd["damage_disconnect_threshold"]
    new_damage = np.where(
        pd["family_mask"] & (new_damage >= disconnect), 1.0, new_damage
    )

    changed = not np.array_equal(new_damage, pd["bond_damage"])
    pd["bond_damage"] = new_damage
    pd["bond_weight"] = np.where(
        pd["family_mask"], 1.0 - pd["bond_damage"], 0.0
    )
    pd["bond_active_mask"] = (
        pd["family_mask"] & (pd["bond_damage"] < disconnect)
    )

    family_count = np.maximum(np.sum(pd["family_mask"], axis=1), 1)
    pd["damage"] = (
        np.sum(pd["bond_damage"] * pd["family_mask"], axis=1) / family_count
    )
    if changed:
        rebuild_shape_tensors(pd)
    return pd["damage"]


# PLAIN PD STATE FUNCTIONS

def initialize_pd_solver(
    ref_pos, horizon, pd_dV, rho_s, lambda_s, mu_s, bulk_s,
    hourglass_coeff, bond_damage_max,
    F_singular_min, F_singular_max, alpha_visc, beta_visc, phi_visc,
    damage_disconnect_threshold,
    critical_energy_density, damage_softening_kappa, damage_relaxation_time,
):
    ref_pos = np.asarray(ref_pos, dtype=float)
    family = build_pd_family(ref_pos, horizon)
    family_idx, family_mask = build_padded_family(family)
    Kinv = precompute_shape_tensor_inverse(ref_pos, family, horizon, pd_dV)
    reverse_local_idx, reverse_valid = precompute_reverse_local_indices(
        family_idx, family_mask
    )

    pd = {
        "ref_pos": ref_pos,
        "n": len(ref_pos),
        "family_idx": family_idx,
        "family_mask": family_mask,
        "Kinv": Kinv,
        "reverse_local_idx": reverse_local_idx,
        "reverse_valid": reverse_valid,
        "bond_damage": np.zeros_like(family_mask, dtype=float),
        "bond_energy": np.zeros_like(family_mask, dtype=float),
        "bond_weight": family_mask.astype(float),
        "damage": np.zeros(len(ref_pos)),
        "bond_active_mask": family_mask.copy(),
        "last_acc_material": np.zeros_like(ref_pos),
        "last_acc_hourglass": np.zeros_like(ref_pos),
        "last_Z_parallel": np.zeros(len(ref_pos)),
        "last_axial_strain": np.zeros(len(ref_pos)),
        "last_max_principal_strain": np.zeros(len(ref_pos)),
        "last_tensile_damage_driver": np.zeros(len(ref_pos)),
        "last_tensile_bond_fraction": np.zeros(len(ref_pos)),
        "no_fracture_mask": None,
        "horizon": horizon,
        "pd_dV": pd_dV,
        "rho_s": rho_s,
        "lambda_s": lambda_s,
        "mu_s": mu_s,
        "bulk_s": bulk_s,
        "hourglass_coeff": hourglass_coeff,
        "bond_damage_max": bond_damage_max,
        "F_singular_min": F_singular_min,
        "F_singular_max": F_singular_max,
        "alpha_visc": alpha_visc,
        "beta_visc": beta_visc,
        "phi_visc": phi_visc,
        "damage_disconnect_threshold": damage_disconnect_threshold,
        "critical_energy_density": critical_energy_density,
        "damage_softening_kappa": damage_softening_kappa,
        "damage_relaxation_time": damage_relaxation_time,
    }

    pd["_ref"] = jnp.asarray(ref_pos)
    pd["_family_idx"] = jnp.asarray(family_idx)
    pd["_family_mask"] = jnp.asarray(family_mask)
    pd["_Kinv"] = jnp.asarray(Kinv)
    pd["_Kinv_hg"] = jnp.asarray(Kinv)
    return pd


def compute_pd(pd, U, V):
    (acc, F, pressure, acc_material, acc_hourglass,
     Z_parallel) = compute_pd_forces_jax(
        pd["_ref"], jnp.asarray(U), jnp.asarray(V),
        pd["_family_idx"], pd["_family_mask"],
        jnp.asarray(pd["bond_active_mask"]), jnp.asarray(pd["bond_weight"]),
        pd["_Kinv"], pd["_Kinv_hg"],
        pd["lambda_s"], pd["mu_s"], pd["bulk_s"], pd["horizon"],
        pd["rho_s"], pd["pd_dV"],
        pd["hourglass_coeff"], pd["F_singular_min"],
        pd["F_singular_max"], pd["alpha_visc"], pd["beta_visc"],
        pd["phi_visc"],
    )
    acc = np.asarray(acc)
    pd["last_acc_material"] = np.asarray(acc_material)
    pd["last_acc_hourglass"] = np.asarray(acc_hourglass)
    pd["last_Z_parallel"] = np.asarray(Z_parallel)
    return acc, np.asarray(F), np.asarray(pressure)


# PLAIN SPH / COUPLING FUNCTIONS

def assemble_interphase_cloud(fluid_pos, fluid_vel, fluid_rho, fluid_p,
                              fixed_pos, fixed_vel, fixed_rho, fixed_p,
                              pd_pos, pd_vel, pd_pressure, sph):
    n_f, n_w, n_pd = len(fluid_pos), len(fixed_pos), len(pd_pos)
    pos = np.vstack([fluid_pos, fixed_pos, pd_pos])
    vel = np.vstack([fluid_vel, fixed_vel, pd_vel])
    rho = np.hstack([fluid_rho, fixed_rho, sph["rho_s"] * np.ones(n_pd)])
    p = np.hstack([fluid_p, fixed_p, pd_pressure])
    mass = np.hstack([
        sph["fluid_mass"] * np.ones(n_f),
        sph["fluid_mass"] * np.ones(n_w),
        sph["pd_mass"] * np.ones(n_pd),
    ])
    ptype = np.zeros(len(pos), dtype=int)
    ptype[n_f:n_f+n_w] = 1
    ptype[n_f+n_w:] = 2
    pd_map = -np.ones(len(pos), dtype=int)
    pd_map[n_f+n_w:] = np.arange(n_pd)
    return pos, vel, rho, p, mass, ptype, pd_map


def compute_sph_and_interphase_forces(pos, vel, rho, p, mass, ptype, pd_map,
                                      sph):
    n = len(pos)
    n_pd = 0 if np.max(pd_map) < 0 else int(np.max(pd_map)) + 1
    h = sph["h"]
    max_sph_pairs = sph["max_sph_pairs"]

    tree = cKDTree(pos)
    try:
        pairs = tree.query_pairs(2.0 * h, output_type="ndarray")
        pairs = np.asarray(pairs, dtype=np.int32)
    except TypeError:
        pairs = np.asarray(list(tree.query_pairs(2.0 * h)), dtype=np.int32)

    if len(pairs) == 0:
        acc = np.zeros_like(pos)
        acc[ptype == 0] += sph["g_vector"]
        return acc, np.zeros(n), np.zeros((n_pd, 2))

    if len(pairs) > max_sph_pairs:
        raise RuntimeError(
            f"SPH pair list has {len(pairs)} pairs > max_sph_pairs={max_sph_pairs}."
        )

    pair_valid = np.zeros(max_sph_pairs, dtype=bool)
    pairs_padded = np.zeros((max_sph_pairs, 2), dtype=np.int32)
    pair_valid[:len(pairs)] = True
    pairs_padded[:len(pairs)] = pairs
    force_on_pd = np.zeros((n_pd, 2), dtype=float)

    acc, drho, force_on_pd = compute_sph_forces_jax(
        jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(rho), jnp.asarray(p),
        jnp.asarray(mass), jnp.asarray(ptype), jnp.asarray(pd_map),
        jnp.asarray(pairs_padded), jnp.asarray(pair_valid),
        jnp.asarray(force_on_pd),
        sph["h"], sph["c0"], sph["delta_sph"], sph["alpha_visc"],
        sph["beta_visc"], sph["phi_visc"], sph["contact_distance"],
        sph["contact_stiffness"], sph["contact_damping"],
        sph["use_fluid_pd_interface_penalty"], sph["use_pd_fixed_wall_penalty"],
        jnp.asarray(sph["g_vector"]),
    )
    return np.asarray(acc), np.asarray(drho), np.asarray(force_on_pd)


def current_time_step(time, dt, final_time):
    return min(dt, final_time - time)


def write_vtk(step, time, fluid_pos, fluid_vel, fluid_rho, fluid_p,
              fixed_pos, fixed_vel, pd_pos, pd_U, pd_V, pd_pressure,
              sph_force_on_pd, pd_damage, pd_detF, rho0_f, rho_s, output_dir,
              pd_acc_material=None, pd_acc_hourglass=None,
              pd_Z_parallel=None,
              pd_axial_strain=None, pd_max_principal_strain=None,
              pd_tensile_damage_driver=None, pd_tensile_bond_fraction=None):
    """Write one ASCII POLYDATA frame (fluid + walls + PD particles)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = Path(output_dir) / f"coupling_{step:07d}.vtk"

    n_f, n_w, n_pd = len(fluid_pos), len(fixed_pos), len(pd_pos)
    points = np.vstack([fluid_pos, fixed_pos, pd_pos])
    n_total = len(points)

    ptype = np.hstack([np.zeros(n_f), np.ones(n_w), 2.0 * np.ones(n_pd)])
    pressure = np.hstack([fluid_p, np.zeros(n_w), pd_pressure])
    density = np.hstack([fluid_rho, rho0_f * np.ones(n_w), rho_s * np.ones(n_pd)])
    velocity = np.vstack([fluid_vel, fixed_vel, pd_V])
    displacement = np.vstack([np.zeros((n_f + n_w, 2)), pd_U])
    force = np.vstack([np.zeros((n_f + n_w, 2)), sph_force_on_pd])
    damage = np.hstack([np.zeros(n_f + n_w), pd_damage])
    if pd_detF is None:
        pd_detF = np.ones(n_pd)
    if pd_acc_material is None:
        pd_acc_material = np.zeros((n_pd, 2))
    if pd_acc_hourglass is None:
        pd_acc_hourglass = np.zeros((n_pd, 2))
    if pd_Z_parallel is None:
        pd_Z_parallel = np.zeros(n_pd)
    if pd_axial_strain is None:
        pd_axial_strain = np.zeros(n_pd)
    if pd_max_principal_strain is None:
        pd_max_principal_strain = np.zeros(n_pd)
    if pd_tensile_damage_driver is None:
        pd_tensile_damage_driver = np.zeros(n_pd)
    if pd_tensile_bond_fraction is None:
        pd_tensile_bond_fraction = np.zeros(n_pd)

    detF_field = np.hstack([np.zeros(n_f + n_w), pd_detF])
    Z_parallel_field = np.hstack([np.zeros(n_f + n_w), pd_Z_parallel])
    axial_strain_field = np.hstack([np.zeros(n_f + n_w), pd_axial_strain])
    max_principal_strain_field = np.hstack([np.zeros(n_f + n_w), pd_max_principal_strain])
    tensile_damage_driver_field = np.hstack([np.zeros(n_f + n_w), pd_tensile_damage_driver])
    tensile_bond_fraction_field = np.hstack([np.zeros(n_f + n_w), pd_tensile_bond_fraction])
    acc_material_field = np.vstack([np.zeros((n_f + n_w, 2)), pd_acc_material])
    acc_hourglass_field = np.vstack([np.zeros((n_f + n_w, 2)), pd_acc_hourglass])

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"PD-SPH coupling step {step}, time {time}\n")
        f.write("ASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n_total} float\n")
        for x, y in points:
            f.write(f"{x:.9e} {y:.9e} 0.0\n")
        f.write(f"VERTICES {n_total} {2 * n_total}\n")
        for k in range(n_total):
            f.write(f"1 {k}\n")
        f.write(f"POINT_DATA {n_total}\n")
        for name, fld in [("particle_type", ptype), ("pressure", pressure),
                          ("density", density), ("damage", damage), ("detF", detF_field),
                          ("hourglass_Z_parallel", Z_parallel_field),
                          ("axial_strain", axial_strain_field),
                          ("max_principal_strain", max_principal_strain_field),
                          ("tensile_damage_driver", tensile_damage_driver_field),
                          ("tensile_bond_fraction", tensile_bond_fraction_field)]:
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for val in fld:
                f.write(f"{val:.9e}\n")
        for name, fld in [("velocity", velocity), ("displacement", displacement),
                          ("sph_force_on_pd", force),
                          ("pd_acc_material", acc_material_field),
                          ("pd_acc_hourglass", acc_hourglass_field)]:
            f.write(f"VECTORS {name} float\n")
            for vx, vy in fld:
                f.write(f"{vx:.9e} {vy:.9e} 0.0\n")


def plot_state(step, time, fluid_pos, fixed_pos, pd_pos, pd_pressure,
               sph_force_on_pd, domain_width, domain_height, title_tag):
    """Live view using the same visual style as the notched-fracture case."""
    plt.clf()
    ax = plt.gca()

    ax.scatter(
        fluid_pos[:, 0], fluid_pos[:, 1],
        s=3.0, c="tab:blue", alpha=0.65, label="SPH fluid"
    )
    ax.scatter(
        fixed_pos[:, 0], fixed_pos[:, 1],
        s=1.0, c="0.65", label="fixed walls"
    )

    # Cases 1 and 2 do not have the notched-fracture damage field used in
    # Case 3, so keep the useful PD-pressure coloring but use the same style.
    sc = ax.scatter(
        pd_pos[:, 0], pd_pos[:, 1],
        s=8.0, c=pd_pressure, cmap="inferno",
        linewidths=0.0, label="PD plate"
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.004, domain_width + 0.004)
    ax.set_ylim(-0.004, domain_height + 0.004)

    maxf = 0.0 if len(sph_force_on_pd) == 0 else np.linalg.norm(sph_force_on_pd, axis=1).max()
    maxp = 0.0 if len(pd_pressure) == 0 else np.max(pd_pressure)
    ax.set_title(
        f"{title_tag} | step {step:,d} | t={time:.5f} s\n"
        f"max PD pressure={maxp:.3e} Pa | max |F_SPH->PD|={maxf:.3e} N"
    )
    ax.legend(loc="upper left", markerscale=2)
    plt.colorbar(sc, ax=ax, label="PD pressure from stress")
    plt.tight_layout()
    plt.pause(0.001)


# DIAGNOSTIC HISTORY

DIAGNOSTIC_COLUMNS = [
    "time", "tip_x", "tip_y", "tip_deflection_x", "tip_deflection_y",
    "max_fluid_pressure", "max_pd_pressure", "max_sph_force_on_pd",
    "max_damage", "max_bond_damage", "min_detF", "max_detF", "max_pd_velocity",
    "max_pd_acc_internal", "max_pd_acc_material", "max_pd_acc_hourglass",
    "max_pd_acc_coupling", "max_hourglass_Z_parallel",
    "max_bond_energy", "min_axial_strain",
    "max_axial_strain", "max_tensile_damage_driver", "max_tensile_bond_fraction",
]


def create_diagnostic_history():
    return {name: [] for name in DIAGNOSTIC_COLUMNS}


def record_diagnostics(history, **values):
    for name in DIAGNOSTIC_COLUMNS:
        history[name].append(values[name])


def save_diagnostics(history, prefix, F_singular_min, F_singular_max):
    table = np.column_stack([np.asarray(history[name]) for name in DIAGNOSTIC_COLUMNS])
    np.savetxt(f"{prefix}_tip_history.csv", table, delimiter=",",
               header=",".join(DIAGNOSTIC_COLUMNS), comments="")

    t = history["time"]
    plt.figure(figsize=(8, 4))
    plt.plot(t, history["tip_deflection_x"], label="tip x-deflection")
    plt.plot(t, history["tip_deflection_y"], label="tip y-deflection")
    plt.xlabel("time [s]"); plt.ylabel("displacement [m]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_tip_deflection.png", dpi=200)

    plt.figure(figsize=(8, 4))
    plt.plot(t, history["min_detF"], label="min detF")
    plt.plot(t, history["max_detF"], label="max detF")
    plt.axhline(F_singular_min**2, linestyle="--", linewidth=1, label="approx lower detF limit")
    plt.axhline(F_singular_max**2, linestyle="--", linewidth=1, label="approx upper detF limit")
    plt.xlabel("time [s]"); plt.ylabel("detF [-]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_detF_history.png", dpi=200)

    plt.figure(figsize=(8, 4))
    plt.semilogy(t, np.maximum(history["max_pd_acc_internal"], 1e-30), label="max |Aint|")
    plt.semilogy(t, np.maximum(history["max_pd_acc_material"], 1e-30), label="max |Amaterial|")
    plt.semilogy(t, np.maximum(history["max_pd_acc_hourglass"], 1e-30), label="max |Ahourglass|")
    plt.semilogy(t, np.maximum(history["max_pd_acc_coupling"], 1e-30), label="max |Acoup|")
    plt.semilogy(t, np.maximum(history["max_pd_velocity"], 1e-30), label="max |Vpd|")
    plt.xlabel("time [s]"); plt.ylabel("diagnostic magnitude"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_pd_diagnostics.png", dpi=200)

    plt.figure(figsize=(8, 4))
    plt.semilogy(t, np.maximum(history["max_hourglass_Z_parallel"], 1e-30), label="max |Z_parallel|")
    plt.xlabel("time [s]"); plt.ylabel("hourglass residual [m]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"{prefix}_hourglass_residuals.png", dpi=200)

    print(f"Saved {prefix}_tip_history.csv")
    print(f"Saved {prefix}_tip_deflection.png")
    print(f"Saved {prefix}_detF_history.png")
    print(f"Saved {prefix}_pd_diagnostics.png")
    print(f"Saved {prefix}_hourglass_residuals.png")


# --- Notch / crack utilities ---


def unique_bond_pairs(pd, threshold: float) -> np.ndarray:
    """Return each physical bond once (i < j) when damage >= threshold."""
    i = np.repeat(np.arange(pd["n"], dtype=np.int32), pd["family_idx"].shape[1])
    j = pd["family_idx"].reshape(-1)
    valid = pd["family_mask"].reshape(-1)
    damage = pd["bond_damage"].reshape(-1)
    keep = valid & (i < j) & (damage >= threshold)
    return np.column_stack((i[keep], j[keep])).astype(np.int32, copy=False)




def write_broken_bond_vtk(filename, current: np.ndarray, pairs: np.ndarray):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\nFully disconnected PD bonds\n")
        f.write("ASCII\nDATASET POLYDATA\n")
        if len(pairs) == 0:
            f.write("POINTS 0 float\nLINES 0 0\n")
            return
        points = np.empty((2 * len(pairs), 2), dtype=float)
        points[0::2] = current[pairs[:, 0]]
        points[1::2] = current[pairs[:, 1]]
        f.write(f"POINTS {len(points)} float\n")
        for x, y in points:
            f.write(f"{x:.9e} {y:.9e} 0.0\n")
        f.write(f"LINES {len(pairs)} {3 * len(pairs)}\n")
        for k in range(len(pairs)):
            f.write(f"2 {2*k} {2*k+1}\n")
        f.write(f"CELL_DATA {len(pairs)}\n")
        f.write("SCALARS fully_broken float 1\nLOOKUP_TABLE default\n")
        for _ in pairs:
            f.write("1.0\n")


def write_pd_crack_vtk(filename, pd_ref, pd_U, pd_V, pd_fixed, pd,
                       pd_F, sph_force_on_pd):
    """Write the PD-only fracture diagnostics used by the notched case."""
    filename = Path(filename)
    current = pd_ref + pd_U
    n = len(current)
    max_bond_damage = np.max(
        np.where(pd["family_mask"], pd["bond_damage"], 0.0), axis=1
    )
    scalars = [
        ("damage", pd["damage"]),
        ("max_bond_damage", max_bond_damage),
        ("detF", np.linalg.det(pd_F)),
        ("fixed", pd_fixed.astype(float)),
        ("fracture_protected", pd["no_fracture_mask"].astype(float)),
        ("axial_strain", pd["last_axial_strain"]),
        ("max_principal_strain", pd["last_max_principal_strain"]),
        ("tensile_damage_driver", pd["last_tensile_damage_driver"]),
    ]
    vectors = [
        ("displacement", pd_U),
        ("velocity", pd_V),
        ("pd_acc_material", pd["last_acc_material"]),
        ("pd_acc_hourglass", pd["last_acc_hourglass"]),
        ("sph_force_on_pd", sph_force_on_pd),
    ]
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\nPD crack fields\n")
        f.write("ASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for x, y in current:
            f.write(f"{x:.9e} {y:.9e} 0.0\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        for name, values in scalars:
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            for value in np.asarray(values):
                f.write(f"{float(value):.9e}\n")
        for name, values in vectors:
            f.write(f"VECTORS {name} float\n")
            for vx, vy in np.asarray(values):
                f.write(f"{float(vx):.9e} {float(vy):.9e} 0.0\n")


def plot_notched_state(step, time, fluid_pos, fixed_pos, pd_current, pd,
                       broken_pairs, y_notch, tip, tip_ref,
                       domain_width, domain_height):
    """Live view for the notched fracture case."""
    max_bond_damage = np.max(
        np.where(pd["family_mask"], pd["bond_damage"], 0.0), axis=1
    )
    plt.clf()
    ax = plt.gca()
    ax.scatter(fluid_pos[:, 0], fluid_pos[:, 1], s=3.0,
               c="tab:blue", alpha=0.65, label="SPH fluid")
    ax.scatter(fixed_pos[:, 0], fixed_pos[:, 1], s=1.0,
               c="0.65", label="fixed walls")
    sc = ax.scatter(pd_current[:, 0], pd_current[:, 1], s=8.0,
                    c=max_bond_damage, cmap="inferno", vmin=0.0, vmax=1.0,
                    linewidths=0.0, label="PD plate")
    if len(broken_pairs):
        seg = np.stack((pd_current[broken_pairs[:, 0]], pd_current[broken_pairs[:, 1]]), axis=1)
        ax.add_collection(LineCollection(seg, colors="red", linewidths=0.25, alpha=0.35))
    ax.axhline(y_notch, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.004, domain_width + 0.004)
    ax.set_ylim(-0.004, domain_height + 0.004)
    ax.set_title(
        f"Small coupled notched fracture | step {step:,d} | t={time:.5f} s\n"
        f"tip dx={(tip[0]-tip_ref[0])*1e3:.2f} mm | "
        f"max d={pd["bond_damage"].max():.3f}"
    )
    ax.legend(loc="upper left", markerscale=2)
    plt.colorbar(sc, ax=ax, label="max bond damage at PD node")
    plt.tight_layout()
    plt.pause(0.001)


def save_crack_history(filename, crack_history):
    columns = [
        "time", "broken_bonds",
        "tip_dx", "max_nodal_damage", "max_bond_damage",
        "min_detF", "max_detF", "max_pd_velocity",
        "max_acc_material", "max_acc_hourglass", "hg_to_material_ratio",
        "max_sph_force_on_pd",
    ]
    with Path(filename).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(crack_history)
    print(f"Saved {filename}")
