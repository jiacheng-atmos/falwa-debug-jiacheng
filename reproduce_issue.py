"""
reproduce_issue.py
------------------
Demonstrates the QGPV blow-up in ``QGFieldNHN22`` that occurs when the
hemispheric static stability becomes very small.

The blow-up has two necessary ingredients:
1. Very small static stability (dtheta/dz << 1 K/m) at some pseudoheight level
   for one hemisphere — this acts as the near-zero denominator.
2. Non-zero spatial temperature anomalies (theta != hemispheric mean theta) —
   this provides the non-zero numerator that is divided by the near-zero stat.

The demonstration is at two levels:
* **Low-level**: directly calling the Fortran kernel ``compute_qgpv_direct_inv``
  with a prescribed tiny static stability to isolate the mechanism.
* **High-level**: using ``QGFieldNHN22`` and ``QGFieldNH18`` with a synthetic
  near-neutral-lapse-rate SH atmosphere that produces a near-zero spline
  derivative at z = 0-2 km.

Run
---
    python reproduce_issue.py

References
----------
* hn2016_falwa issue #157: https://github.com/csyhuang/hn2016_falwa/issues/157
"""

import numpy as np

from falwa import compute_qgpv_direct_inv, compute_qgpv
from falwa.oopinterface import QGFieldNHN22, QGFieldNH18
from diagnose_static_stability import diagnose_static_stability


# ============================================================
# Shared grid parameters
# ============================================================

NLON = 72
NLAT = 181
KMAX = 25
DZ = 1000.0  # m
SCALE_HEIGHT = 7000.0  # m
KAPPA = 287.0 / 1004.0

xlon = np.linspace(0, 355.0, NLON)
ylat = np.linspace(-90.0, 90.0, NLAT)
height = np.arange(KMAX) * DZ
lon_rad = np.deg2rad(xlon)
lat_rad = np.deg2rad(ylat)
equator_idx = int(NLAT // 2) + 1  # Fortran 1-based


# ============================================================
# Helper: build a synthetic atmospheric state (Fortran layout)
# ============================================================

def build_synthetic_fields(wave_amplitude=15.0):
    """
    Return ``(u, v, theta, t0_s, t0_n)`` in Fortran layout (nlon, nlat, kmax).

    Temperature profile: standard 6.5 K/km everywhere.
    Spatial structure: wavenumber-3 zonal wave in theta (zero hemispheric mean).
    """
    u = np.zeros((NLON, NLAT, KMAX))
    v = np.zeros((NLON, NLAT, KMAX))
    theta = np.zeros((NLON, NLAT, KMAX))

    for k in range(KMAX):
        z = height[k]
        T_k = 300.0 - 6.5e-3 * z
        theta_base = T_k * np.exp(KAPPA * z / SCALE_HEIGHT)
        # Wave-3 anomaly with zero hemispheric mean
        T_anom = wave_amplitude * np.cos(3 * lon_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :])
        theta[:, :, k] = theta_base + T_anom * np.exp(KAPPA * z / SCALE_HEIGHT)
        u[:, :, k] = 30.0 * np.exp(-((ylat[np.newaxis, :] - 45.0) / 15.0) ** 2)
        v[:, :, k] = 5.0 * np.sin(3 * lon_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :])

    # Hemispheric mean theta (the anomaly averages to zero)
    t0 = np.array([
        (300.0 - 6.5e-3 * z) * np.exp(KAPPA * z / SCALE_HEIGHT)
        for z in height
    ])
    return u, v, theta, t0.copy(), t0.copy()


# ============================================================
# PART 1 — Low-level demonstration (direct kernel call)
# ============================================================

print("=" * 65)
print("PART 1: Low-level Fortran kernel demonstration")
print("=" * 65)

u, v, theta, t0_s, t0_n = build_synthetic_fields(wave_amplitude=15.0)

# -- 1a: Normal static stability (typical troposphere ~5 K/km) --
stat_normal = 5e-3 * np.ones(KMAX)

qgpv_normal, _ = compute_qgpv_direct_inv(
    equator_idx, u, v, theta, height, t0_s, t0_n,
    stat_normal, stat_normal,
    6.378e6, 7.29e-5, DZ, SCALE_HEIGHT, 287.0, 1004.0,
)
print(f"\n1a. Normal static stability ({stat_normal[0]:.0e} K/m):")
print(f"    max |QGPV SH| = {np.nanmax(np.abs(qgpv_normal[:, :NLAT//2, :])):.3e} s⁻¹")
print(f"    max |QGPV NH| = {np.nanmax(np.abs(qgpv_normal[:, NLAT//2:, :])):.3e} s⁻¹")

# -- 1b: Near-zero SH static stability (triggers the bug) --
stat_tiny_s = 1e-5 * np.ones(KMAX)

qgpv_blowup, _ = compute_qgpv_direct_inv(
    equator_idx, u, v, theta, height, t0_s, t0_n,
    stat_tiny_s, stat_normal,   # tiny stat only in SH
    6.378e6, 7.29e-5, DZ, SCALE_HEIGHT, 287.0, 1004.0,
)
print(f"\n1b. Near-zero SH static stability ({stat_tiny_s[0]:.0e} K/m):")
print(f"    max |QGPV SH| = {np.nanmax(np.abs(qgpv_blowup[:, :NLAT//2, :])):.3e} s⁻¹  <-- BLOWN UP")
print(f"    max |QGPV NH| = {np.nanmax(np.abs(qgpv_blowup[:, NLAT//2:, :])):.3e} s⁻¹")
print(f"    Any inf? {np.any(np.isinf(qgpv_blowup))}")

# -- 1c: Global static stability (the fix) --
stat_global = 0.5 * (stat_tiny_s + stat_normal)

qgpv_fixed, _ = compute_qgpv_direct_inv(
    equator_idx, u, v, theta, height, t0_s, t0_n,
    stat_global, stat_global,   # averaged stat
    6.378e6, 7.29e-5, DZ, SCALE_HEIGHT, 287.0, 1004.0,
)
print(f"\n1c. Global static stability (fix, {stat_global[0]:.3e} K/m):")
print(f"    max |QGPV SH| = {np.nanmax(np.abs(qgpv_fixed[:, :NLAT//2, :])):.3e} s⁻¹")
print(f"    max |QGPV NH| = {np.nanmax(np.abs(qgpv_fixed[:, NLAT//2:, :])):.3e} s⁻¹")

blow_up_ratio = np.nanmax(np.abs(qgpv_blowup)) / np.nanmax(np.abs(qgpv_normal))
fix_ratio = np.nanmax(np.abs(qgpv_fixed)) / np.nanmax(np.abs(qgpv_normal))
print(f"\n    Blow-up ratio (1b/1a): {blow_up_ratio:.0f}×")
print(f"    Fix ratio    (1c/1a): {fix_ratio:.2f}×")


# ============================================================
# PART 2 — High-level QGFieldNHN22 vs QGFieldNH18 comparison
# ============================================================

print("\n" + "=" * 65)
print("PART 2: QGFieldNHN22 vs QGFieldNH18 comparison")
print("=" * 65)
print(
    "\nUsing pressure-level input with a near-neutral SH lower troposphere.\n"
    "The spline derivative falls close to zero at z = 0-2 km in the SH."
)

# -- Build pressure-level data --
plev = np.array(
    [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
     700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
     150, 100, 70, 50, 30, 20, 10], dtype=float,
)
nlev = plev.size
sh_idx = NLAT // 2

T_surface = 300.0 - 40.0 * np.abs(np.sin(np.deg2rad(ylat)))

# Near-neutral SH lower troposphere: lapse rate close to dry adiabatic
# so dtheta/dz is near zero below 2 km.
# Dry adiabatic in pseudoheight coords: gamma_dry = T_surface_mean * kappa / H
clat = np.abs(np.cos(np.deg2rad(ylat)))
T_surface_mean_sh = np.sum(T_surface[:sh_idx] * clat[:sh_idx]) / np.sum(clat[:sh_idx])
gamma_neutral = T_surface_mean_sh * KAPPA / SCALE_HEIGHT  # ~11.4 K/km in pseudoheight

T_field = np.zeros((nlev, NLAT, NLON))
gamma_nh = 6.5e-3

for i, p in enumerate(plev):
    z = -SCALE_HEIGHT * np.log(p / 1000.0)
    T_field[i, sh_idx:, :] = T_surface[sh_idx:, np.newaxis] - gamma_nh * z
    gamma_sh = gamma_neutral if z < 2000.0 else gamma_nh
    T_field[i, :sh_idx, :] = T_surface[:sh_idx, np.newaxis] - gamma_sh * z

# Add wave structure in u and v (so that QGPV has spatial variation)
u_field = np.zeros((nlev, NLAT, NLON))
v_field = np.zeros((nlev, NLAT, NLON))
for i, p in enumerate(plev):
    jet_nh = 40.0 * np.exp(-((ylat - 45.0) / 10.0) ** 2)
    u_field[i] = jet_nh[:, np.newaxis]
    strength = np.exp(-((p - 500.0) / 200.0) ** 2)
    u_field[i] += 5.0 * np.cos(3 * lon_rad[np.newaxis, :]) * np.cos(lat_rad[:, np.newaxis]) * strength
    v_field[i] += 3.0 * np.sin(3 * lon_rad[np.newaxis, :]) * np.cos(lat_rad[:, np.newaxis]) * strength

# -- Diagnose static stability --
print("\n--- Static stability diagnostics ---")
is_problematic, stat_s_diag, stat_n_diag = diagnose_static_stability(
    ylat=ylat, plev=plev, t_field=T_field,
    kmax=KMAX, dz=DZ, threshold=1e-3, verbose=False,
)
print(f"{'z (m)':>8}  {'stat_s (K/m)':>14}  {'stat_n (K/m)':>14}")
print("-" * 42)
for k in range(KMAX):
    flag = " <-- small" if stat_s_diag[k] < 1e-3 or stat_n_diag[k] < 1e-3 else ""
    print(f"{k*DZ:>8.0f}  {stat_s_diag[k]:>14.3e}  {stat_n_diag[k]:>14.3e}{flag}")
print(f"\nIs static stability potentially problematic? {is_problematic}")

# -- NHN22 --
print("\n--- QGFieldNHN22 (hemispheric stat) ---")
qg_nhn22 = QGFieldNHN22(
    xlon=xlon, ylat=ylat, plev=plev,
    u_field=u_field, v_field=v_field, t_field=T_field,
    kmax=KMAX, eq_boundary_index=5, northern_hemisphere_results_only=False,
)
result_nhn22 = qg_nhn22.interpolate_fields()
qgpv_nhn22 = result_nhn22.QGPV
print(f"  max |QGPV| = {np.nanmax(np.abs(qgpv_nhn22)):.3e} s⁻¹")

# -- NH18 --
print("\n--- QGFieldNH18 (global stat — unaffected) ---")
qg_nh18 = QGFieldNH18(
    xlon=xlon, ylat=ylat, plev=plev,
    u_field=u_field, v_field=v_field, t_field=T_field,
    kmax=KMAX, northern_hemisphere_results_only=False,
)
result_nh18 = qg_nh18.interpolate_fields()
qgpv_nh18 = result_nh18.QGPV
print(f"  max |QGPV| = {np.nanmax(np.abs(qgpv_nh18)):.3e} s⁻¹")

nhn22_vs_nh18 = np.nanmax(np.abs(qgpv_nhn22)) / np.nanmax(np.abs(qgpv_nh18))
print(f"\n  NHN22 / NH18 QGPV max = {nhn22_vs_nh18:.1f}×")
if nhn22_vs_nh18 > 2:
    print("  --> NHN22 QGPV is significantly larger, consistent with blow-up.")
