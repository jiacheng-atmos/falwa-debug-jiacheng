"""
diagnose_static_stability.py
-----------------------------
Diagnostic helper for the FALWA static-stability numerical issue.

Before running ``QGFieldNHN22.interpolate_fields()`` on a large dataset, call
``diagnose_static_stability()`` to check whether the hemispheric static stability
values are suspiciously small at any pseudoheight level.  Very small static stability
(order 1e-5 K/m or below) in the denominator of the QGPV stretching term is the root
cause of the blow-up described in hn2016_falwa issue #157.

References
----------
* hn2016_falwa issue #157: https://github.com/csyhuang/hn2016_falwa/issues/157
"""

import warnings
import numpy as np
from scipy.interpolate import UnivariateSpline


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def compute_hemispheric_static_stability(
    ylat,
    plev,
    t_field,
    kmax=49,
    dz=1000.0,
    scale_height=7000.0,
    cp=1004.0,
    dry_gas_constant=287.0,
    p_ground=1000.0,
):
    """
    Replicate the static-stability computation performed inside
    ``QGFieldBase._compute_static_stability_func()`` and return the resulting
    1-D arrays evaluated at the uniform pseudoheight grid.

    Parameters
    ----------
    ylat : np.ndarray
        Latitude array in degrees, shape (nlat,), ascending order including 0.
    plev : np.ndarray
        Pressure levels in hPa, shape (nlev,), descending order
        (surface first, i.e. plev[0] is the highest pressure).
    t_field : np.ndarray
        Temperature field in K, shape (nlev, nlat, nlon).
    kmax : int
        Number of uniform pseudoheight levels (default 49).
    dz : float
        Pseudoheight grid spacing in metres (default 1000).
    scale_height : float
        Atmospheric scale height in metres (default 7000).
    cp : float
        Specific heat of dry air in J/kg/K (default 1004).
    dry_gas_constant : float
        Dry gas constant in J/kg/K (default 287).
    p_ground : float
        Reference pressure in hPa (default 1000).

    Returns
    -------
    stat_s : np.ndarray  shape (kmax,)
        Southern-hemisphere static stability (dtheta/dz) at uniform z-grid (K/m).
    stat_n : np.ndarray  shape (kmax,)
        Northern-hemisphere static stability (dtheta/dz) at uniform z-grid (K/m).
    height : np.ndarray  shape (kmax,)
        Uniform pseudoheight grid in metres.
    """
    kappa = dry_gas_constant / cp
    plev_to_height = -scale_height * np.log(plev / p_ground)  # (nlev,)

    # Potential temperature
    theta_field = t_field * np.exp(
        kappa * plev_to_height[:, np.newaxis, np.newaxis] / scale_height
    )

    # Equator index (Fortran 1-based used for hemisphere split)
    equator_idx = int(np.argwhere(ylat == 0)[0][0]) + 1  # Fortran 1-based
    jd = ylat.size // 2 + ylat.size % 2  # matches self._jd in QGFieldBase

    clat = np.abs(np.cos(np.deg2rad(ylat)))
    csm = clat[:jd].sum()

    # Hemispheric area-weighted mean theta at each pressure level
    t0_s = (
        np.mean(theta_field[:, :jd, :] * clat[np.newaxis, :jd, np.newaxis], axis=-1)
        .sum(axis=-1)
        / csm
    )
    t0_n = (
        np.mean(theta_field[:, -jd:, :] * clat[np.newaxis, -jd:, np.newaxis], axis=-1)
        .sum(axis=-1)
        / csm
    )

    # Fit splines and take derivatives
    spline_s = UnivariateSpline(x=plev_to_height, y=t0_s)
    spline_n = UnivariateSpline(x=plev_to_height, y=t0_n)
    deriv_s = spline_s.derivative()
    deriv_n = spline_n.derivative()

    height = np.arange(kmax) * dz
    return deriv_s(height), deriv_n(height), height


def diagnose_static_stability(
    ylat,
    plev,
    t_field,
    kmax=49,
    dz=1000.0,
    scale_height=7000.0,
    cp=1004.0,
    dry_gas_constant=287.0,
    p_ground=1000.0,
    threshold=1e-4,
    verbose=True,
):
    """
    Diagnose whether the hemispheric static stability may cause QGPV blow-up
    in ``QGFieldNHN22``.

    A warning is issued (and the function returns ``True``) if *any* static
    stability value falls below ``threshold``.  As a rule of thumb, values
    below ~1e-4 K/m are suspicious; values below ~1e-5 K/m reliably cause
    non-physical QGPV.

    Parameters
    ----------
    ylat, plev, t_field, kmax, dz, scale_height, cp, dry_gas_constant, p_ground
        Same as ``compute_hemispheric_static_stability()``.
    threshold : float
        Minimum acceptable static stability (K/m). Default 1e-4.
    verbose : bool
        If True, print a summary table. Default True.

    Returns
    -------
    is_problematic : bool
        True if any static stability value is below ``threshold``.
    stat_s : np.ndarray  shape (kmax,)
        Southern-hemisphere static stability at the uniform pseudoheight grid.
    stat_n : np.ndarray  shape (kmax,)
        Northern-hemisphere static stability at the uniform pseudoheight grid.
    """
    stat_s, stat_n, height = compute_hemispheric_static_stability(
        ylat=ylat,
        plev=plev,
        t_field=t_field,
        kmax=kmax,
        dz=dz,
        scale_height=scale_height,
        cp=cp,
        dry_gas_constant=dry_gas_constant,
        p_ground=p_ground,
    )

    if verbose:
        print(f"{'z (m)':>10}  {'stat_s (K/m)':>16}  {'stat_n (K/m)':>16}")
        print("-" * 48)
        for k in range(kmax):
            flag_s = " <-- WARNING" if stat_s[k] < threshold else ""
            flag_n = " <-- WARNING" if stat_n[k] < threshold else ""
            print(
                f"{height[k]:>10.0f}  {stat_s[k]:>16.3e}{flag_s}"
                f"  {stat_n[k]:>16.3e}{flag_n}"
            )

    is_problematic = bool(np.any(stat_s < threshold) or np.any(stat_n < threshold))
    if is_problematic:
        warnings.warn(
            f"Static stability is below the threshold {threshold:.1e} K/m at one or more "
            "pseudoheight levels.  QGFieldNHN22 may produce non-physical QGPV values.  "
            "Consider using QGFieldNHN22GlobalStat from proposed_fix.py instead.",
            UserWarning,
            stacklevel=2,
        )
    else:
        if verbose:
            print(
                f"\nAll static stability values are above the threshold "
                f"{threshold:.1e} K/m.  No issue expected."
            )

    return is_problematic, stat_s, stat_n
