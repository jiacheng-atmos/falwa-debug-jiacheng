"""
proposed_fix.py
---------------
Provides two drop-in replacement classes for ``QGFieldNHN22`` that resolve
the QGPV blow-up caused by near-zero hemispheric static stability.

Classes
-------
QGFieldNHN22GlobalStat
    Uses the **global** static stability ``0.5 * (stat_s + stat_n)`` in the
    QGPV computation and reference-state initialisation — the same approach
    as ``QGFieldNH18``.  This is the recommended fix.

QGFieldNHN22FloorStat
    Keeps the hemispheric static stability but clamps it from below with a
    configurable floor value.  Suitable when the issue is near-zero but
    still positive static stability.  Note: if ``stat_s`` or ``stat_n`` are
    already negative (superadiabatic layer), a floor may change the sign of
    the stretching term.  In that case ``QGFieldNHN22GlobalStat`` is safer.

Both classes are otherwise identical to ``QGFieldNHN22`` and keep the
NHN22 direct-inversion reference-state solver.

Usage
-----
    from proposed_fix import QGFieldNHN22GlobalStat

    qg = QGFieldNHN22GlobalStat(
        xlon, ylat, plev, u_field, v_field, t_field,
        kmax=49, eq_boundary_index=5
    )
    qg.interpolate_fields()
    qg.compute_reference_states()
    qg.compute_lwa_and_barotropic_fluxes()

References
----------
* hn2016_falwa issue #157: https://github.com/csyhuang/hn2016_falwa/issues/157
"""

import numpy as np

from falwa.oopinterface import QGFieldNHN22


# ---------------------------------------------------------------------------
# Fix 1 — global static stability (recommended)
# ---------------------------------------------------------------------------

class QGFieldNHN22GlobalStat(QGFieldNHN22):
    """
    ``QGFieldNHN22`` variant that uses the **global** static stability
    ``0.5 * (stat_s + stat_n)`` for the QGPV computation.

    This exactly mirrors the approach in ``QGFieldNH18`` and prevents the
    blow-up that occurs when one hemispheric average dips close to zero.

    All other NHN22 algorithms (direct-inversion reference state, etc.)
    are preserved.
    """

    def _compute_qgpv(
        self,
        interpolated_fields_to_return,
        return_named_tuple,
        t0_s,
        t0_n,
        stat_s,
        stat_n,
    ):
        stat_global = 0.5 * (stat_s + stat_n)
        t0_global = 0.5 * (t0_s + t0_n)
        return super()._compute_qgpv(
            interpolated_fields_to_return,
            return_named_tuple,
            t0_s=t0_global,
            t0_n=t0_global,
            stat_s=stat_global,
            stat_n=stat_global,
        )


# ---------------------------------------------------------------------------
# Fix 2 — floor the hemispheric static stability
# ---------------------------------------------------------------------------

class QGFieldNHN22FloorStat(QGFieldNHN22):
    """
    ``QGFieldNHN22`` variant that clamps the hemispheric static stability
    from below with a configurable floor value.

    This preserves the hemispheric asymmetry in the static stability while
    preventing extreme near-zero divisions in the QGPV stretching term.

    .. note::
        If the static stability is already **negative** (superadiabatic
        atmosphere), flooring to a positive value changes the sign of the
        QGPV stretching term, which may introduce unphysical results.
        Use ``QGFieldNHN22GlobalStat`` in that case.

    Parameters
    ----------
    stat_floor : float, optional
        Minimum allowed static stability in K/m.  Default is 1e-4 K/m.
        Values below ~1e-5 K/m reliably cause non-physical QGPV.
    *args, **kwargs
        All other arguments are passed through to ``QGFieldNHN22.__init__``.
    """

    DEFAULT_STAT_FLOOR = 1e-4  # K/m

    def __init__(self, *args, stat_floor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stat_floor = stat_floor if stat_floor is not None else self.DEFAULT_STAT_FLOOR

    def _compute_qgpv(
        self,
        interpolated_fields_to_return,
        return_named_tuple,
        t0_s,
        t0_n,
        stat_s,
        stat_n,
    ):
        stat_s = np.maximum(stat_s, self._stat_floor)
        stat_n = np.maximum(stat_n, self._stat_floor)
        return super()._compute_qgpv(
            interpolated_fields_to_return,
            return_named_tuple,
            t0_s=t0_s,
            t0_n=t0_n,
            stat_s=stat_s,
            stat_n=stat_n,
        )


# ---------------------------------------------------------------------------
# Demo — run as __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from reproduce_issue import (
        xlon, ylat, plev, u_field, v_field, T_field, KMAX,
        qgpv_nhn22, qgpv_nh18,
    )

    print("\n" + "=" * 60)
    print("Fix 1: QGFieldNHN22GlobalStat (recommended)")
    print("=" * 60)
    qg_fixed = QGFieldNHN22GlobalStat(
        xlon=xlon, ylat=ylat, plev=plev,
        u_field=u_field, v_field=v_field, t_field=T_field,
        kmax=KMAX, eq_boundary_index=5, northern_hemisphere_results_only=False,
    )
    result_fixed = qg_fixed.interpolate_fields()
    qgpv_fixed = result_fixed.QGPV

    print(f"\n  max |QGPV| (all levels) = {np.nanmax(np.abs(qgpv_fixed)):.3e} s⁻¹")
    print(f"  Any inf? {np.any(np.isinf(qgpv_fixed))}")
    print(f"  Any NaN? {np.any(np.isnan(qgpv_fixed))}")

    print("\n" + "=" * 60)
    print("Fix 2: QGFieldNHN22FloorStat (floor = 1e-4 K/m)")
    print("=" * 60)
    print(
        "\n  Note: if static stability is already negative, the floor changes\n"
        "  its sign and may produce unphysical results.  Prefer Fix 1."
    )
    qg_floor = QGFieldNHN22FloorStat(
        xlon=xlon, ylat=ylat, plev=plev,
        u_field=u_field, v_field=v_field, t_field=T_field,
        kmax=KMAX, eq_boundary_index=5, northern_hemisphere_results_only=False,
        stat_floor=1e-4,
    )
    result_floor = qg_floor.interpolate_fields()
    qgpv_floor = result_floor.QGPV

    print(f"\n  max |QGPV| (all levels) = {np.nanmax(np.abs(qgpv_floor)):.3e} s⁻¹")
    print(f"  Any inf? {np.any(np.isinf(qgpv_floor))}")
    print(f"  Any NaN? {np.any(np.isnan(qgpv_floor))}")

    print("\n" + "=" * 60)
    print("Comparison — max |QGPV| over all levels")
    print("=" * 60)
    print(f"  NHN22 (original):  {np.nanmax(np.abs(qgpv_nhn22)):.3e} s⁻¹  <-- potentially blown up")
    print(f"  NH18 (reference):  {np.nanmax(np.abs(qgpv_nh18)):.3e} s⁻¹")
    print(f"  GlobalStat fix:    {np.nanmax(np.abs(qgpv_fixed)):.3e} s⁻¹  <-- recommended fix")
    print(f"  FloorStat fix:     {np.nanmax(np.abs(qgpv_floor)):.3e} s⁻¹")
