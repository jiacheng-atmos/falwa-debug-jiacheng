# falwa-debug-jiacheng

A small repository to document, reproduce, and resolve a numerical issue found in [FALWA](https://github.com/csyhuang/hn2016_falwa) v2.3.3 and earlier versions.

Reported in upstream issue: [hn2016_falwa #157](https://github.com/csyhuang/hn2016_falwa/issues/157)

---

## The Issue

`QGFieldNHN22.interpolate_fields()` occasionally produces non-physical, extremely large QGPV
values (i.e. QGPV "blow-up") for certain atmospheric time steps (e.g. ERA5 13 September 1986
06 UTC), particularly in the Southern Hemisphere at z = 1000 m.  The companion class
`QGFieldNH18` does **not** suffer from this problem on the same data.

---

## Root Cause

### QGPV stretching term and static stability

The QGPV is computed as

    QGPV = absolute_vorticity + stretch_term

where

    stretch_term = e^(z/H) * d/dz [ e^(-z/H) * (theta - theta_ref) / (d theta_ref / dz) ]

The denominator `d theta_ref / dz` is the **static stability** (here denoted `stat`).  When
`stat → 0`, the term diverges and QGPV blows up.

### Why does this happen in NHN22 but not NH18?

The two classes differ in **how they compute static stability**:

| Class | Static stability used | Source file |
|---|---|---|
| `QGFieldNH18` | **Global**: `0.5 * (stat_s + stat_n)` | `compute_qgpv.f90` |
| `QGFieldNHN22` | **Hemispheric**: `stat_s` (SH) / `stat_n` (NH) | `compute_qgpv_direct_inv.f90` |

Both classes call `_compute_static_stability_func()` in `oopinterface.py`, which:

1. Computes the area-weighted hemispheric mean potential temperature profile
   `t0_s(z)` / `t0_n(z)` at each pressure level.
2. Fits a `scipy.interpolate.UnivariateSpline` to the profile.
3. Evaluates the **derivative** of that spline at the uniform pseudoheight grid to get the
   static stability.

For certain atmospheric states, the spline derivative at low pseudoheight levels (z = 0–2 km)
can become very small (order **1e-5 K m⁻¹** or less) — especially if the hemispheric mean
potential temperature profile is nearly isothermal in the lower boundary layer.  Because
`QGFieldNHN22` uses the hemispheric static stability directly, it is more susceptible to this
near-zero division.  `QGFieldNH18` averages the two hemispheres first, which almost always
yields a larger, better-behaved value.

### Key code locations

* Static stability computation: `QGFieldBase._compute_static_stability_func()` in
  [`oopinterface.py`](https://github.com/csyhuang/hn2016_falwa/blob/master/src/falwa/oopinterface.py)
* NH18 QGPV kernel: [`compute_qgpv.f90`](https://github.com/csyhuang/hn2016_falwa/blob/master/src/falwa/f90_modules/compute_qgpv.f90) — uses `zmav` (global static stability)
* NHN22 QGPV kernel: [`compute_qgpv_direct_inv.f90`](https://github.com/csyhuang/hn2016_falwa/blob/master/src/falwa/f90_modules/compute_qgpv_direct_inv.f90) — uses `stats` / `statn` (hemispheric static stability)

---

## Reproducing and Diagnosing the Issue

### `diagnose_static_stability.py`

Use this script to check whether the static stability of your data is suspiciously small
before running the full QGPV computation.  It inspects the internal spline derivative at
the uniform pseudoheight grid and warns you if any value falls below a threshold.

```python
from diagnose_static_stability import diagnose_static_stability

diagnose_static_stability(
    ylat, plev, t_field,
    kmax=49, dz=1000.0,
    threshold=1e-4
)
```

### `reproduce_issue.py`

Demonstrates the blow-up with a synthetic atmospheric profile that contains a near-isothermal
lower-tropospheric layer in the Southern Hemisphere.  Run it as:

```bash
python reproduce_issue.py
```

The script creates two `QGField` objects with identical inputs — one `QGFieldNHN22` and one
`QGFieldNH18` — and prints the maximum absolute QGPV at z = 1000 m for each.  The NHN22
result will be orders of magnitude larger (indicating the blow-up), while NH18 remains
well-behaved.

---

## Proposed Fix

### `proposed_fix.py`

Provides a drop-in replacement class `QGFieldNHN22GlobalStat` that inherits from
`QGFieldNHN22` but overrides `_compute_qgpv` to use the **global** static stability
`0.5 * (stat_s + stat_n)` — the same approach used by `QGFieldNH18`.

```python
from proposed_fix import QGFieldNHN22GlobalStat

qg = QGFieldNHN22GlobalStat(
    xlon, ylat, plev, u_field, v_field, t_field,
    kmax=49, eq_boundary_index=5
)
qg.interpolate_fields()
qg.compute_reference_states()
```

This simple change eliminates the blow-up while keeping all other NHN22 algorithmic
differences (e.g. the direct-inversion reference-state solver).

### Alternative: floor the static stability

If you prefer to keep the hemispheric static stability and only prevent extreme values, add a
floor before calling `interpolate_fields`:

```python
import numpy as np
from falwa.oopinterface import QGFieldNHN22

class QGFieldNHN22FloorStat(QGFieldNHN22):
    STAT_FLOOR = 1e-4  # K/m — adjust as needed

    def _compute_qgpv(self, interpolated_fields_to_return, return_named_tuple,
                      t0_s, t0_n, stat_s, stat_n):
        stat_s = np.maximum(stat_s, self.STAT_FLOOR)
        stat_n = np.maximum(stat_n, self.STAT_FLOOR)
        return super()._compute_qgpv(
            interpolated_fields_to_return, return_named_tuple,
            t0_s=t0_s, t0_n=t0_n, stat_s=stat_s, stat_n=stat_n)
```

---

## Files in This Repository

| File | Description |
|---|---|
| `diagnose_static_stability.py` | Diagnostic helper — checks static stability before a run |
| `reproduce_issue.py` | Synthetic demo that triggers the blow-up |
| `proposed_fix.py` | Drop-in `QGFieldNHN22GlobalStat` class and usage example |
| `requirements.txt` | Python dependencies |

---

## Environment

```
python >= 3.9
falwa == 2.3.3
numpy
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## References

* Upstream bug report: <https://github.com/csyhuang/hn2016_falwa/issues/157>
* Neal et al. (2022) NHN22: <https://doi.org/10.1029/2021GL097699>
* Nakamura & Huang (2018) NH18: <https://doi.org/10.1126/science.aat0721>
