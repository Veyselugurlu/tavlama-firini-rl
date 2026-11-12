"""Bölüm metrikleri (docs/04_deney_ve_degerlendirme.md §1)."""

from __future__ import annotations

import numpy as np

RECOVERY_BAND = 5.0   # °C


def recovery_times(err_zone: np.ndarray, t: np.ndarray, arrivals: list[float], band: float = RECOVERY_BAND):
    """Her varış için, bir sonraki varışa kadarki pencerede hatanın ±band dışına son çıktığı ana
    kadar geçen süre [dk]. Pencere sonunda hâlâ bant dışındaysa pencere uzunluğu döner (sansürlü)."""
    out = []
    ends = list(arrivals[1:]) + [t[-1] + (t[1] - t[0])]
    for t0, t1 in zip(arrivals, ends):
        mask = (t >= t0) & (t < t1)
        if not mask.any():
            continue
        outside = np.abs(err_zone[mask]) > band
        if not outside.any():
            out.append(0.0)
        else:
            last = t[mask][np.nonzero(outside)[0][-1]]
            out.append((last - t0) / 60.0)
    return out


def episode_metrics(traj: dict, dt: float, T_max: float = 950.0) -> dict:
    T, T_ref = traj["T_true"], traj["T_ref"]
    err = T - T_ref
    u = traj["u_cmd"]
    arrivals = traj["arrivals"]
    rec = recovery_times(err[:, 0], traj["t"], arrivals)
    return {
        "return": float(np.sum(traj["reward"])),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "iae": float(np.sum(np.abs(err)) * dt / 60.0),          # °C·dk
        "max_drop": float(max(0.0, -err.min())),
        "overshoot": float(max(0.0, err.max())),
        "recovery_min": float(np.mean(rec)) if rec else 0.0,
        "energy_kwh": float(np.sum(traj["energy_kwh"])),
        "smoothness": float(np.mean(np.abs(np.diff(u, axis=0)))) if len(u) > 1 else 0.0,
        "unsafe_min": float(np.sum(np.any(T > T_max, axis=1)) * dt / 60.0),   # T_max üstünde geçen süre
        "n_batches": len(arrivals),
    }
