"""Bölge başına ayrık PID baseline (docs/04_deney_ve_degerlendirme.md §2).

- Türev ölçüm üzerinden ve alçak geçiren filtreyle alınır (set-point kick ve gürültüye karşı).
- Anti-windup: çıkış doymuşken hata doymayı derinleştiriyorsa integral durdurulur.
- Bumpless başlangıç: ilk çağrıda integral, çıkış ortamın mevcut gücüne eşit olacak şekilde kurulur.
"""

from __future__ import annotations

import numpy as np

from furnace_rl.config import load_yaml


class PIDController:
    def __init__(self, kp, ti, td, dt: float, n: int, d_filter: float = 0.2):
        self.kp = np.broadcast_to(np.asarray(kp, float), (n,)).copy()
        self.ti = np.broadcast_to(np.asarray(ti, float), (n,)).copy()   # integral süresi [s]
        self.td = np.broadcast_to(np.asarray(td, float), (n,)).copy()   # türev süresi [s]
        self.dt = dt
        self.n = n
        self.d_filter = d_filter
        self.reset()

    def reset(self):
        self.integral = None          # Ki·∫e katkısı (çıkış birimi)
        self.prev_meas = None
        self.deriv = np.zeros(self.n)

    def __call__(self, error: np.ndarray, meas: np.ndarray, u_init: np.ndarray) -> np.ndarray:
        p = self.kp * error
        if self.prev_meas is None:
            self.prev_meas = meas.copy()
        raw = -(meas - self.prev_meas) / self.dt
        self.deriv += self.d_filter * (raw - self.deriv)
        self.prev_meas = meas.copy()
        d = self.kp * self.td * self.deriv

        if self.integral is None:
            self.integral = np.asarray(u_init, float) - p - d

        ki = np.where(self.ti > 0, self.kp / np.maximum(self.ti, 1e-9), 0.0)
        candidate = self.integral + ki * error * self.dt
        u_unsat = p + candidate + d
        winding = ((u_unsat > 1.0) & (error > 0)) | ((u_unsat < 0.0) & (error < 0))
        self.integral = np.where(winding, self.integral, candidate)
        return np.clip(p + self.integral + d, 0.0, 1.0)


class PIDPolicy:
    """RL politikalarıyla aynı `predict` imzası: değerlendirme kodu hepsi için ortak."""

    def __init__(self, env, kp=0.02, ti=600.0, td=0.0, d_filter: float = 0.2):
        base = env.unwrapped
        self.env = base
        self.slices = base.obs_slices
        self.err_scale = base.cfg["obs"]["T_err_scale"]
        self.ctrl = PIDController(kp, ti, td, base.dt, base.n, d_filter)

    @classmethod
    def from_yaml(cls, env, path):
        return cls(env, **load_yaml(path))

    def reset(self):
        self.ctrl.reset()

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        obs = np.asarray(obs, float)
        meas = obs[self.slices["T_err"]] * self.err_scale      # T_meas − T_ref [°C]
        u_prev = obs[self.slices["u_prev"]]
        u = self.ctrl(-meas, meas, u_init=u_prev)
        return self.env.power_to_action(u, u_prev), None
