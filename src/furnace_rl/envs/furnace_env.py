"""Çok bölgeli sürekli tavlama fırını – gymnasium ortamı.

Denklemler docs/03_simulator_tasarimi.md §2, MDP tanımı docs/02_mdp_tasarimi.md ile birebir eşleşir.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from furnace_rl.config import DEFAULT_ENV_CONFIG, merge_config
from furnace_rl.envs.scenarios import generate_schedule


class FurnaceEnv(gym.Env):
    """Eylem: her bölge için a ∈ [-1, 1].
        action_mode = "incremental" (varsayılan): u_t = clip(u_{t−1} + du_max · a, 0, 1)
        action_mode = "absolute":                 u_t = (a + 1) / 2

    Gözlem (N bölge, d ölü zaman adımı, boyut 6N + 3 + dN):
        [ (T_meas − T_ref)/s | dT/dt | T_ref/s | u_prev | bölgedeki kütle | m_next, v_next, t_arrival
          | u_eff (güç ölçeri) | ölü zamanda bekleyen d komut ]
    Son iki blok gecikmeli sistemi Markov yapar: ajan "yoldaki" ısıyı görür (docs/02 §1.1).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None, scenario: list[dict] | None = None):
        super().__init__()
        self.cfg = merge_config(DEFAULT_ENV_CONFIG, config or {})
        c = self.cfg
        self.n = int(c["n_zones"])
        self.dt = float(c["dt"])
        self.substeps = int(c["substeps"])
        self.dt_phys = self.dt / self.substeps
        self.T_ref = self._per_zone(c["T_ref"])
        self.P_max = self._per_zone(c["P_max"])
        self.C = self._per_zone(c["C"])
        self.h = self._per_zone(c["h_loss"])
        self.k = float(c["k_cond"])
        self.zone_length = float(c["zone_length"])
        self.fixed_scenario = scenario

        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), dtype=np.float32)
        clip = float(c["obs"]["clip"])
        n, d = self.n, int(c["dead_steps"])
        obs_dim = 6 * n + 3 + d * n
        self.observation_space = spaces.Box(-clip, clip, (obs_dim,), dtype=np.float32)
        # Gözlem dilimleri: PID ve analiz kodu bunları kullanır.
        self.obs_slices = {
            "T_err": slice(0, n), "dT": slice(n, 2 * n), "T_ref": slice(2 * n, 3 * n),
            "u_prev": slice(3 * n, 4 * n), "load": slice(4 * n, 5 * n), "preview": slice(5 * n, 5 * n + 3),
            "u_eff": slice(5 * n + 3, 6 * n + 3), "pending": slice(6 * n + 3, obs_dim),
        }
        self.E_max = float(self.P_max.sum() * self.dt / 3.6e6)

    # ------------------------------------------------------------------ yardımcılar
    def _per_zone(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        return np.full(self.n, float(arr)) if arr.ndim == 0 else arr.copy()

    def _conduction(self, T: np.ndarray) -> np.ndarray:
        """Komşu bölgelerden gelen net ısı [W]."""
        flow = self.k * np.diff(T)          # i → i+1 yönünde pozitifse i+1 daha sıcak
        q = np.zeros_like(T)
        q[:-1] += flow
        q[1:] -= flow
        return q

    def action_to_power(self, action) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        if self.cfg["action_mode"] == "incremental":
            return np.clip(self.u_prev + self.cfg["du_max"] * a, 0.0, 1.0)
        return (a + 1.0) / 2.0

    def power_to_action(self, u, u_prev) -> np.ndarray:
        """İstenen güç oranını bu ortamın eylem biçimine çevirir (PID ve sabit politikalar için).
        Artımlı modda değişim ±du_max ile sınırlanır; bu, RL ile aynı aktüatör hız sınırıdır."""
        u = np.asarray(u, dtype=np.float64)
        if self.cfg["action_mode"] == "incremental":
            a = (u - np.asarray(u_prev, dtype=np.float64)) / self.cfg["du_max"]
        else:
            a = 2.0 * u - 1.0
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def steady_state_power(self, T: np.ndarray | None = None) -> np.ndarray:
        """Yüksüz dengede T sıcaklığını tutmak için gereken güç oranı."""
        T = self.T_ref if T is None else T
        u = (self.h * (T - self.T_amb) - self._conduction(T)) / self.P_max
        return np.clip(u, 0.0, 1.0)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        c, rng = self.cfg, self.np_random

        self.t = 0.0
        self.step_count = 0
        self.T_amb = c["T_amb"] + rng.uniform(-c["T_amb_jitter"], c["T_amb_jitter"])
        if "T_init" in options:
            self.T = self._per_zone(options["T_init"])
        else:
            self.T = self.T_ref + rng.uniform(-c["T_init_jitter"], c["T_init_jitter"], self.n)

        u_ss = self.steady_state_power()
        self.u_eff = u_ss.copy()
        self.u_prev = u_ss.copy()
        self.act_buf = deque([u_ss.copy() for _ in range(int(c["dead_steps"]))])

        self.T_tc = self.T.copy()
        self.T_meas = self._measure()
        self.T_filt = self.T_meas.copy()
        self.dT = np.zeros(self.n)

        scenario = options.get("scenario", self.fixed_scenario)
        if scenario is None:
            scenario = generate_schedule(rng, c)
        self.pending = sorted((dict(b) for b in scenario), key=lambda b: b["t_arrival"])
        self.arrivals: list[float] = []
        # Fırındaki partiler: konum [m], sıcaklık [°C], kütle [kg], hız [m/s]
        self.b_x = np.zeros(0)
        self.b_T = np.zeros(0)
        self.b_m = np.zeros(0)
        self.b_v = np.zeros(0)
        self.exited: list[dict] = []

        return self._obs(), self._info(energy_kwh=0.0, terms=None, u_cmd=self.u_prev)

    def step(self, action):
        c, b = self.cfg, self.cfg["batch"]
        u_cmd = self.action_to_power(action)
        if self.act_buf:
            self.act_buf.append(u_cmd)
            u_del = self.act_buf.popleft()
        else:
            u_del = u_cmd

        # Bu adımda varan partileri fırının girişine koy.
        while self.pending and self.pending[0]["t_arrival"] <= self.t:
            p = self.pending.pop(0)
            self.arrivals.append(p["t_arrival"])
            self.b_x = np.append(self.b_x, 0.0)
            self.b_T = np.append(self.b_T, b["T_in"])
            self.b_m = np.append(self.b_m, p["mass"])
            self.b_v = np.append(self.b_v, p["speed"] / 60.0)

        energy_J = 0.0
        dtp = self.dt_phys
        for _ in range(self.substeps):
            self.u_eff += dtp / c["tau_act"] * (u_del - self.u_eff)
            P = self.P_max * self.u_eff

            q_load = np.zeros(self.n)
            if self.b_x.size:
                zone = np.minimum((self.b_x // self.zone_length).astype(int), self.n - 1)
                Q = b["hA_per_kg"] * self.b_m * (self.T[zone] - self.b_T)
                np.add.at(q_load, zone, Q)
                self.b_T += dtp * Q / (self.b_m * b["c_m"])
                self.b_x += self.b_v * dtp

            q_loss = self.h * (self.T - self.T_amb)
            self.T += dtp / self.C * (P - q_loss + self._conduction(self.T) - q_load)
            self.T_tc += dtp / c["tau_tc"] * (self.T - self.T_tc)
            energy_J += P.sum() * dtp

            if self.b_x.size:
                out = self.b_x >= self.n * self.zone_length
                if out.any():
                    for T_exit, m in zip(self.b_T[out], self.b_m[out]):
                        self.exited.append({"t": self.t, "T_exit": float(T_exit), "mass": float(m)})
                    keep = ~out
                    self.b_x, self.b_T, self.b_m, self.b_v = (
                        self.b_x[keep], self.b_T[keep], self.b_m[keep], self.b_v[keep])

        self.t += self.dt
        self.step_count += 1

        T_filt_prev = self.T_filt.copy()
        self.T_meas = self._measure()
        self.T_filt += c["deriv_filter"] * (self.T_meas - self.T_filt)
        self.dT = (self.T_filt - T_filt_prev) / self.dt * 60.0     # °C/dk

        energy_kwh = energy_J / 3.6e6
        reward, terms = self._reward(u_cmd, energy_kwh)
        self.u_prev = u_cmd

        # Güvenlik ihlali bölümü bitirmez: ödül hep negatif olduğundan erken bitiş, kalan adımların
        # maliyetinden kaçmak için istismar edilebilir (bkz. docs/02 §5). İhlal her adımda cezalandırılır.
        terminated = False
        truncated = self.step_count >= c["episode_steps"]
        return self._obs(), reward, terminated, truncated, self._info(energy_kwh, terms, u_cmd)

    # ------------------------------------------------------------------ iç hesaplar
    def _measure(self) -> np.ndarray:
        return self.T_tc + self.np_random.normal(0.0, self.cfg["noise_std"], self.n)

    def _reward(self, u_cmd: np.ndarray, energy_kwh: float):
        r = self.cfg["reward"]
        e = (self.T - self.T_ref) / r["dT_scale"]
        track = float(np.mean(e ** 2))
        energy = energy_kwh / self.E_max
        smooth = float(np.mean(np.abs(u_cmd - self.u_prev)))
        unsafe = bool(np.any(self.T > self.cfg["T_max"]))
        safety = r["safety_penalty"] if unsafe else 0.0
        reward = -(r["w_track"] * track + r["w_energy"] * energy + r["w_smooth"] * smooth) - safety
        terms = {"track": track, "energy": energy, "smooth": smooth, "safety": safety}
        return float(reward), terms

    def zone_load(self) -> np.ndarray:
        load = np.zeros(self.n)
        if self.b_x.size:
            zone = np.minimum((self.b_x // self.zone_length).astype(int), self.n - 1)
            np.add.at(load, zone, self.b_m)
        return load

    def next_batch(self) -> dict | None:
        """Önizleme ufku içinde duyurulmuş ilk parti."""
        if self.pending and self.pending[0]["t_arrival"] - self.t <= self.cfg["batch"]["preview_min"] * 60.0:
            return self.pending[0]
        return None

    def _obs(self) -> np.ndarray:
        o = self.cfg["obs"]
        preview = np.array([0.0, 0.0, 1.0])
        nb = self.next_batch() if self.cfg["use_preview"] else None
        if nb is not None:
            horizon = self.cfg["batch"]["preview_min"] * 60.0
            preview = np.array([
                nb["mass"] / o["mass_scale"],
                nb["speed"] / o["speed_scale"],
                max(nb["t_arrival"] - self.t, 0.0) / horizon,
            ])
        obs = np.concatenate([
            (self.T_meas - self.T_ref) / o["T_err_scale"],
            self.dT / o["dT_scale"],
            self.T_ref / o["T_ref_scale"],
            self.u_prev,
            self.zone_load() / o["mass_scale"],
            preview,
            self.u_eff,
            *self.act_buf,
        ])
        return np.clip(obs, -o["clip"], o["clip"]).astype(np.float32)

    def _info(self, energy_kwh: float, terms: dict | None, u_cmd: np.ndarray) -> dict:
        return {
            "t": self.t,
            "T_true": self.T.copy(),
            "T_meas": self.T_meas.copy(),
            "T_ref": self.T_ref.copy(),
            "u_cmd": np.asarray(u_cmd).copy(),
            "u_eff": self.u_eff.copy(),
            "energy_kwh": energy_kwh,
            "reward_terms": terms,
            "zone_load": self.zone_load(),
            "arrivals": list(self.arrivals),
        }
