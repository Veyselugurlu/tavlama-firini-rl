"""Varsayılan ortam ayarları ve YAML ile üzerine yazma.

Fiziksel anlamları ve kaynakları: docs/03_simulator_tasarimi.md §3.
Deney dosyaları (configs/*.yaml) yalnızca değişen anahtarları içerir.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_ENV_CONFIG: dict = {
    "n_zones": 3,
    "dt": 10.0,                 # kontrol adımı [s]
    "substeps": 5,              # fizik alt adımı = dt / substeps
    "episode_steps": 720,       # 2 saat
    "T_ref": [750.0, 850.0, 850.0],   # hedef profil [°C]
    "T_amb": 25.0,
    "T_amb_jitter": 10.0,       # bölüm başında ± [°C]
    "T_init_jitter": 10.0,      # bölge başlangıç sıcaklığı ± [°C]
    "T_max": 950.0,             # güvenlik sınırı [°C]
    "P_max": [250e3, 200e3, 160e3],   # bölge maksimum gücü [W]
    "C": 1.0e6,                 # bölge ısıl kapasitesi [J/K]
    "h_loss": 75.0,             # duvar kaybı [W/K]
    "k_cond": 30.0,             # bölgeler arası iletim [W/K]
    "tau_act": 150.0,           # aktüatör zaman sabiti [s]
    "dead_steps": 3,            # aktüatör ölü zamanı [kontrol adımı]
    "tau_tc": 20.0,             # termokupl zaman sabiti [s]
    "noise_std": 1.0,           # termokupl gürültüsü [°C]
    "deriv_filter": 0.3,        # türev için EMA katsayısı
    "zone_length": 5.0,         # [m]
    "action_mode": "incremental",   # "incremental": eylem = güç değişimi, "absolute": eylem = güç
    "du_max": 0.1,              # artımlı modda adım başı en büyük güç değişimi
    "use_preview": True,        # gelecek parti bilgisi gözlemde mi (E8 ablasyonu)
    "batch": {
        "mass_range": [300.0, 1500.0],    # [kg]
        "speed_range": [0.1, 0.3],        # [m/dk]
        "T_in": 25.0,                     # giriş sıcaklığı [°C]
        "c_m": 600.0,                     # özgül ısı [J/(kg·K)]
        "hA_per_kg": 0.3,                 # ısı transferi [W/(K·kg)]
        "mean_interval_min": 35.0,        # Poisson varış aralığı ortalaması
        "min_interval_min": 15.0,         # üretim planı: partiler arası en az süre
        "first_arrival_min": [5.0, 20.0],
        "preview_min": 10.0,              # varıştan bu kadar önce duyurulur
    },
    "reward": {
        "w_track": 1.0,
        "w_energy": 0.2,
        "w_smooth": 0.05,
        "dT_scale": 20.0,           # 10 °C hata → adım başı 0.25 maliyet
        "safety_penalty": 2.0,      # T_max üstünde geçen her adım için
    },
    "obs": {
        "T_err_scale": 25.0,      # (T_meas − T_ref) / ölçek; ±250 °C'de kırpılır
        "dT_scale": 10.0,         # [°C/dk] / ölçek
        "T_ref_scale": 1000.0,
        "mass_scale": 1500.0,
        "speed_scale": 0.3,
        "clip": 10.0,
    },
}


def merge_config(base: dict, override: dict) -> dict:
    """`override` anahtarlarını `base`in derin kopyası üzerine yazar."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_config(path: str | Path | None = None, **overrides) -> dict:
    """Varsayılan + YAML dosyası + anahtar kelime üzerine yazmaları."""
    cfg = DEFAULT_ENV_CONFIG
    if path is not None:
        cfg = merge_config(cfg, load_yaml(path))
    return merge_config(cfg, overrides)
