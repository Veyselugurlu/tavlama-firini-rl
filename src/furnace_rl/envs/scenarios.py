"""Soğuk parti varış takvimi üretimi ve sabit test senaryoları.

Bir senaryo, varış zamanına göre sıralı parti listesidir:
    [{"t_arrival": s, "mass": kg, "speed": m/dk}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate_schedule(rng: np.random.Generator, cfg: dict) -> list[dict]:
    """Poisson varışlı rastgele parti takvimi (bölüm süresi boyunca)."""
    b = cfg["batch"]
    duration = cfg["episode_steps"] * cfg["dt"]
    t = rng.uniform(*b["first_arrival_min"]) * 60.0
    schedule = []
    while t < duration:
        schedule.append({
            "t_arrival": float(t),
            "mass": float(rng.uniform(*b["mass_range"])),
            "speed": float(rng.uniform(*b["speed_range"])),
        })
        gap_min = max(b["min_interval_min"], rng.exponential(b["mean_interval_min"]))
        t += gap_min * 60.0
    return schedule


def special_scenarios() -> dict[str, list[dict]]:
    """Elle tanımlı zor durumlar (docs/03 §4, deney E7)."""
    return {
        "tek_agir_parti": [{"t_arrival": 600.0, "mass": 1500.0, "speed": 0.2}],
        "art_arda_iki_parti": [
            {"t_arrival": 600.0, "mass": 1200.0, "speed": 0.2},
            {"t_arrival": 1080.0, "mass": 1200.0, "speed": 0.2},
        ],
        "dagilim_disi_2500kg": [{"t_arrival": 600.0, "mass": 2500.0, "speed": 0.2}],
        "yavas_bant": [{"t_arrival": 600.0, "mass": 1000.0, "speed": 0.05}],
        "hizli_bant": [{"t_arrival": 600.0, "mass": 1000.0, "speed": 0.4}],
    }


def make_test_set(cfg: dict, n: int = 20, base_seed: int = 10_000) -> list[dict]:
    """Sabit seed'li test senaryoları. Her öğe: {"name", "seed", "schedule"}."""
    out = []
    for i in range(n):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        out.append({"name": f"test_{i:02d}", "seed": seed, "schedule": generate_schedule(rng, cfg)})
    return out


def save_scenarios(scenarios: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")


def load_scenarios(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
