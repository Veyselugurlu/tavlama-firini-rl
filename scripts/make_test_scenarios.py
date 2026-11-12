"""Sabit test senaryolarını üretir: configs/test_scenarios.json, configs/special_scenarios.json."""

from furnace_rl.config import DEFAULT_ENV_CONFIG
from furnace_rl.envs.scenarios import make_test_set, save_scenarios, special_scenarios

if __name__ == "__main__":
    test = make_test_set(DEFAULT_ENV_CONFIG, n=20, base_seed=10_000)
    save_scenarios(test, "configs/test_scenarios.json")
    special = [{"name": name, "seed": 20_000 + i, "schedule": sch}
               for i, (name, sch) in enumerate(special_scenarios().items())]
    save_scenarios(special, "configs/special_scenarios.json")
    n_batches = [len(s["schedule"]) for s in test]
    print(f"test: {len(test)} senaryo, bölüm başına parti {min(n_batches)}–{max(n_batches)}")
    print(f"özel: {[s['name'] for s in special]}")
