"""E8 – önizleme ablasyonu: önizlemeli ve önizlemesiz ajanın doğrulama eğrileri (3 seed ort. ± std).

İki grup da aynı seed ve ayarlarla eğitildi; tek fark gözlemdeki gelecek parti bilgisi (configs/env_no_preview.yaml).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from furnace_rl.plots import plot_learning_curves

GROUPS = {
    "Q-tablosu (önizlemeli)": [f"q_table_seed{s}" for s in range(3)],
    "Q-tablosu (önizlemesiz)": [f"e8q_nopreview_seed{s}" for s in range(3)],
}


def main():
    for key, ylabel in [("return", "bölüm ödülü (doğrulama)"), ("rmse", "RMSE [°C] (doğrulama)")]:
        curves = {}
        for label, runs in GROUPS.items():
            logs = [pd.read_csv(Path("results/models") / r / "evaluations.csv") for r in runs]
            Y = np.stack([l[key].to_numpy() for l in logs])
            curves[label] = (logs[0].step.to_numpy(), Y)
            best = Y.max(1) if key == "return" else Y.min(1)
            print(f"{label:22s} {key}: en iyi {best.mean():.1f} ± {best.std():.1f}   son {Y[:, -1].mean():.1f}")
        plot_learning_curves(curves, f"results/figures/e8_{key}.png", ylabel)


if __name__ == "__main__":
    main()
