#!/usr/bin/env bash
# Eğitilmiş Q-tablolarından tüm test tablolarını ve şekilleri üretir: bash scripts/run_evaluations.sh [paralel_iş_sayısı]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-.venv/Scripts/python.exe}
P=${1:-6}
M=results/models
RUNS="$M/q_son_seed0 $M/q_son_seed1 $M/q_son_seed2"   # son ayar
mkdir -p results/logs results/eval results/figures

run_one() {
  name="${1%%|*}"; args="${1#*|}"
  # shellcheck disable=SC2086
  if eval "$PY $args" > "results/logs/eval_$name.log" 2>&1; then echo "bitti $name"; else echo "HATA $name"; fi
}
export -f run_one; export PY

# Birbirinden bağımsız değerlendirmeler
{
  echo "e2|scripts/plot_learning_curves.py --runs $RUNS --config configs/q_table_son.yaml --env-config configs/env_w_energy_1.0.yaml --out results/figures/e2"
  echo "e3|scripts/compare_e3.py --runs $RUNS --tag e3"
  echo "e8|scripts/compare_e3.py --runs $M/e8q_nopreview_seed0 $M/e8q_nopreview_seed1 $M/e8q_nopreview_seed2 --env-config configs/env_no_preview.yaml --tag e8_nopreview --label 'Q-tablosu (önizlemesiz)'"
  echo "e8_egri|scripts/plot_e8.py"
  echo "e4|scripts/plot_e4.py"
  echo "e4_titresim|scripts/plot_e4_smooth.py"
  echo "e5|scripts/plot_e5.py"
  echo "e6_e7|scripts/e6_e7_robustness.py --runs $RUNS"
  echo "animasyon|scripts/make_gif.py --scenario test_12 --run $M/q_son_seed0"
  echo "politika|scripts/plot_policy.py --run $M/q_son_seed0 --zone 0 --out results/figures/politika_b1.png"
} | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$1"' _ {}

# E3 yörüngelerine bağlı olduğu için en sonda
run_one "odul_analizi|scripts/reward_analysis.py"
echo DEGERLENDIRME_BITTI
