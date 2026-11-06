#!/usr/bin/env bash
# Eğitilmiş modellerden tüm test tablolarını ve şekilleri üretir: bash scripts/run_evaluations.sh [paralel_iş_sayısı]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-.venv/Scripts/python.exe}
P=${1:-6}
M=results/models
mkdir -p results/logs results/eval results/figures

run_one() {
  name="${1%%|*}"; args="${1#*|}"
  # shellcheck disable=SC2086
  if eval "$PY $args" > "results/logs/eval_$name.log" 2>&1; then echo "bitti $name"; else echo "HATA $name"; fi
}
export -f run_one; export PY

# Birbirinden bağımsız değerlendirmeler
{
  echo "e2|scripts/plot_learning_curves.py --runs $M/mbrl_seed0 $M/mbrl_seed1 $M/mbrl_seed2 --out results/figures/e2"
  echo "e3|scripts/compare_e3.py --runs $M/mbrl_seed0 $M/mbrl_seed1 $M/mbrl_seed2 --tag e3 --label MB-RL"
  echo "e8|scripts/compare_e3.py --runs $M/e8_nopreview_seed0 $M/e8_nopreview_seed1 $M/e8_nopreview_seed2 --env-config configs/env_no_preview.yaml --tag e8_nopreview --label 'MB-RL önizlemesiz'"
  echo "e8_egri|scripts/plot_e8.py"
  echo "e4|scripts/plot_e4.py"
  echo "e4_titresim|scripts/plot_e4_smooth.py"
  echo "e5|scripts/plot_e5.py"
  echo "e6_e7|scripts/e6_e7_robustness.py --runs $M/mbrl_seed0 $M/mbrl_seed1 $M/mbrl_seed2"
  echo "yontemler|scripts/compare_methods.py"
} | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$1"' _ {}

# E3 yörüngelerine bağlı olduğu için en sonda
run_one "odul_analizi|scripts/reward_analysis.py"
echo DEGERLENDIRME_BITTI
