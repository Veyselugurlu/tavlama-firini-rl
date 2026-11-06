#!/usr/bin/env bash
# Tüm eğitimleri paralel çalıştırır: bash scripts/run_experiments.sh [paralel_iş_sayısı]
# Her iş bir satırdır; çıktı results/logs/<ad>.log dosyasına yazılır, bitmiş işler atlanır.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-.venv/Scripts/python.exe}
P=${1:-6}
mkdir -p results/logs
jobs_list() {
  for s in 0 1 2; do
    echo "mbrl_seed$s|-m furnace_rl.agents.mbrl --seed $s --name mbrl_seed$s"
    echo "e8_nopreview_seed$s|-m furnace_rl.agents.mbrl --seed $s --env-config configs/env_no_preview.yaml --name e8_nopreview_seed$s"
    for w in 0.0 0.1 0.5 1.0; do
      echo "e4_wenergy${w}_seed$s|-m furnace_rl.agents.mbrl --seed $s --env-config configs/env_w_energy_$w.yaml --name e4_wenergy${w}_seed$s"
    done
    echo "e4_wsmooth0.0_seed$s|-m furnace_rl.agents.mbrl --seed $s --env-config configs/env_w_smooth_0.0.yaml --name e4_wsmooth0.0_seed$s"
    for kv in horizon=30 horizon=120 replan=3 cem_pop=50 bias_filter=0.0 fresh_rho=0.99 fresh_rho=0.999 lam=1.0; do
      n=${kv/=/}
      extra=""
      [ "$kv" = "replan=3" ] && extra=""
      echo "e5_${n}_seed$s|-m furnace_rl.agents.mbrl --seed $s --set $kv --name e5_${n}_seed$s"
    done
    echo "lspi_seed$s|-m furnace_rl.agents.lspi --seed $s --name lspi_seed$s"
    echo "q_tile_seed$s|-m furnace_rl.agents.q_tile --seed $s --name q_tile_seed$s"
    echo "ars_seed$s|-m furnace_rl.agents.ars --seed $s --workers 2 --name ars_seed$s"
  done
}
run_one() {
  name="${1%%|*}"; args="${1#*|}"
  log="results/logs/$name.log"
  grep -q "^DONE" "$log" 2>/dev/null && { echo "atlandı $name"; return 0; }
  echo "başladı $name"
  # shellcheck disable=SC2086
  if $PY $args > "$log" 2>&1; then echo "DONE" >> "$log"; echo "bitti $name"; else echo "HATA $name"; fi
}
export -f run_one; export PY
jobs_list | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$1"' _ {}
echo TUM_ISLER_BITTI
