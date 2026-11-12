#!/usr/bin/env bash
# Tüm Q-tablosu eğitimlerini paralel çalıştırır: bash scripts/run_experiments.sh [paralel_iş_sayısı]
# q_son_seed*: son ayar (configs/q_table_son.yaml); q_table_seed*: E4/E5/E8 taramalarının referansı.
# Her iş bir satırdır; çıktı results/logs/<ad>.log dosyasına yazılır, bitmiş işler atlanır.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=${PY:-.venv/Scripts/python.exe}
P=${1:-6}
mkdir -p results/logs
RATE_GRID='bins={"error":[-100,-60,-30,-15,-8,-3,3,8,20],"rate":[-15,-8,-4,-2,0,2,4,8,15],"load":[0.05,0.5],"coming":[0.05,0.4]}'
jobs_list() {
  for s in 0 1 2; do
    echo "q_son_seed$s|--seed $s --config configs/q_table_son.yaml --env-config configs/env_w_energy_1.0.yaml --name q_son_seed$s"
    echo "q_table_seed$s|--seed $s --name q_table_seed$s"
    echo "e8q_nopreview_seed$s|--seed $s --env-config configs/env_no_preview.yaml --name e8q_nopreview_seed$s"
    echo "e5q_levels_seed$s|--seed $s --name e5q_levels_seed$s --set 'levels=[0.0,0.2,0.3,0.4,0.6,1.0]'"
    echo "e5q_levels_w1_seed$s|--seed $s --env-config configs/env_w_energy_1.0.yaml --name e5q_levels_w1_seed$s --set 'levels=[0.0,0.2,0.3,0.4,0.6,1.0]'"
    echo "e5q_alphamin_seed$s|--seed $s --name e5q_alphamin_seed$s --set alpha_min=0.05"
    echo "e5q_eps_seed$s|--seed $s --name e5q_eps_seed$s --set eps_start=0.5 eps_end=0.05"
    echo "e5q_errorrate_seed$s|--seed $s --name e5q_errorrate_seed$s --set 'state_mode=\"error_rate\"' '$RATE_GRID'"
    for w in 0.0 1.0; do
      echo "e4q_wenergy${w}_seed$s|--seed $s --env-config configs/env_w_energy_$w.yaml --name e4q_wenergy${w}_seed$s"
    done
    echo "e4q_wsmooth0.0_seed$s|--seed $s --env-config configs/env_w_smooth_0.0.yaml --name e4q_wsmooth0.0_seed$s"
  done
}
run_one() {
  name="${1%%|*}"; args="${1#*|}"
  log="results/logs/$name.log"
  grep -q "^DONE" "$log" 2>/dev/null && { echo "atlandı $name"; return 0; }
  echo "başladı $name"
  if eval "$PY -m furnace_rl.agents.q_table $args" > "$log" 2>&1; then echo "DONE" >> "$log"; echo "bitti $name"; else echo "HATA $name"; fi
}
export -f run_one; export PY
jobs_list | xargs -d '\n' -P "$P" -I{} bash -c 'run_one "$1"' _ {}
echo TUM_ISLER_BITTI
