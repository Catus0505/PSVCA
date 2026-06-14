#!/usr/bin/env bash
set -u   # 故意不开 -e:单个 config 失败不能中断整夜的批处理
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

NJOBS="${NJOBS:-36}"
TIER="${TIER:-formal}"

case "${1:-}" in
  tonight)        # gate 优先:weather_pl96 先,再 ETT 全网格
    CONFIGS=(weather_pl96 \
             etth1_pl96 etth1_pl192 etth1_pl336 etth1_pl720 \
             etth2_pl96 etth2_pl192 etth2_pl336 etth2_pl720 \
             ettm1_pl96 ettm1_pl192 ettm1_pl336 ettm1_pl720 \
             ettm2_pl96 ettm2_pl192 ettm2_pl336 ettm2_pl720) ;;
  weather_rest)   # 审过 pl96 之后再跑
    CONFIGS=(weather_pl192 weather_pl336 weather_pl720) ;;
  *) echo "usage: $0 {tonight|weather_rest}"; exit 1 ;;
esac

mkdir -p runs/logs
MASTER="runs/logs/grid_$(date +%Y%m%d_%H%M%S).log"
echo "batch=$1 n_jobs=$NJOBS tier=$TIER n_configs=${#CONFIGS[@]}" | tee "$MASTER"
for cfg in "${CONFIGS[@]}"; do
  echo "===== $(date +%H:%M:%S) START $cfg =====" | tee -a "$MASTER"
  log="runs/logs/${cfg}_${TIER}.log"
  if python scripts/run_reference.py --config "$cfg" --tier "$TIER" --n-jobs "$NJOBS" >"$log" 2>&1; then
    grep -E "dataset|pred_len|effective_B|n_e_certified" "$log" | sed 's/^/  /' | tee -a "$MASTER"
    echo "  $(date +%H:%M:%S) DONE $cfg" | tee -a "$MASTER"
  else
    echo "  $(date +%H:%M:%S) FAILED $cfg (见 $log)" | tee -a "$MASTER"
  fi
done
echo "===== $(date +%H:%M:%S) ALL DONE =====" | tee -a "$MASTER"
