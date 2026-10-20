#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EASYR1_DIR:?Set EASYR1_DIR to the EasyR1 source checkout}"
MODEL="${1:?model path}"
TRAIN="${2:?training JSON}"
TEST="${3:?validation JSON}"
OUT="${4:?output directory}"
if [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then echo "Use a fresh output directory" >&2; exit 2; fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
python -m verl.trainer.main "config=$ROOT/scripts/config_vanilla.yaml" \
  "data.train_files=$TRAIN" "data.val_files=$TEST" \
  "worker.actor.model.model_path=$MODEL" worker.rollout.tensor_parallel_size=1 \
  "trainer.save_checkpoint_path=$OUT" worker.reward.reward_type=sequential \
  "worker.reward.reward_function=$EASYR1_DIR/examples/reward_function/r1v.py:compute_score" \
  "trainer.n_gpus_per_node=${NUM_GPUS:-8}" 2>&1 | tee "$OUT/training.log"
