#!/usr/bin/env bash
# Evaluate every tokenizer in benchmark/tokenizers.txt on the African benchmark, in parallel.
#   ./benchmark/run_evals.sh [JOBS]          (default 6)
# Needs: .venv with requirements.txt; HF token with read access to the private benchmark
# (and to gated tokenizer repos). Results: $DATA_DIR/results/evaluations/, logs: logs/eval/.
# Remote code in model repos is never run (no --trust-remote-code): such tokenizers fail
# and are listed at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
JOBS="${1:-6}"
export HF_REPO_BENCHMARK="${HF_REPO_BENCHMARK:-AfriSpeech/african-tokenizer-benchmark}"
export HF_REPO_RESULTS="${HF_REPO_RESULTS:-AfriSpeech/african-tokenizer-results}"
export DATA_DIR="${DATA_DIR:-$PWD/data}"
mkdir -p logs/eval

# Fetch the benchmark once, before the parallel jobs share it.
.venv/bin/python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ['HF_REPO_BENCHMARK'], repo_type='dataset',
                  local_dir=os.path.join(os.environ['DATA_DIR'], 'benchmark'))
print('benchmark ready')"

grep -vE '^\s*(#|$)' benchmark/tokenizers.txt | xargs -P "$JOBS" -I{} bash -c '
  name="{}"; log="logs/eval/$(echo "$name" | tr "/" "__").log"
  if .venv/bin/python tokenizer_evaluate.py "$name" > "$log" 2>&1; then
    echo "ok     $name"
  else
    echo "FAILED $name  (see $log)"
  fi'
echo "done"
