#!/usr/bin/env bash
# Run DQN OPE, extended comparison, and poster LaTeX export using the project venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/cs224r/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing project venv. From repo root:"
  echo "  python3.10 -m venv cs224r"
  echo "  source cs224r/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

cd "$ROOT"
echo "Using: $PY ($("$PY" -c 'import numpy; print("numpy", numpy.__version__)'))"

"$PY" algorithms/evaluate_dqn_models.py \
  --best-per-algo \
  --reward-variants reward_dense \
  --splits test \
  --n-bootstrap 300

"$PY" algorithms/extended_policy_comparison.py --n-bootstrap 300

"$PY" algorithms/visualize_policy_comparison.py --export-latex

echo "Done. LaTeX rows: models/poster_results_table_rows.tex"
