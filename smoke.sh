#!/usr/bin/env bash
# (2/3) SMOKE — fast sanity check, no GPU. Validates configs, the pipeline logic,
# the venv router, reward/curriculum/metrics, and (lightly) that the venvs exist.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== venv presence ==="
ok=1
for v in .venv .venv-serve .venv-rl; do
  if [ -x "$v/bin/python" ]; then echo "  $v  OK"; else echo "  $v  MISSING (run ./setup.sh)"; ok=0; fi
done

echo "=== unit/smoke tests (in .venv) ==="
.venv/bin/python -m pytest tests/ -q

if [ "$ok" -eq 1 ]; then
  echo "=== import check across stacks ==="
  .venv/bin/python        -c "import src.config, src.data.build_cpt, src.eval.report, src.venvs; print('  .venv pipeline+train imports OK')"
  .venv-serve/bin/python  -c "import vllm; print('  .venv-serve vLLM', vllm.__version__, 'OK')"
  .venv-rl/bin/python     -c "import skyrl.train.entrypoints.main_base, src.harness.skyrl_env; print('  .venv-rl SkyRL + env adapter OK')"
fi
echo "=== SMOKE DONE ==="
