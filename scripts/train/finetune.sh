#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
dataset="${1:-stead}"
if [[ "$#" -gt 0 ]]; then shift; fi
case "$dataset" in
  stead|instance) ;;
  *) echo 'Usage: finetune.sh {stead|instance} [Hydra overrides...]' >&2; exit 2 ;;
esac
exec uv run --no-sync python main.py --config-name "finetune/$dataset" "$@"
