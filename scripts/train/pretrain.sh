#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec uv run --no-sync python main.py --config-name pretrain/seispss "$@"
