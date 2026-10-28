#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python evaluation/phi_fdt.py --check-only
spinnme run --config configs/phi/fdt.yaml
