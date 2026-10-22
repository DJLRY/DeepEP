#!/usr/bin/env bash
# Copyright (c) 2026, Lu Lu

set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON:-python3}" -I "$SCRIPT_DIR/bs32_ab.py" --experiment dispatch "$@"
