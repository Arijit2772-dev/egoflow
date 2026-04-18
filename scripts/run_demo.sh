#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-../round_1/sample_video/DSJ_0000000_000000_20250221030623.MP4}"
python3 egoflow.py --input "$VIDEO" --resume
python3 egoflow.py --serve
