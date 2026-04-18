#!/usr/bin/env bash
# EgoFlow <-> 100DOH integration boundary.
#
# Contract (called by Contact100DOH.predict):
#   scripts/run_100doh.sh --image /abs/path/keyframe.jpg \
#                         --output /abs/path/100doh_result.json
#
# Expected JSON payload written to --output:
#   {
#     "hands": [
#       {"side": "left"|"right", "bbox": [x1,y1,x2,y2], "confidence": 0..1,
#        "contact_state": "no_contact"|"self_contact"|"person_contact"|
#                         "portable_object"|"stationary_object",
#        "in_contact_object_bbox": [x1,y1,x2,y2] | null,
#        "raw_state_id": 0..4}
#     ],
#     "objects": [{"bbox": [x1,y1,x2,y2], "confidence": 0..1}],
#     "source": "100doh"
#   }
#
# This shim activates the external Python 3.8 + PyTorch 1.12 + CUDA 11.3
# environment that ships with ddshan/hand_object_detector. If the env or
# checkpoint are missing, we exit non-zero with a clear message. We do NOT
# emit fake detections. EgoFlow will log the failure and fall back to its
# overlap heuristic.
#
# Required environment variables (set by the operator once 100DOH is wired):
#   EGOFLOW_100DOH_DIR         Absolute path to a clone of
#                              https://github.com/ddshan/hand_object_detector
#   EGOFLOW_100DOH_CHECKPOINT  Absolute path to the handobj_100K+ego checkpoint
#                              (faster_rcnn_1_8_132028.pth)
#   EGOFLOW_100DOH_PYTHON      Absolute path to the python interpreter inside
#                              the 100DOH Python 3.8 env (conda/virtualenv)
#   EGOFLOW_100DOH_ENTRY       Optional. Absolute path to an inference entry
#                              script adhering to the contract above. If
#                              unset, falls back to
#                              $EGOFLOW_100DOH_DIR/egoflow_infer.py
#                              which the operator writes once (a thin wrapper
#                              around demo.py that dumps JSON).
#
# Exit codes:
#   0  success, JSON written to --output
#   2  invalid arguments
#   3  environment not configured (one of the required vars/paths missing)
#   4  inference process failed

set -euo pipefail

IMAGE=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "run_100doh.sh: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$IMAGE" || -z "$OUTPUT" ]]; then
  echo "run_100doh.sh: --image and --output are required" >&2
  exit 2
fi

if [[ ! -f "$IMAGE" ]]; then
  echo "run_100doh.sh: image not found: $IMAGE" >&2
  exit 2
fi

missing=()
[[ -n "${EGOFLOW_100DOH_DIR:-}" && -d "$EGOFLOW_100DOH_DIR" ]] || missing+=("EGOFLOW_100DOH_DIR (hand_object_detector repo)")
[[ -n "${EGOFLOW_100DOH_CHECKPOINT:-}" && -f "$EGOFLOW_100DOH_CHECKPOINT" ]] || missing+=("EGOFLOW_100DOH_CHECKPOINT (faster_rcnn_1_8_132028.pth)")
[[ -n "${EGOFLOW_100DOH_PYTHON:-}" && -x "$EGOFLOW_100DOH_PYTHON" ]] || missing+=("EGOFLOW_100DOH_PYTHON (python interpreter in the 3.8 env)")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "run_100doh.sh: 100DOH environment not configured. Missing:" >&2
  for m in "${missing[@]}"; do echo "  - $m" >&2; done
  echo "Set the variables above and re-run, or keep annotation.enable_100doh=false." >&2
  exit 3
fi

ENTRY="${EGOFLOW_100DOH_ENTRY:-$EGOFLOW_100DOH_DIR/egoflow_infer.py}"
if [[ ! -f "$ENTRY" ]]; then
  echo "run_100doh.sh: inference entry script not found: $ENTRY" >&2
  echo "Create it as a thin wrapper around demo.py that accepts --image/--output" >&2
  echo "and writes the JSON schema documented in this shim's header." >&2
  exit 3
fi

if ! "$EGOFLOW_100DOH_PYTHON" "$ENTRY" --image "$IMAGE" --output "$OUTPUT" --checkpoint "$EGOFLOW_100DOH_CHECKPOINT"; then
  echo "run_100doh.sh: 100DOH inference failed" >&2
  exit 4
fi

if [[ ! -s "$OUTPUT" ]]; then
  echo "run_100doh.sh: 100DOH produced no output JSON at $OUTPUT" >&2
  exit 4
fi
