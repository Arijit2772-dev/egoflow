#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights/100doh weights/yolo_world weights/hamer weights/sam2
echo "Created weights directories."
echo "MediaPipe downloads assets automatically when available."
echo "For real YOLO-World inference, place yolov8s-world.pt in weights/yolo_world/ and set EGOFLOW_USE_REAL_YOLO=1."
echo "For 100DOH, HaMeR, and SAM2, follow their official repositories and place weights in the matching directories."
