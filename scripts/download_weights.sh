#!/usr/bin/env bash
set -euo pipefail

mkdir -p weights/100doh weights/yolo_world weights/hamer weights/sam2
echo "Created weights directories."
echo "MediaPipe downloads assets automatically when available."
echo "For real YOLO-World inference, place yolov8s-world.pt in weights/yolo_world/ and set EGOFLOW_USE_REAL_YOLO=1."
echo ""
echo "100DOH contact-state source:"
echo "  repo:    https://github.com/ddshan/hand_object_detector"
echo "  project: https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/"
echo "  model:   handobj_100K+ego / faster_rcnn_1_8_132028.pth"
echo "  target:  weights/100doh/hand_object_detector/models/res101_handobj_100K+ego/pascal_voc/faster_rcnn_1_8_132028.pth"
echo ""
echo "Note: do not use https://github.com/ddshan/hand_detector.d2 for contact state; it is a hand-box detector, not the full hand-object contact detector."
echo "For HaMeR and SAM2, follow their official repositories and place weights in weights/hamer/ and weights/sam2/."
