# EgoFlow

The first India-focused, research-backed egocentric data pipeline for humanoid robotics.

EgoFlow converts raw first-person video into manipulation-ready training data for robot imitation learning. Unlike generic annotation pipelines, EgoFlow outputs hand-object contact states, grasp types, 3D hand poses, and pixel-level object masks - the supervision signals used by modern robot policy pipelines.

Built on six research foundations:

- 100DOH - Shan et al., CVPR 2020
- HaMeR - Pavlakos et al., ICCV 2023
- YOLO-World - Cheng et al., CVPR 2024
- SAM2 - Ravi et al., arXiv 2024
- Ego4D schema - Grauman et al., CVPR 2022
- CLIP - Radford et al., ICML 2021

The MVP runs on a laptop with no model training. Heavy models are fail-soft: when local weights or API keys are absent, EgoFlow keeps the same schema and uses deterministic fallbacks so the demo remains end-to-end.

## Quick Start

```bash
cd egoflow
.venv311/bin/python -m pip install -r requirements.txt
cp .env.example .env
.venv311/bin/python egoflow.py ../round_1/sample_video/DSJ_0000000_000000_20250221030623.MP4
.venv311/bin/python egoflow.py --serve
```

Open `http://localhost:8000` for the viewer.

## CLI

```bash
.venv311/bin/python egoflow.py --input video.mp4
.venv311/bin/python egoflow.py video.mp4
.venv311/bin/python egoflow.py --input video.mp4 --phases 1,2,3
.venv311/bin/python egoflow.py --input video.mp4 --resume
.venv311/bin/python egoflow.py --serve
```

## Outputs

Each run writes file-based handoffs under `output/<video_uid>/`:

- `meta.json`
- `segments.json`
- `annotations/clip_NNN.json`
- `narrations/clip_NNN.json`
- `dataset.json`
- `validation_report.json`

## Architecture

```text
video.mp4
  -> phase1_ingest: normalized.mp4 + sampled frames + meta.json
  -> phase2_segment: segments.json + clips
  -> phase3_annotate: hands, objects, contact, grasp, optional pose and masks
  -> phase4_describe: structured narration
  -> phase5_assemble: Ego4D-compatible dataset.json
  -> phase6_validate: quality checks and semantic consistency
```

## Research Links

- 100DOH: https://arxiv.org/abs/2006.06669
- 100DOH code: https://github.com/ddshan/hand_object_detector
- 100DOH project page: https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/
- HaMeR: https://arxiv.org/abs/2312.05251
- YOLO-World: https://arxiv.org/abs/2401.17270
- SAM2: https://arxiv.org/abs/2408.00714
- Ego4D: https://arxiv.org/abs/2110.07058
- CLIP: https://arxiv.org/abs/2103.00020

## 100DOH Contact-State Model

Use the official `ddshan/hand_object_detector` repository for the contact-state module. That is the CVPR 2020 hand-object detector that outputs hand boxes, side, contact state, and contacted-object boxes.

Recommended model for EgoFlow:

- `handobj_100K+ego`
- checkpoint: `faster_rcnn_1_8_132028.pth`
- expected path: `weights/100doh/hand_object_detector/models/res101_handobj_100K+ego/pascal_voc/faster_rcnn_1_8_132028.pth`

Do not use `ddshan/hand_detector.d2` for the core contact-state claim. It is useful for hand boxes, but it does not provide the full hand-object contact output EgoFlow needs.

The official detector is CUDA/Faster-RCNN based and its README targets a separate Python 3.8 + PyTorch 1.12.1 + CUDA 11.3 environment. EgoFlow stays fail-soft: when the full detector is absent, it preserves the schema and uses the local hand-object overlap fallback.
