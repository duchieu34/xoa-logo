# CPU-only Veo Watermark Removal Research

Focused research project for removing the default Veo watermark from video while reconstructing the underlying image as naturally and temporally consistently as possible. The project does not use CUDA/GPU, does not crop, and does not treat blur or a flat patch as removal.

The current milestone is **Experiment 0 only**: inspect the benchmark video and export ROI diagnostics. It does not remove the watermark or produce an output video yet.

## Requirements

- Python 3.11+
- FFmpeg and `ffprobe` on `PATH`
- CPU-only OpenCV and NumPy

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Benchmark input

Place the local benchmark at:

```text
samples/ft-vid-23.mp4
```

Media and generated diagnostics are ignored by Git.

## Run Experiment 0

```powershell
.\.venv\Scripts\python.exe main.py samples/ft-vid-23.mp4 --diagnostics diagnostics/experiment0 --frames 8
```

The default ROI was calibrated from the benchmark and becomes `(1824, 1004, 96, 76)` at 1920×1080. If another default-Veo export uses different placement, override the resolution-independent ROI:

```powershell
.\.venv\Scripts\python.exe main.py samples/ft-vid-23.mp4 --roi 0.95,0.93,0.05,0.07
```

Each number is a fraction of frame width/height: `x,y,width,height`. Experiment 0 writes original frames, ROI overlays, raw ROI crops, contact sheets, `median_roi.png`, `temporal_std_roi.png`, and `report.json`.

## Repository layout

```text
main.py                         CLI entry point
veo_watermark_remover/          research package
  config.py                     relative ROI model
  video_io.py                   ffprobe metadata
  diagnostics.py                diagnostic artifacts
  experiment0.py                inspection-only experiment
assets/                         calibrated measurements; future templates
samples/                        local benchmark inputs (ignored)
diagnostics/                    generated evidence (ignored)
tests/                          unit tests
RESEARCH.md                     hypotheses, outcomes, decisions
STATUS.md                       current best method and next work
```

## Scope guardrails

No GUI, general object removal, heavy AI model, GPU path, or unrelated video feature is part of this phase. Experiment 1 must not begin until the benchmark evidence from Experiment 0 has been reviewed.
