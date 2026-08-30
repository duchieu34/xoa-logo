# Veo Watermark Removal — Research Log

This project is intentionally developed as a sequence of measured experiments. The quality order is reconstruction fidelity, temporal stability, then speed. All processing must remain CPU-only.

## Experiment 0 — Video and candidate ROI inspection

### Hypothesis

Before choosing a removal method, representative frames and a resolution-independent ROI are required to establish whether the default Veo mark is spatially fixed, how its apparent pixels vary with background, and whether alpha deblending is plausible.

### Method

- Probe the container and streams with `ffprobe`.
- Select eight frames uniformly across the complete timeline.
- Convert a relative ROI into pixel coordinates for the source resolution.
- Save each original frame, ROI overlay, and unmodified ROI crop.
- Save contact sheets, the temporal median ROI, a temporal standard-deviation visualization, and per-frame descriptive statistics.
- Do not alter or encode a video.

### Result

Pending real sample. `samples/ft-vid-23.mp4` was not present when Experiment 0 was initialized. Therefore no claim is currently made about exact position, size, color, opacity, or stationarity of the Veo watermark.

The default ROI `(0.78, 0.78, 0.21, 0.20)` is only a generous bottom-right candidate region and must be calibrated against the diagnostics from the real sample.

The implementation was smoke-tested on a generated 640×360, 24 FPS, one-second H.264/AAC video. It correctly probed both streams, sampled the full timeline, and produced all documented diagnostics. This validates plumbing only, not watermark analysis or restoration quality.

### Advantages

- Makes no destructive assumptions about the logo.
- Works at arbitrary resolutions through relative coordinates.
- Produces evidence needed to choose difficult, textured, and moving frames.

### Limitations

- Uniform temporal sampling is a first pass; it does not semantically classify scene difficulty.
- Median and temporal variation images can reveal stable structures, but are not an alpha mask.
- No restoration quality can be assessed until the sample exists.

### Decision

Do not proceed to Experiment 1 until the real sample is run, the ROI is calibrated tightly with context margin, and representative frames are reviewed visually.

## Proposed recovery strategy (provisional)

The data-driven strategy will be finalized after Experiment 0. The current candidate is: estimate a stable logo color/alpha template from temporal observations; invert compositing where the estimate is well-conditioned; flow-warp clean evidence from both temporal directions with forward/backward consistency checks; blend only trusted pixels; use shape-tight CPU inpainting solely for unresolved mask pixels. Scene cuts and occlusions must disable invalid temporal donors.
