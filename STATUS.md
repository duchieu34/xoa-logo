# Project Status

## Current best method

None. Experiment 0 performs inspection only and deliberately does not remove pixels.

## Current milestone

Project skeleton and Experiment 0 implementation are ready. Five unit tests and a synthetic H.264/AAC end-to-end smoke test pass. Execution against the benchmark is pending because `samples/ft-vid-23.mp4` is absent.

## Known problems

- Exact Veo watermark position, dimensions, color, opacity, and stationarity are not yet measured.
- The default candidate ROI is intentionally broad and uncalibrated.
- OpenCV is not installed in the active system Python; use the documented Python 3.11 virtual environment.
- No continuous-video quality evaluation can occur before a restoration experiment and benchmark input exist.

## Next experiment

Run Experiment 0 on the benchmark, visually inspect all diagnostics, tighten the ROI, and record measured watermark properties. Only then design the shape-accurate mask for Experiment 1 (TELEA versus Navier–Stokes baseline).
