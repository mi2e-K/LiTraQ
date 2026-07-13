# LiTraQ

LiTraQ is an independently developed, general-purpose Python toolkit for movement quantification and transit-event quality control in linear-track experiments. It reports distance, speed, edge occupancy, shuttle behavior, and straight-transit events from DeepLabCut tracking data. Both a GUI and a command-line interface are included.

LiTraQ was originally developed to analyze the alternate poking reward omission (APRO) paradigm, but its analysis pipeline is task-agnostic and can be applied to compatible tracking data from other linear-track experiments. LiTraQ is not an official implementation of the APRO paradigm and is not affiliated with or endorsed by its original authors.

## Files

- `litraq.py`: calibration, movement analysis, event detection, and QC output
- `litraq_gui.py`: PyQt6 GUI for single-file and batch analysis

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- OpenCV
- Matplotlib
- PyQt6
- PyAV (optional; improves random video access)

```bash
pip install numpy pandas opencv-python matplotlib PyQt6 av
```

## Usage

Start the GUI:

```bash
python litraq_gui.py
```

Select a processed video, a DeepLabCut filtered CSV or H5 file, and an arena calibration JSON file. Use the Batch tab to process multiple videos.

For command-line help:

```bash
python litraq.py --help
python litraq.py analyze --help
```

Main outputs include per-frame movement metrics, time-bin and edge-region summaries, straight-transit candidates and accepted events, shuttle events, and optional QC videos.

## Methods text (copy-ready)

The following text describes APRO-session analysis performed with the current default parameters:

> Mouse trajectories during alternate poking reward omission (APRO) sessions were analyzed using LiTraQ (Linear-track Trajectory Analysis and Quantification; https://github.com/mi2e-K/LiTraQ). DeepLabCut coordinates were perspective-rectified to the arena plane and converted to centimeters using a four-corner homography and a known-length reference. Body-center samples with a DeepLabCut likelihood below 0.90 or outside a 2.0-cm arena margin were treated as invalid; gaps of up to 0.50 s were linearly interpolated, and positions were smoothed with a centered 0.20-s rolling median. Movement was defined as speed of at least 1.0 cm/s after bridging interruptions of up to 0.20 s, and bouts shorter than 0.20 s or with net displacement below 0.50 cm were excluded. End-to-end transit candidates were movements between opposing 6.5-cm end zones, with a 0.5-cm completion tolerance. Straight transits required path efficiency of at least 0.95, maximum deviation from the start-end line of no more than 6.0 cm, at least 95% valid tracking, no frame-to-frame jump above 8.0 cm, no direction reversal above 3.25 cm, and no continuous period at or below 1.0 cm/s lasting at least 0.10 s. Candidates showing sustained wall-oriented posture (nose less than 1.0 cm from a long wall while the body center was more than 1.5 cm away for at least 60% of valid frames, with valid nose, body-center, and tail-base tracking for at least 80% of the event) overlapping speed at or below 5.0 cm/s for at least 0.10 s were rejected; threshold-borderline candidates were retained but flagged for visual QC. Video and tracking frame counts were required to agree within one frame.

## APRO reference

Naik AA, Ma X, Munyeshyaka M, Leibenluft E, Li Z. A New Behavioral Paradigm for Frustrative Nonreward in Juvenile Mice. *Biological Psychiatry: Global Open Science*. 2024;4:31-38. https://doi.org/10.1016/j.bpsgos.2023.09.007

## Notes

Verify video and DLC frame counts, arena calibration, and straight-transit QC before interpreting results. The wall-posture classifier is a 2D proxy derived from a top-down view and does not directly establish rearing.

## License

LiTraQ is released under the [MIT License](LICENSE).
