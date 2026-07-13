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

> Mouse trajectories during alternate poking reward omission (APRO) sessions were analyzed from DeepLabCut tracking data using LiTraQ (Linear-track Trajectory Analysis and Quantification; https://github.com/mi2e-K/LiTraQ). Coordinates were perspective-corrected and converted to centimeters. Samples with likelihood below 0.90 were excluded, gaps up to 0.50 s were interpolated, and tracks were median-smoothed over 0.20 s. Movement was defined as speed of at least 1.0 cm/s after excluding bouts shorter than 0.20 s or with net displacement below 0.50 cm. End-to-end transits between opposing 6.5-cm end zones were classified using default thresholds for path efficiency, deviation, pausing, tracking validity, direction reversal, and wall-posture interruption; borderline events were flagged for visual QC. Video and tracking frame counts were verified before analysis.

## APRO reference

Naik AA, Ma X, Munyeshyaka M, Leibenluft E, Li Z. A New Behavioral Paradigm for Frustrative Nonreward in Juvenile Mice. *Biological Psychiatry: Global Open Science*. 2024;4:31-38. https://doi.org/10.1016/j.bpsgos.2023.09.007

## Notes

Verify video and DLC frame counts, arena calibration, and straight-transit QC before interpreting results. The wall-posture classifier is a 2D proxy derived from a top-down view and does not directly establish rearing.

## License

LiTraQ is released under the [MIT License](LICENSE).
