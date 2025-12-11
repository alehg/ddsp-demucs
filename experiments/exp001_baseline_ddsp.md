# Experiment 001: Baseline DDSP Model

## Objective
Establish baseline performance of DDSP decoder on vocal separation task.

## Configuration
- Model: DDSPDecoder
- Sample rate: 22050 Hz
- Frame rate: 250 fps
- Harmonics: 64
- Noise bands: 65
- RNN units: 256

## Training
- Batch size: 8
- Learning rate: 1e-3
- Epochs: 50
- Optimizer: Adam

## Results
- Validation SI-SDR: TBD
- Spectral convergence: TBD

## Notes
Baseline experiment to establish performance before adding residual connections.

