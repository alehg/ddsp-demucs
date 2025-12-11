# Experiment 002: Residual DDSP Model

## Objective
Improve vocal separation by predicting residual correction to Demucs output.

## Configuration
- Model: ResidualDDSPDecoder
- Sample rate: 22050 Hz
- Frame rate: 250 fps
- Harmonics: 64
- Noise bands: 65
- RNN units: 256
- Residual connection: Yes (learnable dry/wet mix)

## Training
- Batch size: 8
- Learning rate: 1e-3
- Epochs: 50
- Optimizer: Adam

## Results
- Validation SI-SDR: TBD
- Spectral convergence: TBD
- Improvement over baseline: TBD

## Notes
Residual model should improve upon Demucs baseline by learning correction terms.

