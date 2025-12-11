# DDSP-Demucs

Research code for improving vocal stem separation by combining Hybrid Transformer Demucs with Differentiable DSP. This project implements residual additive synthesis, multi-resolution spectral losses, and analyses of fundamental frequency noise and model stability.

## Overview

This repository contains a complete pipeline for:
1. **Source Separation**: Using Demucs to extract vocal stems from music
2. **Feature Extraction**: Computing F0, loudness, and spectral features
3. **DDSP Synthesis**: Training differentiable DSP models to refine vocal separation
4. **Evaluation**: Computing SI-SDR, spectral convergence, and other metrics

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- MUSDB18-HQ dataset

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ddsp-demucs
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install DDSP (from source):
```bash
pip install --no-deps "git+https://github.com/magenta/ddsp@main#egg=ddsp"
```

5. Install the package in development mode:
```bash
pip install -e .
```

## Project Structure

```
ddsp-demucs/
├── src/
│   └── ddsp_demucs/
│       ├── __init__.py
│       ├── data.py          # Loading, slicing, TFRecords
│       ├── dsp.py           # DDSP modules, spectral losses
│       ├── demucs_wrappers.py  # Demucs inference wrappers
│       ├── model.py         # Residual additive model, architectures
│       ├── train.py         # Training loops, callbacks, checkpoints
│       ├── evaluate.py      # SI-SDR, metrics, plots
│       └── utils.py         # Logging, config, paths
│
├── notebooks/
│   ├── 01_musdb_demucs_inference.ipynb
│   ├── 02_features_and_gate.ipynb
│   ├── 03_build_tfrecords.ipynb
│   ├── 04_train_ddsp.ipynb
│   └── ...
│
├── configs/
│   ├── base.yaml
│   ├── ddsp_residual_small.yaml
│   └── ddsp_residual_large.yaml
│
├── experiments/
│   ├── exp001_baseline_ddsp.md
│   └── exp002_residual_model.md
│
├── scripts/
│   ├── train_ddsp_residual.sh
│   └── evaluate_experiment.sh
│
├── tests/
│   └── (unit tests - nice to have)
│
├── README.md
├── requirements.txt
├── pyproject.toml
└── .gitignore
```

## Quick Start

### 1. Prepare Data

First, run Demucs inference to extract vocal stems:

```python
from ddsp_demucs.demucs_wrappers import DemucsSeparator
from pathlib import Path
import musdb

# Initialize separator
separator = DemucsSeparator(model_name="htdemucs")

# Process MUSDB tracks
db = musdb.DB(root="path/to/musdb18hq", subsets=["train", "test"], is_wav=True)
for track in db:
    mixture = track.audio  # (T, C) numpy array
    vocals = separator.extract_vocals(mixture)
    # Save vocals...
```

### 2. Extract Features

Extract F0, loudness, and other features (see notebooks for full pipeline).

### 3. Build TFRecords

```python
from ddsp_demucs.data import build_tfrecords_from_segments
from pathlib import Path

build_tfrecords_from_segments(
    accepted_segments_csv=Path("data/features/accepted_segments.csv"),
    stems_dir=Path("data/stems/demucs_htdemucs44k"),
    musdb_root=Path("data/musdb18hq"),
    tfrecords_dir=Path("data/tfrecords"),
    train_sr=22050,
    win_s=4.0,
    hop_s=1.0,
)
```

### 4. Train Model

```python
from ddsp_demucs.model import ResidualDDSPDecoder
from ddsp_demucs.train import DDSPTrainer, setup_training_environment, create_callbacks
from ddsp_demucs.utils import load_config
import tensorflow as tf

# Setup environment
setup_training_environment(use_mixed_precision=True)

# Load config
config = load_config("configs/base.yaml")

# Create model
model = ResidualDDSPDecoder(
    sample_rate=config["training"]["sample_rate"],
    frame_rate=config["training"]["frame_rate"],
    n_harmonics=config["model"]["n_harmonics"],
    n_noise_bands=config["model"]["n_noise_bands"],
)

# Create trainer
trainer = DDSPTrainer(model)

# Compile
optimizer = tf.keras.optimizers.Adam(config["training"]["learning_rate"])
trainer.compile(optimizer=optimizer)

# Create callbacks
callbacks = create_callbacks(
    checkpoint_dir=Path("exp/run001/ckpt"),
    log_dir=Path("exp/run001/tb"),
)

# Train
trainer.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config["training"]["epochs"],
    callbacks=callbacks,
)
```

Or use the shell script:

```bash
chmod +x scripts/train_ddsp_residual.sh
./scripts/train_ddsp_residual.sh configs/base.yaml exp001_baseline
```

### 5. Evaluate

```python
from ddsp_demucs.evaluate import evaluate_model, plot_evaluation_results

results = evaluate_model(model, val_dataset, n_examples=100)
plot_evaluation_results(results, output_path=Path("results/eval.png"))
```

Or use the shell script:

```bash
chmod +x scripts/evaluate_experiment.sh
./scripts/evaluate_experiment.sh exp001_baseline
```

## Configuration

Configuration files are in YAML format under `configs/`. Key parameters:

- **Dataset**: Paths, sample rates, gate thresholds
- **Training**: Batch size, learning rate, epochs
- **Model**: Architecture parameters (harmonics, RNN units, etc.)
- **Loss**: Loss function weights and FFT sizes

## Notebooks

The `notebooks/` directory contains exploratory analysis and prototyping:

- `01_musdb_demucs_inference.ipynb`: Demucs separation pipeline
- `02_features_and_gate.ipynb`: Feature extraction and gating
- `03_build_tfrecords.ipynb`: TFRecord dataset creation
- `04_train_ddsp.ipynb`: Training experiments
- Additional notebooks for model variants and analysis

## Experiments

Experiment logs and results are documented in `experiments/`. Each experiment includes:
- Configuration used
- Training hyperparameters
- Results and metrics
- Notes and observations

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black src/
flake8 src/
```

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) for source separation
- [DDSP](https://github.com/magenta/ddsp) for differentiable DSP
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html) for the dataset
