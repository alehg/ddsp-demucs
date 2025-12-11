"""Tests for utility functions."""

import pytest
from pathlib import Path
from ddsp_demucs.utils import load_config, get_project_paths


def test_load_config(tmp_path):
    """Test config loading."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
dataset:
  kind: hq
  sample_rate: 44100
paths:
  project: .
  stems_dir: data/stems
""")
    
    config = load_config(config_file)
    assert config["dataset"]["kind"] == "hq"
    assert config["dataset"]["sample_rate"] == 44100


def test_get_project_paths(tmp_path):
    """Test path extraction."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
paths:
  project: .
  stems_dir: data/stems
  features_dir: data/features
  tfrecords_dir: data/tfrecords
  exp_dir: exp
""")
    
    config = load_config(config_file)
    paths = get_project_paths(config)
    
    assert "stems_dir" in paths
    assert "features_dir" in paths
    assert isinstance(paths["stems_dir"], Path)

