"""Utility functions for paths, config loading, and logging."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, looks for env/config.yaml
        
    Returns:
        Dictionary with configuration
    """
    if config_path is None:
        # Try to find config relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "env" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_paths(config: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Get project paths from config.
    
    Args:
        config: Config dict. If None, loads from default location.
        
    Returns:
        Dictionary of path names to Path objects
    """
    if config is None:
        config = load_config()
    
    paths = config.get("paths", {})
    return {
        "project": Path(paths.get("project", ".")),
        "stems_dir": Path(paths.get("stems_dir", "data/stems")),
        "features_dir": Path(paths.get("features_dir", "data/features")),
        "tfrecords_dir": Path(paths.get("tfrecords_dir", "data/tfrecords")),
        "exp_dir": Path(paths.get("exp_dir", "exp")),
    }


def setup_logging(log_dir: Optional[Path] = None, level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        log_dir: Directory for log files. If None, uses exp_dir/logs
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    import logging
    
    if log_dir is None:
        config = load_config()
        paths = get_project_paths(config)
        log_dir = paths["exp_dir"] / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "ddsp_demucs.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

