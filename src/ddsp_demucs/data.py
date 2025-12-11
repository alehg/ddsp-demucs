"""Data loading, TFRecord building, and dataset utilities."""

import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import musdb


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Read WAV file as mono.
    
    Args:
        path: Path to WAV file
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    y, sr = sf.read(str(path), dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def slice_sec(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Slice audio array by time in seconds.
    
    Args:
        x: Audio array
        sr: Sample rate
        start_s: Start time in seconds
        end_s: End time in seconds
        
    Returns:
        Sliced audio array
    """
    a = int(round(start_s * sr))
    b = int(round(end_s * sr))
    a = max(0, min(a, len(x)))
    b = max(0, min(b, len(x)))
    if b <= a:
        return np.zeros(1, dtype=np.float32)
    return x[a:b]


def resample_if_needed(y: np.ndarray, sr: int, tgt: int) -> Tuple[np.ndarray, int]:
    """Resample audio if sample rates differ.
    
    Args:
        y: Audio array
        sr: Source sample rate
        tgt: Target sample rate
        
    Returns:
        Tuple of (resampled audio, sample rate)
    """
    if sr == tgt:
        return y, sr
    return librosa.resample(y, orig_sr=sr, target_sr=tgt), tgt


def rms_db(x: np.ndarray, eps: float = 1e-8) -> float:
    """Compute RMS in dB.
    
    Args:
        x: Audio array
        eps: Epsilon for numerical stability
        
    Returns:
        RMS in dB
    """
    rms = np.sqrt(np.mean(x.astype(np.float32)**2) + eps)
    return 20 * np.log10(rms + eps)


def make_tfrecord_example(x_in: np.ndarray,
                          x_tgt: np.ndarray,
                          sr: int,
                          track: str,
                          start_s: float,
                          end_s: float,
                          subset: str) -> tf.train.Example:
    """Create a TFRecord Example from audio data.
    
    Args:
        x_in: Input audio (float32 array)
        x_tgt: Target audio (float32 array)
        sr: Sample rate
        track: Track name
        start_s: Start time in seconds
        end_s: End time in seconds
        subset: Dataset subset ('train', 'val', 'test')
        
    Returns:
        TFRecord Example
    """
    x_in = np.asarray(x_in, dtype=np.float32)
    x_tgt = np.asarray(x_tgt, dtype=np.float32)
    
    feat = {
        "audio/inputs": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[x_in.tobytes()])
        ),
        "audio/targets": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[x_tgt.tobytes()])
        ),
        "audio/sample_rate": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(sr)])
        ),
        "audio/length": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(len(x_in))])
        ),
        "meta/track": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[track.encode()])
        ),
        "meta/subset": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[subset.encode()])
        ),
        "meta/start_sec": tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(start_s)])
        ),
        "meta/end_sec": tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(end_s)])
        ),
        "meta/source": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b"demucs_htdemucs->ddsp"])
        ),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat))


def parse_tfrecord_example(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parse a TFRecord example.
    
    Args:
        serialized: Serialized TFRecord example
        
    Returns:
        Dictionary of parsed features
    """
    feature_description = {
        "audio/inputs": tf.io.FixedLenFeature([], tf.string),
        "audio/targets": tf.io.FixedLenFeature([], tf.string),
        "audio/sample_rate": tf.io.FixedLenFeature([], tf.int64),
        "audio/length": tf.io.FixedLenFeature([], tf.int64),
        "meta/track": tf.io.FixedLenFeature([], tf.string),
        "meta/subset": tf.io.FixedLenFeature([], tf.string),
        "meta/start_sec": tf.io.FixedLenFeature([], tf.float32),
        "meta/end_sec": tf.io.FixedLenFeature([], tf.float32),
    }
    
    ex = tf.io.parse_single_example(serialized, feature_description)
    sr = tf.cast(ex["audio/sample_rate"], tf.int32)
    xin = tf.io.decode_raw(ex["audio/inputs"], tf.float32)
    xgt = tf.io.decode_raw(ex["audio/targets"], tf.float32)
    xin.set_shape([None])
    xgt.set_shape([None])
    
    return {
        "audio_input": xin,
        "audio_target": xgt,
        "sample_rate": sr,
        "track": ex["meta/track"],
        "subset": ex["meta/subset"],
        "start_sec": ex["meta/start_sec"],
        "end_sec": ex["meta/end_sec"],
    }


def build_tfrecords_from_segments(
    accepted_segments_csv: Path,
    stems_dir: Path,
    musdb_root: Path,
    tfrecords_dir: Path,
    train_sr: int = 22050,
    win_s: float = 4.0,
    hop_s: float = 1.0,
    examples_per_shard: int = 512,
    val_ratio: float = 0.1,
    min_rms_db: float = -50.0,
    seed: int = 1337
) -> Dict[str, int]:
    """Build TFRecords from accepted segments.
    
    Args:
        accepted_segments_csv: Path to CSV with accepted segments
        stems_dir: Directory with Demucs stems
        musdb_root: Root of MUSDB dataset
        tfrecords_dir: Output directory for TFRecords
        train_sr: Training sample rate
        win_s: Window size in seconds
        hop_s: Hop size in seconds
        examples_per_shard: Examples per shard file
        val_ratio: Validation split ratio
        min_rms_db: Minimum RMS in dB to keep example
        seed: Random seed for splitting
        
    Returns:
        Dictionary with counts of examples per split
    """
    # Load segments
    segs = pd.read_csv(accepted_segments_csv)
    segs = segs[segs.segment_pass == True].copy()
    segs["dur_s"] = segs["end_s"] - segs["start_s"]
    segs = segs[segs["dur_s"] > 0.1]
    
    # Load MUSDB
    db = musdb.DB(root=str(musdb_root), subsets=['train', 'test'], is_wav=True)
    name2track = {t.name: t for t in db.tracks}
    subset_map = {t.name: t.subset for t in db.tracks}
    
    # Create splits
    rng = np.random.default_rng(seed)
    tracks = sorted(segs.track.unique())
    val_candidates = [t for t in tracks if subset_map.get(t, "train") == "train"]
    rng.shuffle(val_candidates)
    n_val = int(round(len(val_candidates) * val_ratio))
    val_set = set(val_candidates[:n_val])
    
    def split_of(track):
        s = subset_map.get(track, "train")
        if s == "test":
            return "test"
        return "val" if track in val_set else "train"
    
    segs["split"] = segs.track.map(split_of)
    
    # Shard writer helper
    def shard_writer(prefix: Path, split: str, max_per_shard: int):
        prefix.mkdir(parents=True, exist_ok=True)
        count = 0
        shard_idx = 0
        writer = None
        
        def _open_new():
            nonlocal writer, shard_idx, count
            if writer is not None:
                writer.close()
            shard_path = prefix / f"{split}-{shard_idx:05d}.tfrecord"
            writer = tf.io.TFRecordWriter(str(shard_path))
            shard_idx += 1
            count = 0
        
        _open_new()
        
        def write(example):
            nonlocal writer, count
            writer.write(example.SerializeToString())
            count += 1
            if count >= max_per_shard:
                _open_new()
        
        def close():
            if writer is not None:
                writer.close()
        
        return write, close
    
    writers = {
        "train": shard_writer(tfrecords_dir / "train", "train", examples_per_shard),
        "val": shard_writer(tfrecords_dir / "val", "val", max(64, examples_per_shard // 2)),
        "test": shard_writer(tfrecords_dir / "test", "test", max(64, examples_per_shard // 2)),
    }
    
    # Cache GT vocals
    gt_cache = {}
    
    def get_gt_vocals(track_name: str):
        if track_name in gt_cache:
            return gt_cache[track_name]
        mt = name2track.get(track_name, None)
        if mt is None:
            raise ValueError(f"Track not found in MUSDB: {track_name}")
        y = mt.targets['vocals'].audio
        if y.ndim == 2:
            y = y.mean(axis=1)
        gt_cache[track_name] = (y.astype(np.float32), int(mt.rate))
        return gt_cache[track_name]
    
    # Build examples
    examples_total = {"train": 0, "val": 0, "test": 0}
    
    for track_name, group in segs.groupby("track"):
        in_path = stems_dir / track_name / "vocals.mono.wav"
        if not in_path.exists():
            continue
        
        x_in, sr_in = read_wav_mono(in_path)
        x_gt, sr_gt = get_gt_vocals(track_name)
        
        for _, row in group.iterrows():
            start_s, end_s = float(row.start_s), float(row.end_s)
            seg_in = slice_sec(x_in, sr_in, start_s, end_s)
            seg_gt = slice_sec(x_gt, sr_gt, start_s, end_s)
            
            seg_in, _ = resample_if_needed(seg_in, sr_in, train_sr)
            seg_gt, _ = resample_if_needed(seg_gt, sr_gt, train_sr)
            
            N = len(seg_in)
            win = int(round(win_s * train_sr))
            hop = int(round(hop_s * train_sr))
            
            if N < win:
                continue
            
            for a in range(0, N - win + 1, hop):
                b = a + win
                xin = seg_in[a:b]
                xgt = seg_gt[a:b]
                
                if rms_db(xin) < min_rms_db or rms_db(xgt) < min_rms_db:
                    continue
                
                example = make_tfrecord_example(
                    xin, xgt, train_sr, track_name,
                    start_s + a / train_sr, start_s + b / train_sr,
                    split_of(track_name)
                )
                write, _ = writers[split_of(track_name)]
                write(example)
                examples_total[split_of(track_name)] += 1
    
    # Close writers
    for _, (_, close) in writers.items():
        close()
    
    return examples_total

