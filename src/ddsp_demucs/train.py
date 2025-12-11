"""Training loops, callbacks, and checkpointing."""

import tensorflow as tf
import tensorflow.keras as keras
from pathlib import Path
from typing import Dict, Optional, Callable
from ddsp_demucs.dsp import create_spectral_loss, mel_spectrogram, spectral_centroid


class DDSPTrainer(keras.Model):
    """Trainer wrapper for DDSP models with custom loss functions."""
    
    def __init__(self,
                 model: keras.Model,
                 loss_fn: Optional[Callable] = None,
                 mel_weight: float = 1.0,
                 centroid_weight: float = 0.05,
                 sisdr_weight: float = 0.0,
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 n_mels: int = 64):
        """Initialize trainer.
        
        Args:
            model: DDSP model to train
            loss_fn: Spectral loss function (if None, creates default)
            mel_weight: Weight for mel loss
            centroid_weight: Weight for spectral centroid loss
            sisdr_weight: Weight for SI-SDR loss
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length
            n_mels: Number of mel bins
        """
        super().__init__()
        self.model = model
        self.mel_weight = mel_weight
        self.centroid_weight = centroid_weight
        self.sisdr_weight = sisdr_weight
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        if loss_fn is None:
            self.loss_fn = create_spectral_loss()
        else:
            self.loss_fn = loss_fn
        
        # Metrics
        self.train_metric = keras.metrics.Mean(name="loss")
        self.val_metric = keras.metrics.Mean(name="val_loss")
        self.spec_metric = keras.metrics.Mean(name="spec")
        self.mel_metric = keras.metrics.Mean(name="mel")
        self.cent_metric = keras.metrics.Mean(name="cent")
        self.sisdr_metric = keras.metrics.Mean(name="sisdr")
    
    @property
    def metrics(self):
        return [
            self.train_metric, self.val_metric,
            self.spec_metric, self.mel_metric,
            self.cent_metric, self.sisdr_metric
        ]
    
    def compile(self, optimizer, **kwargs):
        """Compile trainer.
        
        Args:
            optimizer: Optimizer instance
            **kwargs: Additional compile arguments
        """
        super().compile(**kwargs)
        self.optimizer = optimizer
        self._uses_loss_scale = hasattr(self.optimizer, "get_scaled_loss")
    
    @tf.function
    def _component_losses(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Compute component losses.
        
        Args:
            y_true: Ground truth audio
            y_pred: Predicted audio
            
        Returns:
            Dictionary of loss components
        """
        # Spectral loss
        Ls = self.loss_fn(y_true, y_pred)
        
        # Mel loss
        Yt = mel_spectrogram(y_true, n_fft=self.n_fft, hop_length=self.hop_length,
                            n_mels=self.n_mels, sr=self.sample_rate)
        Yp = mel_spectrogram(y_pred, n_fft=self.n_fft, hop_length=self.hop_length,
                            n_mels=self.n_mels, sr=self.sample_rate)
        n = tf.minimum(tf.shape(Yt)[1], tf.shape(Yp)[1])
        Lm = tf.reduce_mean(tf.abs(Yt[:, :n, :] - Yp[:, :n, :]))
        
        # Spectral centroid loss
        ct = spectral_centroid(y_true, n_fft=self.n_fft, hop_length=self.hop_length,
                              sr=self.sample_rate)
        cp = spectral_centroid(y_pred, n_fft=self.n_fft, hop_length=self.hop_length,
                              sr=self.sample_rate)
        Lc = tf.reduce_mean(tf.abs(ct - cp)) / (self.sample_rate / 2.0)
        
        # SI-SDR loss (optional)
        if self.sisdr_weight > 0.0:
            Ld = self._si_sdr_loss(y_true, y_pred)
        else:
            Ld = tf.constant(0.0, tf.float32)
        
        total = Ls + self.mel_weight * Lm + self.centroid_weight * Lc + self.sisdr_weight * Ld
        
        return {
            "total": total,
            "spectral": Ls,
            "mel": Lm,
            "centroid": Lc,
            "sisdr": Ld
        }
    
    @tf.function
    def _si_sdr_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
        """Scale-invariant SDR loss.
        
        Args:
            y_true: Ground truth
            y_pred: Prediction
            eps: Epsilon
            
        Returns:
            Negative SI-SDR (loss)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        dot = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        den = tf.reduce_sum(y_true * y_true, axis=-1, keepdims=True) + eps
        s_target = dot / den * y_true
        e_noise = y_pred - s_target
        
        num = tf.reduce_sum(s_target**2, axis=-1) + eps
        den = tf.reduce_sum(e_noise**2, axis=-1) + eps
        si_sdr = 10.0 * tf.math.log(num / den) / tf.math.log(10.0)
        
        return -tf.reduce_mean(si_sdr)  # minimize negative SI-SDR
    
    @tf.function
    def train_step(self, data):
        """Training step.
        
        Args:
            data: Tuple of (conditioning, target)
            
        Returns:
            Dictionary of metrics
        """
        cond, target = data
        with tf.GradientTape() as tape:
            pred = self.model(cond, training=True)
            n = tf.minimum(tf.shape(pred)[1], tf.shape(target)[1])
            y_t = tf.cast(target[:, :n], tf.float32)
            y_p = tf.cast(pred[:, :n], tf.float32)
            
            losses = self._component_losses(y_t, y_p)
            loss = losses["total"]
            
            if self._uses_loss_scale:
                loss = self.optimizer.get_scaled_loss(loss)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self._uses_loss_scale:
            grads = self.optimizer.get_unscaled_gradients(grads)
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update metrics
        self.train_metric.update_state(losses["total"])
        self.spec_metric.update_state(losses["spectral"])
        self.mel_metric.update_state(losses["mel"])
        self.cent_metric.update_state(losses["centroid"])
        self.sisdr_metric.update_state(losses["sisdr"])
        
        return {
            "loss": self.train_metric.result(),
            "spec": self.spec_metric.result(),
            "mel": self.mel_metric.result(),
            "cent": self.cent_metric.result(),
            "sisdr": self.sisdr_metric.result(),
        }
    
    @tf.function
    def test_step(self, data):
        """Validation step.
        
        Args:
            data: Tuple of (conditioning, target)
            
        Returns:
            Dictionary of metrics
        """
        cond, target = data
        pred = self.model(cond, training=False)
        n = tf.minimum(tf.shape(pred)[1], tf.shape(target)[1])
        y_t = tf.cast(target[:, :n], tf.float32)
        y_p = tf.cast(pred[:, :n], tf.float32)
        
        losses = self._component_losses(y_t, y_p)
        
        # Update metrics
        self.val_metric.update_state(losses["total"])
        self.spec_metric.update_state(losses["spectral"])
        self.mel_metric.update_state(losses["mel"])
        self.cent_metric.update_state(losses["centroid"])
        self.sisdr_metric.update_state(losses["sisdr"])
        
        return {
            "val_loss": self.val_metric.result(),
            "spec": self.spec_metric.result(),
            "mel": self.mel_metric.result(),
            "cent": self.cent_metric.result(),
            "sisdr": self.sisdr_metric.result(),
        }


def setup_training_environment(use_mixed_precision: bool = True,
                               gpu_memory_growth: bool = True,
                               xla_jit: bool = False):
    """Setup TensorFlow training environment.
    
    Args:
        use_mixed_precision: Enable mixed precision training
        gpu_memory_growth: Enable GPU memory growth
        xla_jit: Enable XLA JIT compilation
    """
    os = __import__('os')
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    if not xla_jit:
        os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
    
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    
    # GPU memory growth
    if gpu_memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    
    # Mixed precision
    if use_mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")
    
    # XLA
    if not xla_jit:
        tf.config.optimizer.set_jit(False)


def create_callbacks(checkpoint_dir: Path,
                     log_dir: Optional[Path] = None,
                     monitor: str = "val_loss",
                     mode: str = "min",
                     patience: int = 8,
                     save_best_only: bool = True) -> list:
    """Create training callbacks.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        log_dir: Directory for TensorBoard logs
        monitor: Metric to monitor
        mode: 'min' or 'max'
        patience: Early stopping patience
        save_best_only: Only save best model
        
    Returns:
        List of callbacks
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "ddsp.best.weights.h5"),
            save_weights_only=True,
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
        ),
    ]
    
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(log_dir),
                write_graph=False,
                update_freq="epoch"
            )
        )
    
    return callbacks

