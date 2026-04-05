"""Model architectures: DDSP decoder, residual models, and spectral mask models."""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import ddsp
from ddsp.synths import Harmonic, FilteredNoise
from ddsp.effects import Reverb
from ddsp import spectral_ops
from typing import Dict, Tuple, Optional


class ZEncoder(keras.layers.Layer):
    """DDSP-style latent encoder base class producing temporal latent z."""

    def call(self, audio: tf.Tensor, f0_hz: tf.Tensor) -> tf.Tensor:
        """Encode audio to z and align z time-steps to conditioning."""
        time_steps = tf.shape(f0_hz)[1]
        z = self.compute_z(audio)
        return self.expand_z(z, time_steps)

    def compute_z(self, audio: tf.Tensor) -> tf.Tensor:
        """Encode audio and return latent z (override in subclasses)."""
        raise NotImplementedError

    def expand_z(self, z: tf.Tensor, time_steps: tf.Tensor) -> tf.Tensor:
        """Expand/resample z so it matches conditioning frame length."""
        if len(z.shape) == 2:
            z = z[:, tf.newaxis, :]
        z_time_steps = tf.shape(z)[1]
        return tf.cond(
            tf.equal(z_time_steps, time_steps),
            lambda: z,
            lambda: ddsp.core.resample(z, time_steps),
        )


class MfccTimeDistributedRnnEncoder(ZEncoder):
    """MFCC encoder modeled after DDSP's time-distributed RNN encoders."""

    def __init__(self,
                 rnn_channels: int = 512,
                 z_dims: int = 32,
                 z_time_steps: int = 250,
                 **kwargs):
        super().__init__(**kwargs)
        if z_time_steps not in [63, 125, 250, 500, 1000]:
            raise ValueError("z_time_steps must be one of: 63, 125, 250, 500, 1000")

        z_audio_spec = {
            63: {"fft_size": 2048, "overlap": 0.5},
            125: {"fft_size": 1024, "overlap": 0.5},
            250: {"fft_size": 1024, "overlap": 0.75},
            500: {"fft_size": 512, "overlap": 0.75},
            1000: {"fft_size": 256, "overlap": 0.75},
        }
        self.fft_size = z_audio_spec[z_time_steps]["fft_size"]
        self.overlap = z_audio_spec[z_time_steps]["overlap"]
        self.z_norm = keras.layers.LayerNormalization(axis=-1)
        self.rnn = keras.layers.GRU(rnn_channels, return_sequences=True)
        self.dense_out = keras.layers.Dense(z_dims)
        self.z_dims = z_dims

    def compute_z(self, audio: tf.Tensor) -> tf.Tensor:
        # Accept [B, T] or [B, T, 1]
        if audio.shape.rank == 2:
            audio = audio[..., tf.newaxis]
        mfccs = spectral_ops.compute_mfcc(
            audio,
            lo_hz=20.0,
            hi_hz=8000.0,
            fft_size=self.fft_size,
            mel_bins=128,
            mfcc_bins=30,
            overlap=self.overlap,
            pad_end=True,
        )
        z = self.z_norm(mfccs)
        z = self.rnn(z)
        z = self.dense_out(z)
        return z


class DDSPDecoder(keras.Model):
    """DDSP decoder with harmonic + noise + reverb."""
    
    def __init__(self,
                 sample_rate: int = 16000,
                 frame_rate: int = 250,
                 n_harmonics: int = 64,
                 n_noise_bands: int = 65,
                 rnn_units: int = 256,
                 mlp_units: Tuple[int, ...] = (256, 128),
                 f0_midi_range: Tuple[float, float] = (24.0, 84.0),
                 z_dims: int = 32,
                 z_time_steps: int = 250,
                 z_rnn_channels: int = 512,
                 z_encoder: Optional[keras.layers.Layer] = None,
                 **kwargs):
        """Initialize DDSP decoder.
        
        Args:
            sample_rate: Audio sample rate
            frame_rate: Feature frame rate
            n_harmonics: Number of harmonics
            n_noise_bands: Number of noise bands
            rnn_units: GRU units
            mlp_units: MLP layer sizes
            f0_midi_range: F0 range in MIDI notes
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.f0midi_range = f0_midi_range
        self.z_dims = z_dims
        self.z_encoder = z_encoder or MfccTimeDistributedRnnEncoder(
            rnn_channels=z_rnn_channels,
            z_dims=z_dims,
            z_time_steps=z_time_steps,
        )
        
        # Feature encoder
        self.pre = keras.layers.Dense(128, activation='relu')
        self.gru = keras.layers.GRU(rnn_units, return_sequences=True)
        self.post = keras.Sequential([
            keras.layers.Dense(u, activation='relu') for u in mlp_units
        ])
        
        # Output heads
        self.amp_head = keras.layers.Dense(1)
        self.harm_head = keras.layers.Dense(n_harmonics)
        self.noise_head = keras.layers.Dense(n_noise_bands)
        self.noise_detail_head = keras.layers.Dense(n_noise_bands)
        self.transient_head = keras.layers.Dense(1)
        
        # Synths and effects
        self.harm = Harmonic(sample_rate=sample_rate, amp_resample_method='linear')
        self.noise = FilteredNoise(
            n_samples=int(sample_rate * 4.0),  # 4 second default
            scale_fn=ddsp.core.exp_sigmoid,
            initial_bias=-5.0
        )
        self.reverb = Reverb(trainable=True)
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Forward pass.
        
        Args:
            inputs: Dictionary with 'f0_hz', 'loudness_db', and optional 'x_in'
            training: Whether in training mode
            
        Returns:
            Synthesized audio
        """
        f0_hz = tf.cast(inputs["f0_hz"], tf.float32)
        ld_db = tf.cast(inputs["loudness_db"], tf.float32)
        x_in = inputs.get("x_in", None)

        # DDSP-style latent conditioning z from input audio (if available).
        if x_in is not None:
            x_in = tf.cast(x_in, tf.float32)
            z = self.z_encoder(x_in, f0_hz)
        else:
            z = tf.zeros(
                [tf.shape(f0_hz)[0], tf.shape(f0_hz)[1], self.z_dims],
                dtype=tf.float32,
            )
        
        # Convert to MIDI and stack features
        f0_midi = ddsp.core.hz_to_midi(tf.clip_by_value(f0_hz, 1.0, 8000.0))
        f0_midi = tf.clip_by_value(f0_midi, *self.f0midi_range)
        x_base = tf.stack([f0_midi, ld_db], axis=-1)  # [B, T, 2]
        x = tf.concat([x_base, z], axis=-1)  # [B, T, 2 + z_dims]
        
        # Process through network
        x = self.pre(x)
        x = self.gru(x)
        x = self.post(x)
        
        # Generate controls
        amp = ddsp.core.exp_sigmoid(self.amp_head(x))  # [B, T, 1]
        harm_dist = tf.nn.softmax(self.harm_head(x), axis=-1)  # [B, T, H]
        # Two-headed noise control improves high-frequency detail for sibilance.
        noise_logits = self.noise_head(x) + 0.5 * self.noise_detail_head(x)
        noise_mag = ddsp.core.exp_sigmoid(noise_logits)  # [B, T, BANDS]

        # Voicing-aware control: suppress harmonics for unvoiced frames.
        voiced = tf.cast(f0_hz > 1.0, tf.float32)[..., tf.newaxis]  # [B, T, 1]
        voiced = tf.nn.avg_pool1d(voiced, ksize=5, strides=1, padding="SAME")
        voiced = tf.clip_by_value(voiced, 0.0, 1.0)
        unvoiced = 1.0 - voiced

        # Extra transient/noise emphasis helps plosives and sibilance.
        transient_env = tf.nn.sigmoid(self.transient_head(x))  # [B, T, 1]
        amp = amp * (0.5 + 0.5 * voiced)
        noise_mag = noise_mag * (0.6 + 0.8 * unvoiced) * (1.0 + 0.8 * transient_env * unvoiced)
        
        # Synthesize
        f0_hz_3d = f0_hz[..., tf.newaxis]  # [B, T, 1]
        audio_h = self.harm(amplitudes=amp, harmonic_distribution=harm_dist, f0_hz=f0_hz_3d)
        audio_n = self.noise(magnitudes=noise_mag)
        
        # Align lengths
        min_len = tf.minimum(tf.shape(audio_h)[-1], tf.shape(audio_n)[-1])
        audio_h = audio_h[..., :min_len]
        audio_n = audio_n[..., :min_len]
        
        audio = audio_h + audio_n
        audio = self.reverb(audio)
        
        return audio


class ResidualDDSPDecoder(DDSPDecoder):
    """DDSP decoder that predicts residual correction to input."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable dry/wet mix
        self.dry_logit = tf.Variable(-6.0, trainable=True, name="dry_logit")
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Forward pass with residual connection.
        
        Args:
            inputs: Dictionary with 'f0_hz', 'loudness_db', and 'x_in'
            training: Whether in training mode
            
        Returns:
            Corrected audio (dry * x_in + wet * synth)
        """
        # Get base synthesis
        synth = super().call(inputs, training=training)
        
        # Get input if provided
        x_in = inputs.get("x_in", None)
        if x_in is None:
            return synth
        
        # Align lengths
        min_len = tf.minimum(tf.shape(synth)[-1], tf.shape(x_in)[-1])
        synth = synth[..., :min_len]
        x_in = x_in[..., :min_len]
        
        # Dry/wet mix
        dry_g = tf.nn.sigmoid(self.dry_logit)
        return dry_g * x_in + (1.0 - dry_g) * synth


class SpectralMaskEQ(keras.layers.Layer):
    """Time-varying EQ using spectral masking."""
    
    def __init__(self,
                 mel_bins: int = 64,
                 n_bands: int = 65,
                 alpha: float = 0.15,
                 enc_units: int = 64,
                 sample_rate: int = 16000,
                 window_size: int = 176,
                 **kwargs):
        """Initialize spectral mask EQ.
        
        Args:
            mel_bins: Number of mel bins for input
            n_bands: Number of frequency bands
            alpha: Mask deviation scale
            enc_units: Encoder units
            sample_rate: Audio sample rate
            window_size: Window size for frequency filtering
        """
        super().__init__(**kwargs)
        self.mel_bins = mel_bins
        self.n_bands = n_bands
        self.alpha = alpha
        self.sample_rate = sample_rate
        self.window_size = window_size
        
        # Mel encoder
        self.mel_enc = keras.Sequential([
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
        ])
        
        # Temporal processing
        self.pre = keras.layers.Dense(64, activation="relu")
        self.gru = keras.layers.GRU(enc_units, return_sequences=True)
        self.post = keras.layers.Dense(64, activation="relu")
        
        # Mask head
        self.mask_head = keras.layers.Dense(n_bands, activation=None)
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Apply spectral mask to input.
        
        Args:
            inputs: Dictionary with 'x_in', 'mel_in', 'loudness_db'
            training: Whether in training mode
            
        Returns:
            Filtered audio
        """
        x_in = tf.cast(inputs["x_in"], tf.float32)
        mel = tf.cast(inputs["mel_in"], tf.float32)
        ld_db = tf.cast(inputs["loudness_db"], tf.float32)
        
        # Normalize shapes
        if mel.shape.rank == 2:
            mel = mel[tf.newaxis, ...]
        if ld_db.shape.rank == 1:
            ld_db = ld_db[tf.newaxis, ...]
        if x_in.shape.rank == 1:
            x_in = x_in[tf.newaxis, ...]
        
        # Encode features
        z_mel = self.mel_enc(mel)  # [B, T', 32]
        z_ld = ld_db[..., tf.newaxis]  # [B, T', 1]
        z_feat = tf.concat([z_mel, z_ld], axis=-1)  # [B, T', 33]
        
        # Temporal processing
        z = self.pre(z_feat)
        z = self.gru(z)
        z = self.post(z)
        
        # Generate mask
        logits = self.mask_head(z)  # [B, T', n_bands]
        M = 1.0 + self.alpha * tf.tanh(logits)  # ~ [0.85, 1.15]
        M = tf.clip_by_value(M, 0.85, 1.15)
        
        # Time smoothing
        M = tf.nn.avg_pool1d(M, ksize=3, strides=1, padding="SAME")
        
        # Resample mask to match audio frames
        B = tf.shape(x_in)[0]
        T = tf.shape(x_in)[1]
        Tprime = tf.shape(M)[1]
        hop_used = self.window_size // 2
        n_audio_frames = tf.cast(
            tf.math.ceil(tf.cast(T, tf.float32) / float(hop_used)),
            tf.int32
        )
        
        M_match = ddsp.core.resample(M, n_audio_frames, method='linear')
        M_match = tf.clip_by_value(M_match, 1e-3, 8.0)
        
        # Apply frequency filtering
        yhat = ddsp.core.frequency_filter(
            x_in, magnitudes=M_match, window_size=self.window_size
        )
        
        return yhat

