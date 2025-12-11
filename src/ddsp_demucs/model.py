"""Model architectures: DDSP decoder, residual models, and spectral mask models."""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import ddsp
from ddsp.synths import Harmonic, FilteredNoise
from ddsp.effects import Reverb
from typing import Dict, Tuple, Optional


class DDSPDecoder(keras.Model):
    """DDSP decoder with harmonic + noise + reverb."""
    
    def __init__(self,
                 sample_rate: int = 22050,
                 frame_rate: int = 250,
                 n_harmonics: int = 64,
                 n_noise_bands: int = 65,
                 rnn_units: int = 256,
                 mlp_units: Tuple[int, ...] = (256, 128),
                 f0_midi_range: Tuple[float, float] = (24.0, 84.0),
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
            inputs: Dictionary with 'f0_hz' and 'loudness_db'
            training: Whether in training mode
            
        Returns:
            Synthesized audio
        """
        f0_hz = tf.cast(inputs["f0_hz"], tf.float32)
        ld_db = tf.cast(inputs["loudness_db"], tf.float32)
        
        # Convert to MIDI and stack features
        f0_midi = ddsp.core.hz_to_midi(tf.clip_by_value(f0_hz, 1.0, 8000.0))
        f0_midi = tf.clip_by_value(f0_midi, *self.f0midi_range)
        x = tf.stack([f0_midi, ld_db], axis=-1)  # [B, T, 2]
        
        # Process through network
        x = self.pre(x)
        x = self.gru(x)
        x = self.post(x)
        
        # Generate controls
        amp = ddsp.core.exp_sigmoid(self.amp_head(x))  # [B, T, 1]
        harm_dist = tf.nn.softmax(self.harm_head(x), axis=-1)  # [B, T, H]
        noise_mag = ddsp.core.exp_sigmoid(self.noise_head(x))  # [B, T, BANDS]
        
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
                 sample_rate: int = 22050,
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

