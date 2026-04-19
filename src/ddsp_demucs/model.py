"""Model architectures: DDSP decoder, residual models, and spectral mask models."""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import ddsp
from ddsp.synths import Harmonic, FilteredNoise
from ddsp.effects import Reverb, FIRFilter
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
                 harmonic_base_gain: float = 0.5,
                 harmonic_voiced_gain: float = 0.5,
                 noise_base_gain: float = 0.6,
                 noise_unvoiced_gain: float = 0.8,
                 noise_transient_gain: float = 0.8,
                 z_encoder: Optional[keras.layers.Layer] = None,
                 use_learned_output_gate: bool = False,
                 consonant_noise_enhancements: bool = False,
                 voiced_transient_noise_leak: float = 0.32,
                 use_noise_ducking: bool = False,
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
            use_learned_output_gate: If True, multiply harmonic and noise audio by a
                frame-rate gate resampled to samples: smooth(sigmoid(Dense(x))) times
                a smooth sigmoid of loudness_db (learnable center/scale), then linearly
                resampled to waveform length before the harmonic+noise sum.
            consonant_noise_enhancements: If True, add sharper noise dynamics: per-frame
                noise envelope head (unpooled), transient-driven noise on weakly voiced
                frames, and (with output gate) a transient-boosted noise gate vs smooth
                harmonic gate.
            voiced_transient_noise_leak: When enhancements are on, fraction [0,1] of
                transient noise emphasis that also applies on voiced frames (consonants
                in vowel context). Ignored when consonant_noise_enhancements is False.
            use_noise_ducking: If True, attenuate the noise waveform (not harmonics) on
                sustained voiced frames so consonant bursts can decay toward a more tonal
                mix. Applied after branch gates, before harmonic+noise sum.
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands
        self.f0midi_range = f0_midi_range
        self.z_dims = z_dims
        self.harmonic_base_gain = float(harmonic_base_gain)
        self.harmonic_voiced_gain = float(harmonic_voiced_gain)
        self.noise_base_gain = float(noise_base_gain)
        self.noise_unvoiced_gain = float(noise_unvoiced_gain)
        self.noise_transient_gain = float(noise_transient_gain)
        self.use_learned_output_gate = bool(use_learned_output_gate)
        self.consonant_noise_enhancements = bool(consonant_noise_enhancements)
        self.voiced_transient_noise_leak = float(
            min(1.0, max(0.0, voiced_transient_noise_leak))
        )
        self.use_noise_ducking = bool(use_noise_ducking)
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
        self.harmonic_filter_head = keras.layers.Dense(n_noise_bands)
        self.noise_filter_head = keras.layers.Dense(n_noise_bands)
        self.transient_head = keras.layers.Dense(1)
        if self.consonant_noise_enhancements:
            # Sharp per-frame noise loudness (no temporal pooling) for consonant bursts.
            self.noise_envelope_head = keras.layers.Dense(1)
        if self.use_noise_ducking:
            self.noise_duck_head = keras.layers.Dense(1)
            self._noise_duck_strength_raw = self.add_weight(
                name="noise_duck_strength_raw",
                shape=(),
                initializer=keras.initializers.Constant(1.0),
                trainable=True,
                dtype=tf.float32,
            )
        if self.use_learned_output_gate:
            self.gate_head = keras.layers.Dense(1)
            self.gate_ld_bias = self.add_weight(
                name="gate_ld_bias",
                shape=(),
                initializer=keras.initializers.Constant(-55.0),
                trainable=True,
                dtype=tf.float32,
            )
            self.gate_ld_scale = self.add_weight(
                name="gate_ld_scale",
                shape=(),
                initializer=keras.initializers.Constant(0.2),
                trainable=True,
                dtype=tf.float32,
            )
            if self.consonant_noise_enhancements:
                # Scales how much transient_head opens the noise branch of the output gate.
                self.noise_gate_transient_boost = self.add_weight(
                    name="noise_gate_transient_boost",
                    shape=(),
                    initializer=keras.initializers.Constant(1.15),
                    trainable=True,
                    dtype=tf.float32,
                )

        # Synths and effects
        self.harm = Harmonic(sample_rate=sample_rate, amp_resample_method='linear')
        self.noise = FilteredNoise(
            n_samples=int(sample_rate * 4.0),  # 4 second default
            scale_fn=ddsp.core.exp_sigmoid,
            initial_bias=-5.0
        )
        self.harmonic_fir = FIRFilter(window_size=257, scale_fn=ddsp.core.exp_sigmoid)
        self.noise_fir = FIRFilter(window_size=257, scale_fn=ddsp.core.exp_sigmoid)
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
        # Separate DDSP FIR envelopes for harmonic and noise branches.
        harmonic_filter_mag = ddsp.core.exp_sigmoid(self.harmonic_filter_head(x))
        noise_filter_mag = ddsp.core.exp_sigmoid(self.noise_filter_head(x))

        # Voicing-aware control: suppress harmonics for unvoiced frames.
        voiced = tf.cast(f0_hz > 1.0, tf.float32)[..., tf.newaxis]  # [B, T, 1]
        voiced = tf.nn.avg_pool1d(voiced, ksize=5, strides=1, padding="SAME")
        voiced = tf.clip_by_value(voiced, 0.0, 1.0)
        unvoiced = 1.0 - voiced

        # Extra transient/noise emphasis helps plosives and sibilance.
        transient_env = tf.nn.sigmoid(self.transient_head(x))  # [B, T, 1]
        amp = amp * (self.harmonic_base_gain + self.harmonic_voiced_gain * voiced)
        noise_mag = noise_mag * (self.noise_base_gain + self.noise_unvoiced_gain * unvoiced)
        if self.consonant_noise_enhancements:
            # Let consonants on voiced frames drive noise, not only unvoiced F0 bins.
            t_mask = unvoiced + self.voiced_transient_noise_leak * voiced
            noise_mag = noise_mag * (1.0 + self.noise_transient_gain * transient_env * t_mask)
            noise_env = 0.12 + 1.88 * tf.nn.sigmoid(self.noise_envelope_head(x))
            noise_mag = noise_mag * noise_env
        else:
            noise_mag = noise_mag * (1.0 + self.noise_transient_gain * transient_env * unvoiced)
        
        # Synthesize
        f0_hz_3d = f0_hz[..., tf.newaxis]  # [B, T, 1]
        audio_h = self.harm(amplitudes=amp, harmonic_distribution=harm_dist, f0_hz=f0_hz_3d)
        audio_n = self.noise(magnitudes=noise_mag)
        if x_in is not None:
            filtered_h = self.harmonic_fir(audio=audio_h, magnitudes=harmonic_filter_mag)
            if filtered_h.shape.rank == 3:
                filtered_h = filtered_h[..., 0]
            audio_h = tf.cast(filtered_h, tf.float32)
            filtered_n = self.noise_fir(audio=audio_n, magnitudes=noise_filter_mag)
            if filtered_n.shape.rank == 3:
                filtered_n = filtered_n[..., 0]
            audio_n = tf.cast(filtered_n, tf.float32)
        
        # Align lengths
        min_len = tf.minimum(tf.shape(audio_h)[-1], tf.shape(audio_n)[-1])
        audio_h = audio_h[..., :min_len]
        audio_n = audio_n[..., :min_len]

        if self.use_learned_output_gate:
            # Frame-rate multiplicative gate (learned + loudness), temporally smoothed,
            # then resampled to audio samples. Harmonics use the smooth gate; with
            # consonant enhancements, noise uses the same base gate times a transient
            # boost (sharp — no extra pooling) so sibilants/plosives can punch through.
            g_learn = tf.nn.sigmoid(self.gate_head(x))  # [B, T, 1]
            g_learn = tf.nn.avg_pool1d(g_learn, ksize=9, strides=1, padding="SAME")
            g_learn = tf.clip_by_value(g_learn, 0.0, 1.0)
            ld_db_3 = ld_db[..., tf.newaxis] if ld_db.shape.rank == 2 else ld_db
            ld_factor = tf.nn.sigmoid((ld_db_3 - self.gate_ld_bias) * self.gate_ld_scale)
            ld_factor = tf.nn.avg_pool1d(ld_factor, ksize=5, strides=1, padding="SAME")
            ld_factor = tf.clip_by_value(ld_factor, 0.0, 1.0)
            gate_frame = g_learn * ld_factor
            gate_frame = tf.nn.avg_pool1d(gate_frame, ksize=3, strides=1, padding="SAME")
            gate_frame = tf.clip_by_value(gate_frame, 0.0, 1.0)
            n_samples = tf.shape(audio_h)[-1]
            gate_audio_h = ddsp.core.resample(
                tf.squeeze(gate_frame, axis=-1), n_samples, method="linear"
            )
            gate_audio_h = tf.clip_by_value(gate_audio_h, 0.0, 1.0)
            audio_h = audio_h * gate_audio_h
            if self.consonant_noise_enhancements:
                boost = tf.nn.relu(self.noise_gate_transient_boost)
                gate_n = gate_frame * (1.0 + boost * transient_env)
                gate_n = tf.clip_by_value(gate_n, 0.0, 2.5)
                gate_audio_n = ddsp.core.resample(
                    tf.squeeze(gate_n, axis=-1), n_samples, method="linear"
                )
                gate_audio_n = tf.clip_by_value(gate_audio_n, 0.0, 2.5)
                audio_n = audio_n * gate_audio_n
            else:
                audio_n = audio_n * gate_audio_h

        if self.use_noise_ducking:
            # Noise-only ducking: keep noise full on unvoiced / transients; attenuate on
            # sustained voiced (vowel-like) so harmonics dominate, closer to singing.
            transient_smooth = tf.nn.avg_pool1d(
                transient_env, ksize=5, strides=1, padding="SAME"
            )
            transient_smooth = tf.clip_by_value(transient_smooth, 0.0, 1.0)
            sustained_voiced = voiced * (1.0 - transient_smooth)
            sustained_voiced = tf.clip_by_value(sustained_voiced, 0.0, 1.0)
            duck_w = tf.nn.sigmoid(self._noise_duck_strength_raw)
            structure = 1.0 - duck_w * sustained_voiced
            structure = tf.clip_by_value(structure, 0.06, 1.0)
            fine = 0.18 + 0.82 * tf.nn.sigmoid(self.noise_duck_head(x))
            duck_frame = structure * fine
            duck_frame = tf.clip_by_value(duck_frame, 0.06, 1.0)
            n_d = tf.shape(audio_n)[-1]
            duck_audio = ddsp.core.resample(
                tf.squeeze(duck_frame, axis=-1), n_d, method="linear"
            )
            duck_audio = tf.clip_by_value(duck_audio, 0.06, 1.0)
            audio_n = audio_n * duck_audio

        audio = audio_h + audio_n
        audio = self.reverb(audio)
        
        return audio


class ResidualDDSPDecoder(DDSPDecoder):
    """DDSP decoder that predicts residual correction to input."""
    
    def __init__(self, dry_logit_init: float = -6.0, **kwargs):
        """Args:
            dry_logit_init: Initial value for the global dry/wet logit before `sigmoid`.
                Default -6 ⇒ ~0.25% x_in, ~99.75% synth. Use 0.0 for ~50/50 x_in vs synth at init.
        """
        super().__init__(**kwargs)
        # Learnable dry/wet mix (must use add_weight so Keras tracks it for train/save/load).
        self.dry_logit = self.add_weight(
            name="dry_logit",
            shape=(),
            initializer=keras.initializers.Constant(float(dry_logit_init)),
            trainable=True,
            dtype=tf.float32,
        )
    
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

