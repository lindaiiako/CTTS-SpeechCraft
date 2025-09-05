import librosa
import numpy as np
import pyworld as pw

DAC_SR = 44100
FRAME_PERIOD = 513 / 44100 * 1000


def extract_pitch(wav):
    _f0, t = pw.dio(wav, DAC_SR, f0_ceil=400, frame_period=FRAME_PERIOD)
    f0 = pw.stonemask(wav, _f0, t, DAC_SR)
    voiced_f0 = f0[f0 > 0]
    return voiced_f0


def process_audio(path):
    # We always load audio in 44100 SR
    wav, _ = librosa.load(path, sr=DAC_SR, mono=True)
    f0_values = extract_pitch(wav.astype(np.float64))
    mean_pitch = f0_values.mean()
    energy = librosa.feature.rms(y=wav)
    mean_energy = np.mean(energy[~np.isnan(energy)])

    return mean_pitch, mean_energy