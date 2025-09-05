import librosa
import numpy as np

def extract_pitch(wav, sr):
    f0, voiced_flag, _= librosa.pyin(
                wav, 
                fmin=65, 
                fmax=400,
                sr=sr)
    return f0[voiced_flag]

def process_audio(audio, sr):
    f0_values = extract_pitch(audio, sr = sr)
    mean_pitch = f0_values.mean()
    energy = librosa.feature.rms(y=audio)
    mean_energy = np.mean(energy[~np.isnan(energy)])

    return mean_pitch, mean_energy