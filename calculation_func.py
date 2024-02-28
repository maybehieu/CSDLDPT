import librosa
import numpy as np


def read_audio_from_path(_path=r''):
    try:
        y, sr = librosa.load(_path)
        return y, sr
    except:
        print(f'Error loading path {_path}')
    return None


def calculate_average_power(y, sr):
    # calculate STFT
    D = librosa.stft(y)
    # calculate power
    power = np.abs(D) ** 2
    # mean
    average_power = np.mean(power)
    return average_power
