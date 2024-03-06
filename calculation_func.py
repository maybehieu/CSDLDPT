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


def calculate_average_frequency(y, sr):
    # Calculate Short-Time Fourier Transform (STFT)
    D = librosa.stft(y)
    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(D)
    # Calculate the frequency vector
    frequency_vector = librosa.fft_frequencies(sr=sr)
    # Calculate the weighted sum of frequencies
    weighted_sum = np.sum(magnitude_spectrum * frequency_vector)
    # Calculate the total energy
    total_energy = np.sum(magnitude_spectrum)
    # Calculate the average frequency
    average_frequency = weighted_sum / total_energy
    return average_frequency


