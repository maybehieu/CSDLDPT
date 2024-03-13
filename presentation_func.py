import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from utils import read_audio_from_path
from calculation_func import *


def plot_spectrogram(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the spectrogram
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def plot_spectrum(audio_path, _type='default', max_freq=2000):
    """
    Plot the spectrum of an audio file.

    Parameters:
    - audio_path: Path to the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)

    if _type == 'default':
        # Compute the spectrum
        spectrum = np.abs(librosa.stft(y))

        # Plot the spectrum
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(librosa.amplitude_to_db(spectrum, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrum')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    if _type == 'line':
        # Compute the Fourier Transform of the audio signal
        Y = np.fft.fft(y)

        # Calculate the frequency bins
        freqs = np.fft.fftfreq(len(Y), 1 / sr)

        # Plot the spectrum using a line plot
        plt.figure(figsize=(10, 6))
        plt.plot(freqs[:len(Y) // 2], np.abs(Y)[:len(Y) // 2])  # Plot only positive frequencies
        plt.title('Spectrum Plot (Amplitude vs Frequency)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, max_freq)
        plt.grid()
        plt.show()


def plot_2_spectrogram(audio_path1, audio_path2):
    # Load the audio files
    y1, sr1 = librosa.load(audio_path1, sr=None)
    y2, sr2 = librosa.load(audio_path2, sr=None)

    # Compute the spectrograms
    D1 = librosa.stft(y1)
    D2 = librosa.stft(y2)

    # Ensure both spectrograms have the same shape
    min_frames = min(D1.shape[1], D2.shape[1])
    D1 = D1[:, :min_frames]
    D2 = D2[:, :min_frames]

    # Convert to decibel scale
    D1_db = librosa.amplitude_to_db(np.abs(D1), ref=np.max)
    D2_db = librosa.amplitude_to_db(np.abs(D2), ref=np.max)

    # Plot the spectrograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(D1_db, sr=sr1, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(D2_db, sr=sr2, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def plot_spectral_centroid(audio_path, spectral_centroid):
    """
    Plot the spectral centroid of an audio file.

    Parameters:
    - audio_path: Path to the audio file.
    - spectral_centroid: Spectral centroid of the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the time array
    times = librosa.times_like(spectral_centroid)

    # Plot the spectral centroid
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.plot(times, spectral_centroid[0], color='r')
    plt.title('Spectral Centroid')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()


def plot_attack_time(audio_path):
    """
    Visualize the attack time of an audio file.

    Parameters:
    - audio_path: Path to the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Calculate the attack time
    attack_time = calculate_attack_time(y, sr)

    # Create time array
    times = librosa.times_like(attack_time)

    # Plot the waveform and attack time
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.plot(times, attack_time, color='r')
    plt.title('Attack Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Attack Time')
    plt.show()


def plot_spectral_bandwidth(audio_path):
    """
    Visualize the spectral bandwidth of an audio file.

    Parameters:
    - audio_path: Path to the audio file.
    """
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Calculate the spectral bandwidth
    spectral_bandwidth = calculate_spectral_bandwidth(y, sr)

    # Create time array
    times = librosa.times_like(spectral_bandwidth)

    # Plot the waveform and spectral bandwidth
    plt.figure(figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.plot(times, spectral_bandwidth[0], color='r')
    plt.title('Spectral Bandwidth')
    plt.xlabel('Time (s)')
    plt.ylabel('Spectral Bandwidth')
    plt.show()