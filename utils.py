import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def read_audio_from_path(_path=r''):
    try:
        y, sr = librosa.load(_path)
        try:
            y, _ = librosa.effects.trim(y)
        except:
            print(f'Error trimming file {_path}')
        return y, sr
    except:
        print(f'Error loading path {_path}')
    return None


def summarize_audio_files(mother_direc):
    print(os.listdir(mother_direc))
    for sub in [os.path.normpath(mother_direc + '/' + _) for _ in os.listdir(mother_direc)]:
        print(sub)
        audio_files = [file for file in os.listdir(sub) if file.endswith(('.wav', '.aif'))]

        spectral_centroids = []
        attack_times = []
        audio_lengths = []
        chromas = []

        for audio_file in audio_files:
            # Load audio file
            audio_path = os.path.join(sub, audio_file)
            y, sr = librosa.load(audio_path, sr=None)

            # Trim audio to remove silence
            trimmed_audio, _ = librosa.effects.trim(y)

            # Calculate spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=trimmed_audio, sr=sr)[0]
            spectral_centroids.append((spectral_centroid, audio_file))
            # Calculate onset envelope
            onset_env = librosa.onset.onset_strength(y=trimmed_audio, sr=sr)

            # Compute frames where onsets occur
            frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

            # Convert frames to time
            attack_time = librosa.frames_to_time(frames, sr=sr)
            attack_times.append((attack_time, audio_file))

            # Calculate audio length
            audio_length = librosa.get_duration(y=trimmed_audio, sr=sr)
            audio_lengths.append((audio_length, audio_file))

            # Calculate chromatogram
            chroma = librosa.feature.chroma_stft(y=trimmed_audio, sr=sr)
            print(len(chroma[0]))
            chromas.append(chroma)

        # Plot spectral centroids
        plt.figure(figsize=(10, 6))
        for i, centroid in enumerate(spectral_centroids):
            plt.plot(librosa.times_like(centroid[0]), centroid[0], label=f'{centroid[1]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Spectral Centroid')
        plt.title(f'Spectral Centroid of Audio Files ({sub})')
        plt.legend()
        plt.tight_layout()
        # plt.savefig('spectral_centroids.png')
        # plt.close()
        plt.show()

        # Plot attack times
        plt.figure(figsize=(10, 6))
        for i, attack in enumerate(attack_times):
            plt.plot(attack[0], np.ones_like(attack[0]) * i, '|', label=f'{attack[1]}')
        plt.yticks(np.arange(len(audio_files)), [f'{attack[1]}' for i in range(len(audio_files))])
        plt.xlabel('Time (s)')
        plt.ylabel('Audio File')
        plt.title(f'Attack Times of Audio Files ({sub})')
        plt.tight_layout()
        # plt.savefig('attack_times.png')
        # plt.close()
        plt.show()

        # Plot chromatogram
        # add padding to ensure all array have the same shape
        max_shape = max(chroma.shape for chroma in chromas)
        padded = []
        for chroma in chromas:
            pad_rows = max_shape[0] - chroma.shape[0]
            pad_cols = max_shape[1] - chroma.shape[1]
            padded_chroma = np.pad(chroma, ((0, pad_rows), (0, pad_cols)), mode='constant')
            padded.append(padded_chroma)

        padded = np.mean(np.stack(padded, axis=0), axis=0)

        plt.figure(figsize=(10, 6))
        librosa.display.specshow(padded, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chromatogram ({sub})')
        plt.show()

        # Print average length of audio
        average_length = np.mean([tup[0] for tup in audio_lengths])
        print(f"Average length of audio files ({sub}): {average_length:.2f} seconds")


def plot_all_chromatogram(sub):
    audio_files = [file for file in os.listdir(sub) if file.endswith(('.wav', '.aif'))]

    spectral_centroids = []
    attack_times = []
    audio_lengths = []
    chromas = []

    for audio_file in audio_files:
        # Load audio file
        audio_path = os.path.join(sub, audio_file)
        y, sr = librosa.load(audio_path, sr=None)

        # Trim audio to remove silence
        trimmed_audio, _ = librosa.effects.trim(y)

        # Calculate chromatogram
        chroma = librosa.feature.chroma_stft(y=trimmed_audio, sr=sr)
        # chromas.append(chroma)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar()
        plt.title(f'Chromatogram ({audio_file})')
        plt.show()

    # Plot chromatogram
    # # add padding to ensure all array have the same shape
    # max_shape = max(chroma.shape for chroma in chromas)
    # padded = []
    # for chroma in chromas:
    #     pad_rows = max_shape[0] - chroma.shape[0]
    #     pad_cols = max_shape[1] - chroma.shape[1]
    #     padded_chroma = np.pad(chroma, ((0, pad_rows), (0, pad_cols)), mode='constant')
    #     padded.append(padded_chroma)
    #
    # padded = np.mean(np.stack(padded, axis=0), axis=0)