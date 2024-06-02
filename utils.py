import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import time
from typing import Optional
from tqdm import tqdm
import csv


def read_audio_from_path(_path=r""):
    try:
        y, sr = librosa.load(_path, sr=44100)
        # try:
        #     y, _ = librosa.effects.trim(y)
        # except:
        #     print(f'Error trimming file {_path}')
        return y, sr
    except:
        print(f"Error loading path {_path}")
    return None


def norm(
    S: np.ndarray,
    *,
    norm: Optional[float] = 1,
    axis: Optional[int] = 0,
    threshold: Optional[float] = None,
    fill: Optional[bool] = None,
) -> np.ndarray:
    """Normalize an array along a chosen axis."""

    # Avoid div-by-zero
    if threshold is None:
        threshold = np.finfo(S.dtype).tiny

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm is None:
        return S
    elif norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)
    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)
    elif norm == 0:
        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)
    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)
        if axis is None:
            fill_norm = mag.size ** (-1.0 / norm)
        else:
            fill_norm = mag.shape[axis] ** (-1.0 / norm)

    # indices where norm is below the threshold
    small_idx = length < threshold

    Snorm = np.empty_like(S)
    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm[:] = S / length
    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm[:] = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm[:] = S / length

    return Snorm


def power_to_db(
    S: np.ndarray,
    ref: Optional[float] = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> np.ndarray:
    """Convert a power spectrogram (amplitude squared) to dB-scaled spectrogram."""
    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        magnitude = np.abs(S)
    else:
        magnitude = S

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def expand_to(x, ndim, axes):
    # Force axes into a tuple
    axes_tup = None
    try:
        axes_tup = tuple(axes)  # type: ignore
    except TypeError:
        axes_tup = tuple([axes])  # type: ignore

    shape = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def summarize_audio_files(mother_direc):
    print(os.listdir(mother_direc))
    for sub in [
        os.path.normpath(mother_direc + "/" + _) for _ in os.listdir(mother_direc)
    ]:
        print(sub)
        audio_files = [
            file for file in os.listdir(sub) if file.endswith((".wav", ".aif"))
        ]

        # attack_times = []
        # audio_lengths = []
        spectral_centroids = []
        chromas = []
        powers = []
        stftspectros = []
        melspectros = []
        bandwidths = []
        contrasts = []
        flatnesses = []
        rolloffs = []
        onsets = []

        for audio_file in audio_files:
            # Load audio file
            audio_path = os.path.join(sub, audio_file)
            trimmed_audio, sr = librosa.load(audio_path)

            stft = librosa.stft(trimmed_audio)

            # Trim audio to remove silence
            # trimmed_audio, _ = librosa.effects.trim(y)

            # Calculate spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(
                y=trimmed_audio, sr=sr
            )[0]
            spectral_centroids.append((spectral_centroid, audio_file))

            # # Calculate chromatogram
            chroma = librosa.feature.chroma_stft(y=trimmed_audio, sr=sr)
            # print(len(chroma))
            chromas.append(chroma)

            # Calculate power
            power = librosa.power_to_db(np.abs(stft**2))
            # print(len(power), len(power[0]))
            powers.append(power)

            # Calculate stft spectrogram
            stftspectro = np.abs(stft)
            stftspectros.append((stftspectro, audio_file))

            # Calculate mel-spectrogram
            melspectro = librosa.feature.melspectrogram(y=trimmed_audio, sr=sr)
            melspectros.append((melspectro, audio_file))

            # Calculate spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                S=np.abs(stft), sr=sr
            )
            bandwidths.append((spectral_bandwidth, audio_file))

            # Calculate spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(stft))
            contrasts.append((spectral_contrast, audio_file))

            # Calculate spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(S=np.abs(stft))
            flatnesses.append((spectral_flatness, audio_file))

            # Calculate spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft))
            rolloffs.append((spectral_rolloff, audio_file))

            # Calculate onset envelope
            onset_env = librosa.onset.onset_strength(y=trimmed_audio, sr=sr)
            onsets.append((onset_env, audio_file))

            #
            # # Compute frames where onsets occur
            # frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            #
            # # Convert frames to time
            # attack_time = librosa.frames_to_time(frames, sr=sr)
            # attack_times.append((attack_time, audio_file))

            # # Calculate audio length
            # audio_length = librosa.get_duration(y=trimmed_audio, sr=sr)
            # audio_lengths.append((audio_length, audio_file))

        # NEW
        # Plot spectral centroid
        plt.figure(figsize=(25, 150))
        row = int(len(spectral_centroids) / 4 + 1)
        col = 4
        # print(row, col)
        for i, centroid in enumerate(spectral_centroids):
            plt.subplot(row, col, i + 1)
            plt.plot(librosa.times_like(centroid[0]), centroid[0])
            plt.title(centroid[1])
            plt.xlabel("t")
        plt.suptitle(f"Spectral Centroid {sub}")
        plt.tight_layout()
        plt.legend()
        plt.show()

        # # OLD
        # Plot spectral centroids
        # plt.figure(figsize=(10, 6))
        # for i, centroid in enumerate(spectral_centroids):
        #     plt.plot(librosa.times_like(centroid[0]), centroid[0], label=f'{centroid[1]}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Spectral Centroid')
        # plt.title(f'Spectral Centroid of Audio Files ({sub})')
        # plt.legend()
        # plt.tight_layout()
        # # plt.savefig('spectral_centroids.png')
        # # plt.close()
        # plt.show()

        # # Plot attack times
        # plt.figure(figsize=(10, 6))
        # for i, attack in enumerate(attack_times):
        #     plt.plot(attack[0], np.ones_like(attack[0]) * i, '|', label=f'{attack[1]}')
        # plt.yticks(np.arange(len(audio_files)), [f'{attack[1]}' for i in range(len(audio_files))])
        # plt.xlabel('Time (s)')
        # plt.ylabel('Audio File')
        # plt.title(f'Attack Times of Audio Files ({sub})')
        # plt.tight_layout()
        # # plt.savefig('attack_times.png')
        # # plt.close()
        # plt.show()

        # # Plot chromatogram
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
        #
        # plt.figure(figsize=(10, 6))
        # librosa.display.specshow(padded, y_axis='chroma', x_axis='time')
        # plt.colorbar()
        # plt.title(f'Chromatogram ({sub})')
        # plt.show()

        # plt.figure(figsize=(10, 20))
        # for i, S_db in enumerate(powers):
        #     plt.subplot(len(powers), 1, i + 1)
        #     librosa.display.specshow(S_db, x_axis='time', y_axis='mel')
        #     plt.colorbar(format='%+2.0f dB')
        #     plt.title(f'Spectrogram of {audio_files[i]}')
        #     time.sleep(0.1)
        # # plt.tight_layout()
        # plt.show()

        # # # Plot onset
        # plt.figure(figsize=(10, 6))
        # for i, onset in enumerate(onsets):
        #     plt.plot(onset[0], label=f'{onset[1]}')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Audio File')
        # plt.title(f'Attack Times of Audio Files ({sub})')
        # plt.tight_layout()
        # # plt.savefig('attack_times.png')
        # # plt.close()
        # plt.show()

        # Print average length of audio
        # average_length = np.mean([tup[0] for tup in audio_lengths])
        # print(f"Average length of audio files ({sub}): {average_length:.2f} seconds")


def run_func_on_all_datasets(mother_dir, func, arg1):
    res = []
    subfolder = [
        os.path.normpath(mother_dir + "/" + _)
        for _ in os.listdir(mother_dir)
        if os.path.isdir(os.path.normpath(mother_dir + "/" + _))
    ]
    if len(subfolder) == 0:
        audio_files = [
            file for file in os.listdir(mother_dir) if file.endswith((".wav", ".aif"))
        ]
        for audio_file in tqdm(audio_files):
            audio_path = os.path.normpath(os.path.join(mother_dir, audio_file))
            if arg1 is not None:
                res.append((audio_path, func(audio_path, arg1)))
            else:
                res.append(func(audio_path))
    for sub in subfolder:
        print(sub)
        audio_files = [
            file for file in os.listdir(sub) if file.endswith((".wav", ".aif"))
        ]
        for audio_file in tqdm(audio_files):
            audio_path = os.path.normpath(os.path.join(sub, audio_file))
            if arg1 is not None:
                res.append((audio_path, func(audio_path, arg1)))
            else:
                res.append(func(audio_path))

    return res


def plot_all_chromatogram(sub):
    audio_files = [file for file in os.listdir(sub) if file.endswith((".wav", ".aif"))]

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
        librosa.display.specshow(chroma, y_axis="chroma", x_axis="time")
        plt.colorbar()
        plt.title(f"Chromatogram ({audio_file})")
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


def move_audio_time_based(audio_path, output_path="datasets/Scraps/"):
    y, sr = read_audio_from_path(audio_path)
    audio_length = librosa.get_duration(y=y, sr=sr)
    if audio_length > 1.0:
        print(f"moving: {audio_path}")
        index = audio_path.rfind("\\")
        index = audio_path.rfind("\\", 0, index)
        file_context = audio_path[index + 1 :]
        output_path = os.path.normpath(output_path + file_context)
        # create dir if not exists
        dir = os.path.dirname(output_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.move(audio_path, output_path)


def create_feature(path, mode=0):
    trimmed_audio, sr = librosa.load(path, sr=44100)

    stft = librosa.stft(trimmed_audio)

    # Calculate spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=trimmed_audio, sr=sr)

    # # Calculate chromatogram
    chroma = librosa.feature.chroma_stft(y=trimmed_audio, sr=sr)

    # Calculate power
    power = librosa.power_to_db(np.abs(stft**2))

    # Calculate stft spectrogram
    stftspectro = np.abs(stft)

    # Calculate mel-spectrogram
    melspectro = librosa.feature.melspectrogram(y=trimmed_audio, sr=sr)

    # Calculate spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=sr)

    # Calculate spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(stft))

    # Calculate spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(S=np.abs(stft))

    # Calculate spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft))

    # Calculate onset envelope
    onset_env = librosa.onset.onset_strength(y=trimmed_audio, sr=sr)
    onset_env = onset_env.reshape(-1, 1)

    # print(spectral_centroid.shape,chroma.shape,power.shape,stftspectro.shape,melspectro.shape,
    #       spectral_bandwidth.shape,spectral_contrast.shape,spectral_flatness.shape,spectral_rolloff.shape,onset_env.shape)

    # Create the feature vector
    f_dtype = [
        ("centroid", "f4", spectral_centroid.shape),
        ("chroma", "f4", chroma.shape),
        ("power", "f4", power.shape),
        ("stft_spectro", "f4", stftspectro.shape),
        ("mel_spectro", "f4", melspectro.shape),
        ("spectral_bandwidth", "f4", spectral_bandwidth.shape),
        ("spectral_contrast", "f4", spectral_contrast.shape),
        ("spectral_flatness", "f4", spectral_flatness.shape),
        ("spectral_rolloff", "f4", spectral_rolloff.shape),
        ("onset_env", "f4", onset_env.shape),
    ]

    feature_vector = np.empty(1, dtype=f_dtype)

    feature_vector["centroid"] = spectral_centroid
    feature_vector["chroma"] = chroma
    feature_vector["power"] = power
    feature_vector["stft_spectro"] = stftspectro
    feature_vector["mel_spectro"] = melspectro
    feature_vector["spectral_bandwidth"] = spectral_bandwidth
    feature_vector["spectral_contrast"] = spectral_contrast
    feature_vector["spectral_flatness"] = spectral_flatness
    feature_vector["spectral_rolloff"] = spectral_rolloff
    feature_vector["onset_env"] = onset_env

    if mode == 0:
        filename, ext = os.path.splitext(os.path.basename(path))
        np.save(os.path.normpath("features/" + filename + ".npy"), feature_vector)
    else:
        return feature_vector


def load_feature(path):
    return np.load(os.path.normpath(path))


def load_all_features(path, mode=0):
    # feature_files = [file for file in os.listdir(path) if file.endswith(".npy")]
    # paths = [os.path.join(path, file) for file in feature_files]
    # if mode == 0:
    #     return [np.load(path) for path in paths]
    # else:
    #     return [(np.load(path), path) for path in paths]

    # # different version using os.walk
    feature_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".npy"):
                feature_files.append(os.path.join(root, file))
    if mode == 0:
        return [np.load(path) for path in feature_files]
    else:
        return [(np.load(path), path) for path in feature_files]


def build_file_structure(dir1, dir2, output_csv):
    matching_files = []

    for root, _, files in os.walk(dir1):
        for file in files:
            filename1, _ = os.path.splitext(file)
            for root2, _, files2 in os.walk(dir2):
                for file2 in files2:
                    filename2, _ = os.path.splitext(file2)
                    if filename1 == filename2:
                        matching_files.append(
                            (
                                os.path.normpath(os.path.join(root, file)),
                                os.path.normpath(os.path.join(root2, file2)),
                            )
                        )
                        break

    # Write the matching file paths to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["feature", "Audio File"])
        writer.writerows(matching_files)


def rename_file_name(file_path):
    os.rename(file_path, file_path.replace(" (mp3cut.net)", ""))


if __name__ == "__main__":
    build_file_structure(
        "features_v3", "datasets/Processed_v3", "file_structure_v3.csv"
    )
