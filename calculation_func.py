import librosa
import numpy as np
import sklearn.metrics.pairwise


from utils import read_audio_from_path


def cosine_similarity(matrix1, matrix2):
    """
    Compute cosine similarity between two matrices.

    Parameters:
    - matrix1, matrix2: Input matrices to compute cosine similarity.

    Returns:
    - Similarity matrix: A matrix containing cosine similarity scores between corresponding columns.
    """

    # Compute dot product between matrix1 and matrix2
    dot_product = np.dot(matrix1.T, matrix2)

    # Compute magnitudes of each matrix
    magnitude1 = np.sqrt(np.sum(matrix1**2, axis=0))
    magnitude2 = np.sqrt(np.sum(matrix2**2, axis=0))

    # Ensure no division by zero
    magnitude1[magnitude1 == 0] = 1
    magnitude2[magnitude2 == 0] = 1

    # Compute cosine similarity
    similarity_matrix = dot_product / (magnitude1.reshape(-1, 1) * magnitude2)

    return similarity_matrix


def cosine_similarity_sklearn(X, Y=None, dense_output=True):
    """
    Compute cosine similarity between samples in X and Y.

    Parameters:
    - X : {array-like, sparse matrix}, shape (n_samples_X, n_features)
        Input data.
    - Y : {array-like, sparse matrix}, shape (n_samples_Y, n_features)
        Input data. If None, the output will be the pairwise
        similarities between all samples in X.
    - dense_output : boolean, default True
        Whether to return dense output even when the input is sparse.

    Returns:
    - similarity : ndarray, shape (n_samples_X, n_samples_Y)
        The cosine similarity between samples in X and Y.
    """
    # Convert input to numpy arrays
    X = np.asarray(X)
    if Y is not None:
        Y = np.asarray(Y)

    # Compute dot product between X and Y
    if Y is not None:
        # Dot product between X and Y
        dot_product = np.dot(X, Y.T)
    else:
        # Dot product of X with itself
        dot_product = np.dot(X, X.T)

    # Compute magnitudes of each sample
    X_norm = np.sqrt(np.sum(X**2, axis=1))
    if Y is not None:
        Y_norm = np.sqrt(np.sum(Y**2, axis=1))
    else:
        Y_norm = X_norm

    # Ensure no division by zero
    X_norm[X_norm == 0] = 1
    if Y is not None:
        Y_norm[Y_norm == 0] = 1

    # Compute cosine similarity
    similarity = dot_product / np.outer(X_norm, Y_norm)

    # Return dense output if requested
    if not dense_output:
        similarity = np.asarray(similarity)

    return similarity


def calculate_average_power(y, sr):
    # calculate STFT
    D = librosa.stft(y)
    # calculate power
    power = np.abs(D) ** 2
    power = librosa.power_to_db(np.abs(D)**2)
    print(power, len(power))
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
    weighted_sum = np.sum(magnitude_spectrum * frequency_vector[:, np.newaxis], axis=0)
    # Calculate the total energy
    total_energy = np.sum(magnitude_spectrum)
    # Calculate the average frequency
    average_frequency = weighted_sum / total_energy
    return np.mean(average_frequency), average_frequency


def calculate_spectrogram(y, sr):
    return np.abs(librosa.stft(y))


def calculate_mel_spectrogram(y, sr):
    return librosa.features.melspectrogram(y=y, sr=sr)


def compare_spectrogram(a = (), b = ()):
    y1, sr1 = a[0], a[1]
    y2, sr2 = b[0], b[1]

    # Compute the spectrograms
    D1 = np.abs(librosa.stft(y1))
    D2 = np.abs(librosa.stft(y2))

    # Ensure that the spectrograms have the same shape
    min_frames = min(D1.shape[1], D2.shape[1])
    D1 = D1[:, :min_frames]
    D2 = D2[:, :min_frames]

    # Compute cosine similarity between the spectrograms
    # similarity_matrix = cosine_similarity(D1.T, D2.T)
    similarity_matrix = cosine_similarity_sklearn(D1.T, D2.T)
    # similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(D1.T, D2.T)

    # Average the similarity scores to get a single similarity value
    mean_similarity = np.mean(similarity_matrix)

    return mean_similarity


def compare_spectrum(a = (), b = ()):
    y1, sr1 = a[0], a[1]
    y2, sr2 = b[0], b[1]

    spectrum1 = np.abs(librosa.stft(y1)).flatten()
    spectrum2 = np.abs(librosa.stft(y2)).flatten()

    return cosine_similarity_sklearn(spectrum1.reshape(-1, 1), spectrum2.reshape(-1, 1))[0, 0]


def calculate_attack_time(y, sr):
    # Compute the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # Find the time of the first peak in the onset envelope
    attack_time = librosa.frames_to_time(librosa.util.peak_pick(onset_env), sr=sr)

    return attack_time


# def calculate_spectral_bandwidth(y, sr):
#     # Compute the spectral bandwidth
#     spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#
#     return spectral_bandwidth


def calculate_average_pitch(y, sr):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)

    max_indexes = np.argmax(magnitudes, axis=0)
    most_significant_pitches = pitches[max_indexes, range(magnitudes.shape[1])]
    return most_significant_pitches


def calculate_mfcc(y, sr):
    # this function doesn't help in this use case, mostly for speechs
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    avg = np.mean(mfccs, axis=1)
    print(avg, len(avg))
    return avg


def calculate_chroma_stft(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma


def calculate_spectral_centroid(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the short-time Fourier transform (STFT) of the audio signal
    D = librosa.stft(y)

    # Calculate the spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(D), sr=sr)

    return spectral_centroid[0]


def calculate_spectral_bandwidth(y, sr):
    S = np.abs(librosa.stft(y))
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    return spectral_bandwidth


def calculate_spectral_contrast(y, sr):
    S = np.abs(librosa.stft(y))
    spectral_contrast = librosa.feature.spectral_contrast(S=S)
    return spectral_contrast


def calculate_spectral_flatness(y, sr):
    S = np.abs(librosa.stft(y))
    spectral_flatness = librosa.feature.spectral_flatness(S=S)
    return spectral_flatness


def calculate_spectral_rolloff(y, sr):
    S = np.abs(librosa.stft(y))
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S)
    return spectral_rolloff


def calculate_harmonicity(y, sr):
    y_harmonic = librosa.effects.harmonic(y)
    return y_harmonic


def calculate_max_frequency(y, sr):
    spectrogram = np.abs(librosa.stft(y))
    peak = np.argmax(spectrogram, axis=0)
    peak_hz = librosa.fft_frequencies(sr=sr, n_fft=spectrogram.shape[0])[peak]
    return peak_hz


def calculate_onset_envelope(y, sr):
    return librosa.onset.onset_strength(y=y, sr=sr)