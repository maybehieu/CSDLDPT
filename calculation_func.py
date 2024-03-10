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