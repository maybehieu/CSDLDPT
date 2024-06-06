import librosa
import numpy as np
import sklearn.metrics.pairwise


from utils import read_audio_from_path, norm, power_to_db, expand_to


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
    power = librosa.power_to_db(np.abs(D) ** 2)
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


def compare_spectrogram(a=(), b=()):
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


def compare_spectrum(a=(), b=()):
    y1, sr1 = a[0], a[1]
    y2, sr2 = b[0], b[1]

    spectrum1 = np.abs(librosa.stft(y1)).flatten()
    spectrum2 = np.abs(librosa.stft(y2)).flatten()

    return cosine_similarity_sklearn(
        spectrum1.reshape(-1, 1), spectrum2.reshape(-1, 1)
    )[0, 0]


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


def stft(y, n_fft=2048, hop_length=None, window="hann"):
    """
    Compute the Short-Time Fourier Transform (STFT) of an audio signal.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        The audio signal.
    n_fft : int > 0 [scalar]
        FFT window size.
    hop_length : int > 0 [scalar]
        Number of samples between successive frames.
        If not given, defaults to ``n_fft // 4``.
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - A window specification (string, tuple, number); see `scipy.signal.get_window`
        - A window function, such as `scipy.signal.windows.hann`
        - A vector or array of length `n_fft`
        Defaults to a raised cosine window (`'hann'`), which is adequate for most applications in audio signal processing.

    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        STFT of `y`.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    # Generate the window function
    window_func = np.hanning(n_fft) if window == "hann" else window

    # Initialize the STFT matrix
    D = np.zeros((n_fft // 2 + 1, len(y) // hop_length), dtype=np.complex64)

    # Apply the window function and compute the STFT for each frame
    for t in range(D.shape[1]):
        start = t * hop_length
        end = start + n_fft
        frame = y[start:end]
        frame = np.pad(frame, (0, n_fft - len(frame)), mode="constant")
        D[:, t] = np.fft.rfft(frame * window_func)

    return D


def alt_spectral_centroid(D, sr=44100, n_fft=2048, hop_length=512):
    """
    More precisely, the centroid at frame ``t`` is defined as [#]_::

        centroid[t] = sum_k S[k, t] * freq[k] / (sum_j S[j, t])

    where ``S`` is a magnitude spectrogram, and ``freq`` is the array of
    frequencies (e.g., FFT frequencies in Hz) of the rows of ``S``.
    :param D:
    :param sr:
    :param n_fft:
    :param hop_length:
    :return:
    """
    # Generate the frequency bins
    n_fft = 2048  # Assuming this is the window size used in the STFT
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    # Calculate the spectral centroid for each frame
    spectral_centroid = np.sum(freqs[:, np.newaxis] * np.abs(D), axis=0) / np.sum(
        np.abs(D), axis=0
    )
    spectral_centroid[np.isnan(spectral_centroid)] = 0

    return spectral_centroid


def alt_spectral_bandwidth(D, sr=44100, n_fft=2048):
    # Compute the power spectrum
    power_spectrum = np.abs(D) ** 2

    freq = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    centroid = alt_spectral_centroid(D)

    # power_spectrum = librosa.util.normalize(power_spectrum, norm=1, axis=-2)
    power_spectrum = norm(power_spectrum)

    deviation = np.abs(freq[:, np.newaxis] - centroid)
    spectral_bandwidth = np.sum(
        power_spectrum * deviation**2, axis=-2, keepdims=True
    ) ** (1.0 / 2)
    return spectral_bandwidth.astype(float)


def alt_spectral_contrast(
    D,
    sr=44100,
    n_fft=2048,
    fmin: float = 200.0,
    n_bands: int = 6,
    quantile: float = 0.02,
):

    # Compute the STFT
    S = np.abs(D)

    freq = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    # shape of valleys and peaks based on spectrogram
    shape = list(S.shape)
    shape[-2] = n_bands + 1
    valley = np.zeros(shape)
    peak = np.zeros_like(valley)

    for k, (f_low, f_high) in enumerate(zip(octa[:-1], octa[1:])):
        current_band = np.logical_and(freq >= f_low, freq <= f_high)

        idx = np.flatnonzero(current_band)

        if k > 0:
            current_band[idx[0] - 1] = True

        if k == n_bands:
            current_band[idx[-1] + 1 :] = True

        sub_band = S[..., current_band, :]

        if k < n_bands:
            sub_band = sub_band[..., :-1, :]

        # Always take at least one bin from each side
        idx = np.rint(quantile * np.sum(current_band))
        idx = int(np.maximum(idx, 1))

        sortedr = np.sort(sub_band, axis=-2)

        valley[..., k, :] = np.mean(sortedr[..., :idx, :], axis=-2)
        peak[..., k, :] = np.mean(sortedr[..., -idx:, :], axis=-2)

    contrast: np.ndarray
    contrast = power_to_db(peak) - power_to_db(valley)
    return contrast


def alt_spectral_rolloff(D, sr=44100, n_fft=2048, roll_percent=0.85):
    # Compute the STFT
    S = np.abs(D)

    freq = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

    if freq.ndim == 1:
        freq = expand_to(freq, ndim=S.ndim, axes=-2)

    total_energy = np.cumsum(S, axis=-2)
    # (channels,freq,frames)

    threshold = roll_percent * total_energy[..., -1, :]

    # reshape threshold for broadcasting
    threshold = np.expand_dims(threshold, axis=-2)

    ind = np.where(total_energy < threshold, np.nan, 1)

    rolloff = np.nanmin(ind * freq, axis=-2, keepdims=True)
    return rolloff


def alt_create_feature(path, mode=0):
    trimmed_audio, sr = librosa.load(path, sr=44100)

    _stft = stft(trimmed_audio)

    # Calculate spectral centroid
    spectral_centroid = alt_spectral_centroid(_stft)

    # Calculate spectral bandwidth
    spectral_bandwidth = alt_spectral_bandwidth(_stft)[0]

    # Calculate spectral contrast
    spectral_contrast = alt_spectral_contrast(_stft)[0]

    # Calculate spectral rolloff
    spectral_rolloff = alt_spectral_rolloff(_stft)[0]

    # Create the feature vector
    f_dtype = [
        ("centroid", "f4", spectral_centroid.shape),
        ("bandwidth", "f4", spectral_bandwidth.shape),
        ("contrast", "f4", spectral_contrast.shape),
        ("rolloff", "f4", spectral_rolloff.shape),
    ]

    feature_vector = np.empty(1, dtype=f_dtype)

    feature_vector["centroid"] = spectral_centroid
    feature_vector["bandwidth"] = spectral_bandwidth
    feature_vector["contrast"] = spectral_contrast
    feature_vector["rolloff"] = spectral_rolloff

    if mode == 0:
        import os

        filename, ext = os.path.splitext(os.path.basename(path))
        np.save(os.path.normpath("alt_features/" + filename + ".npy"), feature_vector)
    else:
        return feature_vector


def create_feature_v2(path, mode=0):
    y, sr = librosa.load(path, sr=44100)
    sample_windows = 1.0 * sr
    hop_length = 0.5 * sr
    sample_windows, hop_length = int(sample_windows), int(hop_length)

    # define feature arrays
    all_power = []
    all_centroid = []
    all_bandwidth = []
    all_contrast = []
    all_rolloff = []
    all_chroma = []

    # traverse audio data
    cur = 0
    num_of_windows = 0
    for i in range(0, len(y) - sample_windows + 1, hop_length):
        window_y = y[i : i + sample_windows]
        _stft = stft(window_y)
        power = np.sum(window_y ** 2) / len(window_y)
        centroid = alt_spectral_centroid(_stft)
        bandwidth = alt_spectral_bandwidth(_stft)[0]
        contrast = alt_spectral_contrast(_stft)[0]
        rolloff = alt_spectral_rolloff(_stft)[0]
        chroma = librosa.feature.chroma_stft(y=window_y, sr=sr).flatten()
        all_power.append(power)
        all_centroid.append(centroid)
        all_bandwidth.append(bandwidth)
        all_contrast.append(contrast)
        all_rolloff.append(rolloff)
        all_chroma.append(chroma)
        cur = i + sample_windows
        num_of_windows += 1
    if cur < len(y):
        window_y = y[cur:]
        if len(window_y) < sample_windows:
            window_y = np.concatenate(
                (window_y, np.zeros(sample_windows - len(window_y)))
            )
        # skip calculating if array only contains 0
        if np.any(window_y != 0):
            _stft = stft(window_y)
            power = np.sum(window_y ** 2) / len(window_y)
            centroid = alt_spectral_centroid(_stft)
            bandwidth = alt_spectral_bandwidth(_stft)[0]
            contrast = alt_spectral_contrast(_stft)[0]
            rolloff = alt_spectral_rolloff(_stft)[0]
            chroma = librosa.feature.chroma_stft(y=window_y, sr=sr).flatten()
            all_power.append(power)
            all_centroid.append(centroid)
            all_bandwidth.append(bandwidth)
            all_contrast.append(contrast)
            all_rolloff.append(rolloff)
            all_chroma.append(chroma)
            num_of_windows += 1

    all_power = np.array(all_power, dtype="f4")
    all_centroid = np.array(all_centroid, dtype="f4")
    all_bandwidth = np.array(all_bandwidth, dtype="f4")
    all_contrast = np.array(all_contrast, dtype="f4")
    all_rolloff = np.array(all_rolloff, dtype="f4")
    all_chroma = np.array(all_chroma, dtype="f4")

    # Create the feature vector
    f_dtype = [
        ("feat_nums", "i4"),
        ("power", "f4", all_power.shape),
        # ("chroma", "f4", all_chroma.shape),
        ("centroid", "f4", all_centroid.shape),
        ("bandwidth", "f4", all_bandwidth.shape),
        # ("contrast", "f4", all_contrast.shape),
        ("rolloff", "f4", all_rolloff.shape),
    ]

    feature_vector = np.empty(1, dtype=f_dtype)

    feature_vector["feat_nums"] = num_of_windows
    feature_vector["power"] = all_power
    # feature_vector["chroma"] = all_chroma
    feature_vector["centroid"] = all_centroid
    feature_vector["bandwidth"] = all_bandwidth
    # feature_vector["contrast"] = all_contrast
    feature_vector["rolloff"] = all_rolloff

    if mode == 0:
        import os

        rel_path = os.path.relpath(path, os.getcwd())
        new_path = os.path.join("features_v3", rel_path)
        new_dir = os.path.dirname(new_path)
        filename, ext = os.path.splitext(os.path.basename(path))
        os.makedirs(new_dir, exist_ok=True)
        np.save(
            os.path.normpath(os.path.join(new_dir, filename + ".npy")), feature_vector
        )
    else:
        return feature_vector


def create_center_feature(directory_path, mode=0):
    import os

    # read all npy files in the directory
    files = [f for f in os.listdir(directory_path) if f.endswith(".npy")]
    all_centroid = np.array()
    all_bandwidth = np.array()
    all_contrast = np.array()
    all_rolloff = np.array()
    all_chroma = np.array()
    for file in files:
        feature = np.load(os.path.join(directory_path, file), allow_pickle=True)
        all_centroid += feature["centroid"][0]
        all_bandwidth += feature["bandwidth"][0]
        all_contrast += feature["contrast"][0]
        all_rolloff += feature["rolloff"][0]
        all_chroma += feature["chroma"][0]

    all_centroid = np.array(all_centroid)
    all_bandwidth = np.array(all_bandwidth)
    all_contrast = np.array(all_contrast)
    all_rolloff = np.array(all_rolloff)
    all_chroma = np.array(all_chroma)

    res = np.array()
    res["centroid"] = np.mean(all_centroid, axis=0)
    res["bandwidth"] = np.mean(all_bandwidth, axis=0)
    res["contrast"] = np.mean(all_contrast, axis=0)
    res["rolloff"] = np.mean(all_rolloff, axis=0)
    res["chroma"] = np.mean(all_chroma, axis=0)

    if mode == 0:
        import os

        filename = os.path.basename(directory_path)
        np.save(os.path.normpath(os.path.join("features_v3", filename + ".npy")), res)
    else:
        return res


def calculate_similarity_between_feats(feat1, feat2):
    sims = []
    centroid = sklearn.metrics.pairwise.cosine_similarity(
        feat1["centroid"][0], feat2["centroid"][0]
    )
    chroma = sklearn.metrics.pairwise.cosine_similarity(
        feat1["chroma"][0], feat2["chroma"][0]
    )
    power = sklearn.metrics.pairwise.cosine_similarity(
        feat1["power"][0], feat2["power"][0]
    )
    stft_spectro = sklearn.metrics.pairwise.cosine_similarity(
        feat1["stft_spectro"][0], feat2["stft_spectro"][0]
    )
    mel_spectro = sklearn.metrics.pairwise.cosine_similarity(
        feat1["mel_spectro"][0], feat2["mel_spectro"][0]
    )
    bandwidth = sklearn.metrics.pairwise.cosine_similarity(
        feat1["spectral_bandwidth"][0], feat2["spectral_bandwidth"][0]
    )
    contrast = sklearn.metrics.pairwise.cosine_similarity(
        feat1["spectral_contrast"][0], feat2["spectral_contrast"][0]
    )
    flatness = sklearn.metrics.pairwise.cosine_similarity(
        feat1["spectral_flatness"][0], feat2["spectral_flatness"][0]
    )
    rolloff = sklearn.metrics.pairwise.cosine_similarity(
        feat1["spectral_rolloff"][0], feat2["spectral_rolloff"][0]
    )
    onset = sklearn.metrics.pairwise.cosine_similarity(
        feat1["onset_env"][0], feat2["onset_env"][0]
    )

    all_feat = [
        np.mean(np.mean(chroma, axis=0)),
        np.mean(centroid),
        np.mean(power),
        np.mean(stft_spectro),
        np.mean(mel_spectro),
        np.mean(bandwidth),
        np.mean(contrast),
        np.mean(flatness),
        np.mean(rolloff),
        np.mean(onset),
    ]
    return np.mean(all_feat)
