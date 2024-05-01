import librosa
import numpy as np
from calculation_func import (
    stft,
    alt_spectral_contrast,
    alt_spectral_rolloff,
    alt_spectral_bandwidth,
    alt_spectral_centroid,
    alt_create_feature,
)
from utils import load_all_features, read_audio_from_path
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from itertools import chain


def cal_feature(y, sr):
    _stft = stft(y)
    centroid = alt_spectral_centroid(_stft)
    bandwidth = alt_spectral_bandwidth(_stft)[0]
    contrast = alt_spectral_contrast(_stft)[0]
    rolloff = alt_spectral_rolloff(_stft)[0]

    # Create the feature vector
    f_dtype = [
        ("centroid", "f4", centroid.shape),
        ("bandwidth", "f4", bandwidth.shape),
        ("contrast", "f4", contrast.shape),
        ("rolloff", "f4", rolloff.shape),
    ]

    feature_vector = np.empty(1, dtype=f_dtype)

    feature_vector["centroid"] = centroid
    feature_vector["bandwidth"] = bandwidth
    feature_vector["contrast"] = contrast
    feature_vector["rolloff"] = rolloff
    return feature_vector


class AudioHandler:
    def __init__(self) -> None:
        self.all_feats = load_all_features("alt_features", mode=1)
        self.file_map = pd.read_csv('file_structure.csv')
        self.topk = 3

    def get_sims(self, query):
        sims = []
        for feat, filepath in self.all_feats:
            all_feat = [
                np.mean(cosine_similarity(feat["centroid"], query["centroid"])),
                np.mean(cosine_similarity(feat["bandwidth"], query["bandwidth"])),
                np.mean(cosine_similarity(feat["contrast"], query["contrast"])),
                np.mean(cosine_similarity(feat["rolloff"], query["rolloff"])),
            ]
            sims.append((filepath, all_feat))
        sims.sort(reverse=True, key=lambda x: x[1])
        return sims[:self.topk]

    def get_true_audio(self, feat_name):
        return self.file_map.loc[self.file_map['feature'] == feat_name, 'Audio File'].values[0]

    def get_top_filepath(self, sims):
        sims = list(chain.from_iterable(sims))
        sims.sort(reverse=True, key=lambda x: x[1])
        ret_idx = set()
        ret = []
        for (file_path, score) in sims:
            if len(ret_idx) >= 3: break
            if file_path not in ret_idx:
                ret_idx.add(file_path)
                ret.append((self.get_true_audio(file_path), np.mean(score)))
        return ret

    def query(self, filepath):
        y, sr = read_audio_from_path(filepath)
        sample_windows = 1.0 * sr
        hop_length = 0.5 * sr
        sample_windows, hop_length = int(sample_windows), int(hop_length)
        all_sims = []
        # traverse audio data
        cur = 0
        for i in range(0, len(y) - sample_windows + 1, hop_length):
            window_y = y[i : i + sample_windows]
            feat = cal_feature(window_y, sr)
            all_sims.append(self.get_sims(feat))
            cur = i + sample_windows
        if cur < len(y):
            window_y = y[cur:]
            if len(window_y) < sample_windows:
                window_y = np.concatenate(
                    (window_y, np.zeros(sample_windows - len(window_y)))
                )
            # skip calculating if array contain more than 25% actual audio data (due to padding)
            if np.count_nonzero(window_y) >= 0.1 * len(window_y):
                feat = cal_feature(window_y, sr)
                all_sims.append(self.get_sims(feat))
        print(self.get_top_filepath(all_sims))


if __name__ == "__main__":
    handler = AudioHandler()
    while True:
        path = input("audio file path: ")
        if path == "q" or path == "quit": break
        handler.query(path)
