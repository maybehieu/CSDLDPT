import os.path

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
from tqdm import tqdm
import soundfile as sfile


def cal_feature(y, sr):
    _stft = stft(y)
    power = np.sum(y ** 2) / len(y)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
    centroid = alt_spectral_centroid(_stft)
    bandwidth = alt_spectral_bandwidth(_stft)[0]
    contrast = alt_spectral_contrast(_stft)[0]
    rolloff = alt_spectral_rolloff(_stft)[0]
    return power, chroma, centroid, bandwidth, contrast, rolloff


def get_feature(y, sr):
    sample_windows = 1.0 * sr
    hop_length = 0.5 * sr
    sample_windows, hop_length = int(sample_windows), int(hop_length)

    # define feature arrays
    all_power = []
    all_chroma = []
    all_centroid = []
    all_bandwidth = []
    all_contrast = []
    all_rolloff = []

    # traverse audio data
    cur = 0
    num_of_windows = 0
    for i in range(0, len(y) - sample_windows + 1, hop_length):
        window_y = y[i : i + sample_windows]
        power, chroma, cent, band, cont, roll = cal_feature(window_y, sr)
        all_power.append(power)
        all_chroma.append(chroma)
        all_centroid.append(cent)
        all_bandwidth.append(band)
        all_contrast.append(cont)
        all_rolloff.append(roll)
        cur = i + sample_windows
        num_of_windows += 1
    if cur < len(y):
        window_y = y[cur:]
        # add padding if remaining data length hasn't reached window size
        if len(window_y) < sample_windows:
            window_y = np.concatenate(
                (window_y, np.zeros(sample_windows - len(window_y)))
            )
        # skip calculating if array only contains 0
        if np.any(window_y != 0):
            power, chroma, cent, band, cont, roll = cal_feature(window_y, sr)
            all_power.append(power)
            all_chroma.append(chroma)
            all_centroid.append(cent)
            all_bandwidth.append(band)
            all_contrast.append(cont)
            all_rolloff.append(roll)
            num_of_windows += 1

    all_power = np.array(all_power, dtype="f4")
    all_chroma = np.array(all_chroma, dtype="f4")
    all_centroid = np.array(all_centroid, dtype="f4")
    all_bandwidth = np.array(all_bandwidth, dtype="f4")
    all_contrast = np.array(all_contrast, dtype="f4")
    all_rolloff = np.array(all_rolloff, dtype="f4")

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
    return feature_vector


class AudioHandler:
    def __init__(self) -> None:
        self.all_feats = load_all_features("features_v3", mode=1)
        self.file_map = pd.read_csv("file_structure_v3.csv")
        self.topk = 3
        self.extract = True
        self.output_dir = "export_v2/"

        self.current_file = ""
        self.current_windowsz = 0
        self.current_filename = ""
        self.cache = None

    def get_sims(self, query):
        all_sims = []
        db_num_windows = 5  # number of windows inside feature in database
        sit_type = 0
        if query["feat_nums"] < db_num_windows:
            sit_type = 1
        for feat, filepath in tqdm(self.all_feats):
            data, slider, window_size = None, None, None
            if sit_type == 0:
                data = query
                slider = feat
                window_size = feat["feat_nums"][0]
            else:
                data = feat
                slider = query
                window_size = query["feat_nums"][0]
            self.current_windowsz = window_size

            sims = []
            for hop in range(0, data["feat_nums"][0] - window_size + 1):
                # need element access [0] because data is stored in feature files as an element inside a list
                # eg: feat["centroid"] = [[data0], [data1], [...]...] -> feat["centroid"][0] = [data0]
                # in which: data0: feature vector of spectral centroid with shape (1, n) or (k, n)
                #           k: number of windows, n: number of elem in feature
               sims.append(
                    [
                        np.mean(
                            cosine_similarity(
                                data["power"][0][hop: window_size + hop, :].reshape(
                                    1, -1
                                ),
                                slider["power"][0].reshape(1, -1),
                            )
                        ),
                        # np.mean(
                        #     cosine_similarity(
                        #         data["chroma"][0][hop : window_size + hop, :].reshape(
                        #             1, -1
                        #         ),
                        #         slider["chroma"][0].reshape(1, -1),
                        #     )
                        # ),
                        np.mean(
                            cosine_similarity(
                                data["centroid"][0][hop : window_size + hop, :].reshape(
                                    1, -1
                                ),
                                slider["centroid"][0].reshape(1, -1),
                            )
                        ),
                        np.mean(
                            cosine_similarity(
                                data["bandwidth"][0][
                                    hop : window_size + hop, :
                                ].reshape(1, -1),
                                slider["bandwidth"][0].reshape(1, -1),
                            )
                        ),
                        # np.mean(cosine_similarity(data["contrast"][0][hop:window_size + hop, :].reshape(1, -1), slider["contrast"][0].reshape(1, -1))),
                        np.mean(
                            cosine_similarity(
                                data["rolloff"][0][hop : window_size + hop, :].reshape(
                                    1, -1
                                ),
                                slider["rolloff"][0].reshape(1, -1),
                            )
                        ),
                    ]
                )
            # find max similarity and its timestamp
            sims = np.array(sims)
            time_index = np.argmax(np.mean(sims, axis=1))
            _max = sims[time_index]
            all_sims.append((filepath, _max, time_index))

        all_sims.sort(reverse=True, key=lambda x: np.mean(x[1]))
        print(all_sims[: self.topk])
        self.cache = all_sims
        return all_sims[: self.topk]

    def get_true_audio(self, feat_name):
        return self.file_map.loc[
            self.file_map["feature"] == feat_name, "Audio File"
        ].values[0]

    def get_result(self, sims):
        ret = []
        for file_path, score, time_index in sims:
            ret.append((self.get_true_audio(file_path), np.mean(score), time_index))
        return ret

    def print(self, res):
        print(f"===================\nResult of {self.current_file}: ")
        rank = 1
        for file_path, score, time_index in res:
            # note to self: redo timestamp calculation -> currently wrong
            print(
                f"Rank {rank}: {file_path}, similarity: {score}, segment: {float(time_index) * .5}-{float(time_index) * .5 + \
                        self.current_windowsz - (self.current_windowsz - 1) * 0.5}s"
            )
            rank += 1
        print("====================")

    def export_to_file(self):
        np.set_printoptions(suppress=True)
        with open(
            os.path.join("export", os.path.basename(self.current_file) + "_v1.5.txt"),
            "a",
        ) as f:
            print("feat", self.cache[:5], file=f)
            print("", file=f)

    def extract_result_audio(self, res):
        if not os.path.exists(self.output_dir + self.current_filename):
            os.makedirs(self.output_dir + self.current_filename)
        rank = 1
        for filepath, score, time_index in res:
            y, sr = read_audio_from_path(filepath)
            sample_windows = 1.0 * sr
            hop_length = 0.5 * sr
            sample_windows, hop_length = int(sample_windows), int(hop_length)
            result = y[
                hop_length * time_index : hop_length * time_index
                + self.current_windowsz * sample_windows
                - (self.current_windowsz - 1) * hop_length
            ]
            sfile.write(
                self.output_dir + self.current_filename + f"/rank_{rank}.wav",
                result,
                sr,
                "PCM_24",
            )
            rank += 1

    def query(self, filepath):
        self.current_file = os.path.basename(filepath)
        filename, ext = os.path.splitext(self.current_file)
        self.current_filename = filename
        y, sr = read_audio_from_path(filepath)
        # feature vector extraction
        feat = get_feature(y, sr)
        # find most similar
        sims = self.get_sims(feat)
        # get actual file accordingly
        res = self.get_result(sims)
        # print result
        self.print(res)
        # extract audio segment
        if self.extract:
            try:
                self.extract_result_audio(res)
            except:
                print("Failed to save query result")


if __name__ == "__main__":
    handler = AudioHandler()
    while True:
        path = input("audio file path: ")
        if path == "q" or path == "quit":
            break
        handler.query(path)
        handler.export_to_file()
