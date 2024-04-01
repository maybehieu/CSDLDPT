import os.path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_all_features, create_feature

filepath = r'datasets/Test/474.wav'
filename, ext = os.path.splitext(os.path.basename(filepath))

all_feats = load_all_features(r'features', mode=1)

query = create_feature(filepath, mode=1)

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
feats = []
i = 0
for (feat, filepath) in all_feats:
    print(i)
    i += 1
    centroid = cosine_similarity(
        feat["centroid"][0], query["centroid"][0]
    )
    spectral_centroids.append((filepath, np.mean(centroid)))
    chroma = cosine_similarity(
        feat["chroma"][0], query["chroma"][0]
    )
    chromas.append((filepath, np.mean(np.mean(chroma, axis=0))))
    power = cosine_similarity(
        feat["power"][0], query["power"][0]
    )
    powers.append((filepath, np.mean(power)))
    stft_spectro = cosine_similarity(
        feat["stft_spectro"][0], query["stft_spectro"][0]
    )
    stftspectros.append((filepath, np.mean(stft_spectro)))
    mel_spectro = cosine_similarity(
        feat["mel_spectro"][0], query["mel_spectro"][0]
    )
    melspectros.append((filepath, np.mean(mel_spectro)))
    bandwidth = cosine_similarity(
        feat["spectral_bandwidth"][0], query["spectral_bandwidth"][0]
    )
    bandwidths.append((filepath, np.mean(bandwidth)))
    contrast = cosine_similarity(
        feat["spectral_contrast"][0], query["spectral_contrast"][0]
    )
    contrasts.append((filepath, np.mean(contrast)))
    flatness = cosine_similarity(
        feat["spectral_flatness"][0], query["spectral_flatness"][0]
    )
    flatnesses.append((filepath, np.mean(flatness)))
    rolloff = cosine_similarity(
        feat["spectral_rolloff"][0], query["spectral_rolloff"][0]
    )
    rolloffs.append((filepath, np.mean(rolloff)))
    onset = cosine_similarity(
        feat["onset_env"][0], query["onset_env"][0]
    )
    onsets.append((filepath, np.mean(onset)))

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
    feats.append((filepath, np.mean(all_feat)))

spectral_centroids.sort(reverse=True, key=lambda x: x[1])
chromas.sort(reverse=True, key=lambda x: x[1])
powers.sort(reverse=True, key=lambda x: x[1])
stftspectros.sort(reverse=True, key=lambda x: x[1])
melspectros.sort(reverse=True, key=lambda x: x[1])
bandwidths.sort(reverse=True, key=lambda x: x[1])
flatnesses.sort(reverse=True, key=lambda x: x[1])
contrasts.sort(reverse=True, key=lambda x: x[1])
rolloffs.sort(reverse=True, key=lambda x: x[1])
onsets.sort(reverse=True, key=lambda x: x[1])
feats.sort(reverse=True, key= lambda x: x[1])

with open(os.path.join('demo_output', filename + '.txt'), 'a') as f:
    print("feat", feats[:5], file=f)
    print("centroid", spectral_centroids[:5], file=f)
    print("chrom", chromas[:5], file=f)
    print("pow", powers[:5], file=f)
    print("stft", stftspectros[:5], file=f)
    print("mel", melspectros[:5], file=f)
    print("band", bandwidths[:5], file=f)
    print("cons", contrasts[:5], file=f)
    print("roll", rolloffs[:5], file=f)
    print("ons", onsets[:5], file=f)
    print("flat", flatnesses[:5], file=f)
    print("", file=f)
