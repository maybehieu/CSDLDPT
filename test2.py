import os.path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_all_features, create_feature
from calculation_func import alt_create_feature
from tqdm import tqdm

filepath = r'datasets/Test/422561.wav'
filename, ext = os.path.splitext(os.path.basename(filepath))

# all_feats = load_all_features(r'features', mode=1)
all_feats = load_all_features(r'features_v2', mode=1)

# query = create_feature(filepath, mode=1)
query = alt_create_feature(filepath, mode=1)

spectral_centroids = []
bandwidths = []
contrasts = []
rolloffs = []
feats = []
ave_pows = []
i = 0
for (feat, filepath) in tqdm(all_feats):
    a = feat["centroid"]
    b = query["centroid"]
    centroid = cosine_similarity(
        # feat["centroid"][0], query["centroid"][0]
        feat["centroid"], query["centroid"]
    )
    spectral_centroids.append((filepath, np.mean(centroid)))

    bandwidth = cosine_similarity(
        # feat["spectral_bandwidth"][0], query["spectral_bandwidth"][0]
        feat["bandwidth"][0], query["bandwidth"][0]
    )
    bandwidths.append((filepath, np.mean(bandwidth)))
    contrast = cosine_similarity(
        # feat["spectral_contrast"][0], query["spectral_contrast"][0]
        feat["contrast"][0], query["contrast"][0]
    )
    contrasts.append((filepath, np.mean(contrast)))
    rolloff = cosine_similarity(
        # feat["spectral_rolloff"][0], query["spectral_rolloff"][0]
        feat["rolloff"][0], query["rolloff"][0]
    )
    rolloffs.append((filepath, np.mean(rolloff)))
    ave_pow = cosine_similarity(
        # feat["spectral_rolloff"][0], query["spectral_rolloff"][0]
        feat["ave_pow"], query["ave_pow"]
    )
    ave_pows.append((filepath, np.mean(ave_pow)))

    all_feat = [
        np.mean(centroid),
        np.mean(bandwidth),
        np.mean(contrast),
        np.mean(rolloff),
        np.mean(ave_pow)
    ]
    feats.append((filepath, np.mean(all_feat)))

spectral_centroids.sort(reverse=True, key=lambda x: x[1])
bandwidths.sort(reverse=True, key=lambda x: x[1])
contrasts.sort(reverse=True, key=lambda x: x[1])
rolloffs.sort(reverse=True, key=lambda x: x[1])
ave_pows.sort(reverse=True, key=lambda x: x[1])
feats.sort(reverse=True, key= lambda x: x[1])

if __name__ == "__main__":
    with open(os.path.join('demo_output', filename + '_v1.5.txt'), 'a') as f:
        print("feat", feats[:5], file=f)
        print("chroma", ave_pows[:5], file=f)
        print("centroid", spectral_centroids[:5], file=f)
        print("band", bandwidths[:5], file=f)
        print("cons", contrasts[:5], file=f)
        print("roll", rolloffs[:5], file=f)
        print("", file=f)
