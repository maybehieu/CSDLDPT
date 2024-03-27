import os.path

import librosa
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise

from utils import *
from calculation_func import *
from presentation_func import *


def test_func(audio_path):
    # Compute the RMS value for each frame
    y, sr = librosa.load(audio_path, sr=None)
    y, _ = librosa.effects.trim(y)
    rms = librosa.feature.rms(y=y)
    audio_length = librosa.get_duration(y=y, sr=sr)
    print(sr, len(rms[0]), audio_length)
    # # Plot the RMS values
    # plt.figure(figsize=(10, 4))
    # plt.semilogy(librosa.times_like(rms), rms[0], label='RMS Energy')
    # plt.xlabel('Time (s)')
    # plt.ylabel('RMS Energy')
    # plt.legend()
    # plt.show()


def func(path):
    y, sr = librosa.load(path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    N = len(y)
    T = N / float(sr)
    t = np.linspace(0, T, len(onset_env))
    # Plot the onset envelope:

    plt.figure(figsize=(14, 5))
    plt.plot(t, onset_env)
    plt.xlabel('Time (sec)')
    plt.xlim(xmin=0)
    plt.ylim(0)
    plt.show()


# # calculate STFT
#     D = librosa.stft(y)
#     # calculate power
#     power = np.abs(D) ** 2
#     power = librosa.power_to_db(np.abs(D)**2)
#     print(power, len(power))
#     # mean
#     average_power = np.mean(power)
#     print(average_power)
# Load the audio file

# Compute the Mel spectrogram
# S = librosa.feature.melspectrogram(y=y, sr=sr,
#                                    # n_mels=128
#                                    )
# S = np.abs(librosa.stft(y))
# print(len(S), len(S[0]))
# # Convert to decibel scale
# S_db = librosa.power_to_db(S, ref=np.max)
#
# # Calculate the average value of the spectrogram over time
# average_spectrogram = np.mean(S_db, axis=0)
#
# # Calculate the time axis
# time_axis = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr)
#
# # Plot the average value of the spectrogram over time
# plt.figure(figsize=(10, 4))
# plt.plot(time_axis, average_spectrogram)
# plt.xlabel('Time (s)')
# plt.ylabel('Average dB')
# plt.title('Average Spectrogram Value Over Time')
# plt.show()


def get_name(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    print(filename, ext)


if __name__ == "__main__":
    run_func_on_all_datasets('datasets/Processed', create_feature, None)
    # summarize_audio_files('datasets/Processed')
    # plot_spectral_contrast('datasets/Processed/Castanets/castanet2.ff.stereo.wav')
    # create_feature_file('datasets/Processed/Castanets/castanet2.ff.stereo.wav')
    # func('datasets/Hand/Castanets/castanet2.ff.stereo.aif')

    feat1 = create_feature('datasets/Processed/Clap/2040.wav')
    feat2 = create_feature('datasets/Processed/Mostly Drum/422461.wav')

    # feat1 = feat1.astype(np.float64)

    # print(calculate_similarity_between_feats(feat1, feat2))
    # sims = run_func_on_all_datasets('datasets/Processed', calculate_similarity_between_feats,
    #                                 'datasets/Processed/Clap/2040.wav')
    # print(sims)

    # print(compare_spectrogram(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
    #                           read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    # print(compare_spectrum(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
    #                           read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    # plot_spectrogram('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')
    # plot_spectrogram('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')
    # plot_2_spectrogram('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav',
    #                    'datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')
    # plot_spectrum('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav', _type='line')
    # plot_spectrum('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav', _type='line')
    # plot_spectrum('datasets/Hand/Mostly Drum/422458.wav', _type='line', max_freq=2000)

    # plot_spectral_centroid('datasets/Hand/Claves/clave1.ff.stereo.aif',
    #                        calculate_spectral_centroid('datasets/Hand/Claves/clave1.ff.stereo.aif'))
    # plot_spectral_centroid('datasets/Hand/Castanets/castanet2.ff.stereo.aif',
    #                        calculate_spectral_centroid('datasets/Hand/Castanets/castanet2.ff.stereo.aif'))
    # print(calculate_average_power(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_average_frequency(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_average_frequency(*read_audio_from_path('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')))

    # plot_attack_time('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # print(calculate_mfcc(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))

    # plot_all_chromatogram('datasets/Hand/Castanets')
    # print(calculate_spectral_bandwidth(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_spectral_contrast(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_spectral_flatness(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_spectral_rolloff(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # plot_spectral_bandwidth('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # plot_spectral_contrast('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # plot_spectral_flatness('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # plot_spectral_rolloff('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # print(calculate_average_pitch(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # plot_chroma('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # chroma1 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Granite Blocks/graniteblock5.pp.aif'))
    # print(len(chroma1[0]))
    # print(chroma1)
    # chroma2 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Castanets/castanet1.ff.aif'))
    # print(len(chroma2[0]))
    # print(chroma2)
    # print(calculate_harmonic(*read_audio_from_path('datasets/Hand/Castanets/castanet1.ff.aif')))
    # plot_harmornicity('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')
    # print(calculate_max_frequency(*read_audio_from_path('datasets/Hand/Mostly Drum/422458.wav')))
