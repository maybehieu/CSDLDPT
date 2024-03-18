# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Load audio file
# audio_path = r'F:\Documents\CSDLDPT\datasets\MDLib2.2\Sorted\Cymbals\Dark Crash\Clamp\DI_Dark Crash_Clamp_1111.1.wav'
# y, sr = librosa.load(audio_path)
#
# # Display the waveform
# plt.figure(figsize=(12, 6))
# librosa.display.waveshow(y, sr=sr)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Amplitude')
# plt.title('Waveform')
# plt.show()
#
# # Calculate and display the spectrogram
# D = librosa.stft(y)
# D_db = librosa.amplitude_to_db(abs(D), ref=np.max)
#
# plt.figure(figsize=(12, 6))
# librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram')
# plt.show()
import sklearn.metrics.pairwise

from utils import *
from calculation_func import *
from presentation_func import *


if __name__ == "__main__":
    # print(compare_spectrogram(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
    #                           read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    # print(compare_spectrum(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
    #                           read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    # plot_spectrogram('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # plot_spectrogram('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')
    # plot_2_spectrogram('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav',
    #                    'datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')
    # plot_spectrum('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav', _type='line')
    # plot_spectrum('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav', _type='line')
    # plot_spectrum('datasets/Hand/Claves/clave1.ff.stereo.aif', _type='line', max_freq=15000)

    # plot_spectral_centroid('datasets/Hand/Claves/clave1.ff.stereo.aif',
    #                        calculate_spectral_centroid('datasets/Hand/Claves/clave1.ff.stereo.aif'))
    # plot_spectral_centroid('datasets/Hand/Castanets/castanet2.ff.stereo.aif',
    #                        calculate_spectral_centroid('datasets/Hand/Castanets/castanet2.ff.stereo.aif'))
    # print(calculate_average_power(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_average_frequency(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # print(calculate_average_frequency(*read_audio_from_path('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')))

    # plot_attack_time('datasets/Hand/Castanets/castanet2.ff.stereo.aif')
    # print(calculate_mfcc(*read_audio_from_path('datasets/Hand/Castanets/castanet2.ff.stereo.aif')))
    # summarize_audio_files('datasets/Hand')
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
    chroma1 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Granite Blocks/graniteblock5.pp.aif'))
    print(len(chroma1[0]))
    print(chroma1)
    chroma2 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Castanets/castanet1.ff.aif'))
    print(len(chroma2[0]))
    print(chroma2)