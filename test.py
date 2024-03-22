import librosa

from utils import *
from calculation_func import *
from presentation_func import *

def test_func(audio_path):
    # Compute the RMS value for each frame
    y, sr = librosa.load(audio_path, sr=None)
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


if __name__ == "__main__":
    run_func_on_all_datasets('datasets/Processed/Beijing', test_func)

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
    # chroma1 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Granite Blocks/graniteblock5.pp.aif'))
    # print(len(chroma1[0]))
    # print(chroma1)
    # chroma2 = calculate_chroma_stft(*read_audio_from_path('datasets/Hand/Castanets/castanet1.ff.aif'))
    # print(len(chroma2[0]))
    # print(chroma2)
    # print(calculate_harmonic(*read_audio_from_path('datasets/Hand/Castanets/castanet1.ff.aif')))
    # plot_harmornicity('datasets/Hand/Woodblocks/5.5wb.ff.stereo.aif')
    # print(calculate_max_frequency(*read_audio_from_path('datasets/Hand/Mostly Drum/422458.wav')))