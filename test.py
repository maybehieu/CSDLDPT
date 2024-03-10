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

from utils import read_audio_from_path
from calculation_func import compare_spectrogram, compare_spectrum
from presentation_func import plot_spectrogram, plot_2_spectrogram


if __name__ == "__main__":
    print(compare_spectrogram(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
                              read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    print(compare_spectrum(read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav'),
                              read_audio_from_path('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')))
    # plot_spectrogram('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav')
    # plot_spectrogram('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')
    # plot_2_spectrogram('datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.1.wav',
    #                    'datasets/MDLib2.2/Sorted/Kick/Long Kick/Press/DI_Long Kick_Press_1111.4.wav')