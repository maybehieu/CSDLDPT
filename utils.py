import librosa


def read_audio_from_path(_path=r''):
    try:
        y, sr = librosa.load(_path)
        return y, sr
    except:
        print(f'Error loading path {_path}')
    return None