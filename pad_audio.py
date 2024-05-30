from utils import run_func_on_all_datasets
from pydub import AudioSegment, utils
import os
import librosa


def get_prober_name():
    return "C://Program Files (x86)//ffmpeg//bin//ffprobe.exe"


AudioSegment.converter = "C://Program Files (x86)//ffmpeg//bin//ffmpeg.exe"
utils.get_prober_name = get_prober_name


def pad_audio(
    audio_path,
    output_path="datasets/Processed_v2/",
    target_length=3000,
    target_sr=44100,
):
    """
    Add padding to the audio in order to achieve fixed length
    Change sampling rate to fixed
    :param target_length:
    :param target_sr:
    :param audio_path:
    :param output_path:
    :return:
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_sr)
    index = audio_path.rfind("\\")
    index = audio_path.rfind("\\", 0, index)
    file_context = audio_path[index + 1 :]
    base, _ = os.path.splitext(file_context)
    file_context = base + ".wav"
    output_path = os.path.normpath(output_path + file_context)
    # create dir if not exists
    dir = os.path.dirname(output_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if target_length > len(audio):
        silence = AudioSegment.silent(duration=target_length - len(audio))
        padded = audio + silence
        padded.export(output_path, format="wav")
        return
    if len(audio) >= target_length:
        audio = audio[:target_length]
        audio.export(output_path, format="wav")
        return


def get_native_sr(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    print(audio_path, sr)


if __name__ == "__main__":
    run_func_on_all_datasets(r"datasets\Processed_v3", pad_audio, None)
    # run_func_on_all_datasets('datasets/Storage/Ver1', get_native_sr, None)
