from utils import run_func_on_all_datasets
from pydub import AudioSegment, utils
import os


def get_prober_name():
    return "E://mabe//CodeStuff//Personal//CSDLDPT//ffmpeg//bin//ffprobe.exe"

AudioSegment.converter = "E://mabe//CodeStuff//Personal//CSDLDPT//ffmpeg//bin//ffmpeg.exe"
utils.get_prober_name = get_prober_name


def pad_audio(audio_path, output_path='datasets/Processed/', target_length=1000, target_sr=44100):
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
    index = audio_path.rfind('\\')
    index = audio_path.rfind('\\', 0, index)
    file_context = audio_path[index + 1:]
    base, _ = os.path.splitext(file_context)
    file_context = base + '.wav'
    output_path = os.path.normpath(output_path + file_context)
    # create dir if not exists
    dir = os.path.dirname(output_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if len(audio) >= 1000: return
    silence = AudioSegment.silent(duration=target_length-len(audio))
    padded = audio + silence
    padded.export(output_path, format='wav')


if __name__ == '__main__':
    run_func_on_all_datasets('datasets/Beijing', pad_audio)