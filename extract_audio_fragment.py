import librosa
import soundfile as sf
import numpy as np
import os

audio_dir = 'test_audio'
output_dir = 'segments_1_25_06065'

def extract_peak_segment(audio_path, output_path, pre_peak=0.6, post_peak=0.65):
    audio, sr = librosa.load(audio_path, sr=None)
    audio_length = len(audio)
    segment_length = int((pre_peak + post_peak) * sr)
    peak_index = np.argmax(np.abs(audio))

    start_index = min(int(peak_index - pre_peak * sr), audio_length - segment_length)
    end_index = max(int(peak_index + post_peak * sr), segment_length)

    start_index = max(start_index, 0)
    end_index = min(end_index, audio_length)

    segment = audio[start_index:end_index]
    sf.write(output_path, segment, sr)

for file in os.listdir(audio_dir):
    print('Processing file: ', file)
    audio_path = os.path.join(audio_dir, file)
    output_path = os.path.join(output_dir, file)
    if audio_path[-4:] == '.wav':
        extract_peak_segment(audio_path, output_path)
    print('Exported to: ', output_path)
