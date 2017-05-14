import os
import random
import librosa
import re
import fnmatch
import json
import numpy as np

ID_List = './wavenet/ID_List.json'
#for root, dirs, files in os.walk("./VCTK-Corpus"):
#    for name in files:
#        print name
#	print(os.path.join(root, name))
#    for name in dirs:
#        print(os.path.join(root, name))
def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    random.shuffle(files)
    return files

def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < 2048:
        return audio[0:0]
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

def load_vctk_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the VCTK dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    speaker_re = re.compile(r'p([0-9]+)_([0-9]+)\.wav')
    with open(ID_List, 'r') as f:
        IDs = json.load(f)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        matches = speaker_re.findall(filename)[0]
        speaker_id, recording_id = [int(id_) for id_ in matches]
        speaker_id = IDs[str(speaker_id)]  #change speaker_id to [0,109)
        yield audio, filename, speaker_id

n = 0
result = [] 
iterator = load_vctk_audio('./VCTK-Corpus',16000)
for audio, filename, speaker_id in iterator:
    print audio
    print filename
    print speaker_id
    print audio[0:0]
        # Remove silence
    audio = trim_silence(audio[:, 0], 0.25)
    if audio.size == 0:
        print("Warning: {} was ignored as it contains only silence. Consider decreasing trim_silence threshold, or adjust volume of the audio.".format(filename))
    result.append(audio)
    #print len(result)
    print result[n]
    n = n+1
    if n==10:
        #print result
        break
