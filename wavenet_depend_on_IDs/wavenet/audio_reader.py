import fnmatch
import os
import re
import threading
import random

import json
import librosa
import numpy as np
import tensorflow as tf

ID_List = './wavenet/ID_List.json'

def find_files(directory, pattern='*.wav'):#含有wav的文件
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    random.shuffle(files)
    return files


def load_generic_audio(directory, sample_rate):#wavenet
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)#生成的一维numpy数组
        audio = audio.reshape(-1, 1)#audio转成这种二位矩阵然后的处理呢？
        yield audio, filename #带有yield的函数在Python中被称之为生成器


def load_vctk_audio(directory, sample_rate):#modified wavenet
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
        speaker_id = IDs[str(speaker_id)]  #change speaker_id to [0,109)225编号改成从0开始编号
        yield audio, filename, speaker_id


def load_CHN_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the CHN dataset, and
    additionally the ID of the corresponding speaker.'''
    files = find_files(directory)
    for filename in files:
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)#audio.size即向量大小?
        audio = audio.reshape(-1, 1)
        ind = filename.find('m001')
        if ind > 0:
            speaker_id = 0   # 0 for man, 1 for woman
        else:
            speaker_id = 1
        yield audio, filename, speaker_id


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < 2048:
        return audio[0:0]
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=2048):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.silence_threshold = silence_threshold
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,#一个FIFOQueue ，同时根据padding支持batching变长的tensor
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.IDs_placeholder = tf.placeholder(dtype=tf.int32, shape=())
        self.IDs_queue = tf.PaddingFIFOQueue(queue_size,
                                         ['int32'],
                                         shapes=[()])
        self.IDs_enqueue = self.IDs_queue.enqueue([self.IDs_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        if not find_files(audio_dir):
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        IDs = self.IDs_queue.dequeue_many(num_elements)
        return output, IDs

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_vctk_audio(self.audio_dir, self.sample_rate) #load_generic_audio(self.audio_dir, self.sample_rate)

            for audio, filename, speaker_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    #audio = trim_silence(audio[:, 0], self.silence_threshold)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                if self.sample_size:
                    # Cut samples into fixed size pieces
                    if audio.size < self.sample_size:
                        print("Warning: {0} was ignored as it contains only {1}."
                              .format(filename, audio.size))
                        continue
                    sample_n_float = 1.0 * audio.size / self.sample_size 
                    sample_n_int = int(sample_n_float)
                    sample_n = sample_n_int + int( sample_n_float - sample_n_int > random.random() )
                    for i in range(sample_n):#每一次采样中
                        start_ = random.randint(0, audio.size - self.sample_size)
                        piece = np.reshape(audio[start_ : start_+self.sample_size], [-1, 1])
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        sess.run(self.IDs_enqueue,
                                 feed_dict={self.IDs_placeholder: speaker_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    sess.run(self.IDs_enqueue,
                             feed_dict={self.IDs_placeholder: speaker_id})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
