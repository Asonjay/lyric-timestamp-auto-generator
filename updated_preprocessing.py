# data_preprocessing.py
# Author: Jason Xu

import glob
import os
import re

import librosa
import torch
import numpy as np

SEC_PER_MIN = 60

def generate_lyrics_label():
    for filename in glob.glob(os.path.join(os.getcwd() + '\Model_Data\lyrics', '*.txt')):
        with open(filename, 'r') as r:
            lines = r.readlines()
            label_list = [0] * 2 * SEC_PER_MIN
            for line in lines:
                timestamp = re.split(r'\[|\]', line)[1]
                time = re.split(r'[.:]', timestamp)
                # time format: [min, sec, minisec]
                if int(time[0]) < 2:
                    sec = int(time[1]) + int(time[0]) * SEC_PER_MIN
                    label_list[sec] = 1
            with open(filename.split('.')[0] + ".label", 'w') as w:
                w.write(' '.join(str(s) for s in label_list))
                w.close()
        r.close()


def zip_label_wav(set_name):
    reg_dir = os.getcwd() + ('/Model_Data/' + set_name + '/songs/vocal_reg/')
    iso_dir = os.getcwd() + ('/Model_Data/' + set_name + '/songs/vocal_iso/')
    label_dir = os.getcwd() + ('/Model_Data/' + set_name + '/lyrics/')
    reg_wav_list = []
    iso_wav_list = []
    label_list = []
    for filename in os.scandir(reg_dir):
        print("Loading file:", filename.name)
        reg_wav, _ = librosa.load(filename.path, duration=120)
        iso_wav, _ = librosa.load(iso_dir + filename.name, duration=120)

        #breaks down mfccs into their time intervals
        audio_length = len(reg_wav) / 22050 # in seconds
        step = 512 / 22050 # in seconds
        intervals_s = np.arange(0, audio_length, step)

        with open(label_dir + filename.name.split('.')[0] + '.txt', 'r') as r:
            stop_times = []
            file = r.readlines()
            for line in file:
                minute = float(line[line.index('[')+1:line.index(':')]) * 60
                second = float(line[line.index(':')+1:line.index('.')])
                sub_second = float(line[line.index('.')+1:line.index(']')]) / 100
                time_s = minute+second+sub_second
                stop_times.append(time_s)
            r.close()

            song_label_torch = torch.zeros(intervals_s.shape)

            for i in range(len(intervals_s)-1):
                for stop in stop_times:
                    if stop > intervals_s[i] and stop <= intervals_s[i+1]:
                        song_label_torch[i] = 1.0
                        song_label_torch[i+1] = 1.0

        reg_wav_list.append(reg_wav)
        iso_wav_list.append(iso_wav)
        label_list.append(song_label_torch)
    return label_list, reg_wav_list, iso_wav_list

if __name__ == '__main__':
    # Pack the label and .wav file
    iso_labels, iso_wavs, reg_wavs = zip_label_wav()
