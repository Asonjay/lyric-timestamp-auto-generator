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


def zip_label_wav():
    reg_dir = os.getcwd() + '/Model_Data/songs/vocal_reg/'
    iso_dir = os.getcwd() + '/Model_Data/songs/vocal_iso/'
    label_dir = os.getcwd() + '/Model_Data/lyrics/'
    reg_wav_list = []
    iso_wav_list = []
    label_list = []
    for filename in os.scandir(reg_dir):
        print("Loading file:", filename.name)
        reg_wav, _ = librosa.load(filename.path, duration=120)
        iso_wav, _ = librosa.load(iso_dir + filename.name, duration=120)
        with open(label_dir + filename.name.split('.')[0] + '.label', 'r') as r:
            line = r.readline()
            label_list.append(str(line).split(' '))
            r.close()
        reg_wav_list.append(reg_wav)
        iso_wav_list.append(iso_wav)
    label_torch = torch.from_numpy(np.float_(label_list)).float()
    return label_torch, reg_wav_list, iso_wav_list

if __name__ == '__main__':
    # Generate lyric file classifier
    generate_lyrics_label()
    # Pack the label and .wav file
    iso_labels, iso_wavs, reg_wavs = zip_label_wav()
