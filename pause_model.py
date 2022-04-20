# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import numpy as np

from data_preprocessing import *

# Build the neural network for classifying pauses against nonpauses
class PauseNet1(nn.Module):
  def __init__(self, num_mfccs, hid1, hid2, out):
    super().__init__()
    self.fc1 = nn.Linear(num_mfccs, hid1)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.fc2 = nn.Linear(hid1, hid2)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    self.fc3 = nn.Linear(hid2, out)
    torch.nn.init.xavier_uniform_(self.fc3.weight)


  # pass forward for nn
  def forward(self, x):
    x=self.fc1(x)
    x=F.sigmoid(x)
    x=self.fc2(x)
    x=F.sigmoid(x)
    x=self.fc3(x)

    return x


if __name__ == '__main__':
    #set parameters
    num_epochs = 3
    num_mfccs = 13
    hidden1_size = 100
    hidden2_size = 25
    out_size = 2

    # set other constants
    sr = 22050
    hop_length = 512
    song_dur = 120

    # create neural network
    pause_net = PauseNet1(num_mfccs, hidden1_size, hidden2_size, out_size)
    optimizer = optim.Adam(pause_net.parameters(), lr=0.001)
    loss_func = nn.BCELoss()

    # get data
    generate_lyrics_label()
    # Pack the label and .wav file
    iso_labels, iso_wavs, reg_wavs = zip_label_wav()

    for epoch in range(num_epochs):

        total_loss = 0
        for song, labels in zip(reg_wavs, iso_labels):

            # sample window of 2048 and hop size of 512 samples
            mfccs = librosa.feature.mfcc(y=song, n_mfcc=num_mfccs, center=False) #(num_mfccs, 5161)

            audio_length = song_dur / sr # in seconds
            step = hop_length / sr # in seconds
            intervals_s = np.arange(0, audio_length, step)
            print(intervals_s)

            for sec, label in zip(range(0, mfccs.shape[1], mfcc_sec), labels):
                # indexes a single second of song from MFCCs
                song_s = mfccs[:, sec:sec+mfcc_sec]
                #average the values over a second
                inp = np.mean(mfccs, axis=1)

                optimizer.zero_grad()
                pred = pause_net.forward(inp)

                # nonpause proper label is 0, pause proper label is 1
                loss = loss_func(pred, label)
                total_loss += loss

                loss.backward()
                optimizer.step()
