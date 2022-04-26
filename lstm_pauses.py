# import standard PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import numpy as np
import random

from updated_preprocessing import *


# Build the neural network for classifying pauses against nonpauses
class PauseNet2(nn.Module):

    def __init__(self, num_mfccs, hid1, hid2, out):
        super().__init__()
        self.lstm = nn.LSTM(num_mfccs, hid1, batch_first=True)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        self.fc1 = nn.Linear(hid1, hid2)
        self.fc2 = nn.Linear(hid2, out)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    # define forward function
    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.fc1(out)
        x = self.relu(x)
        x = self.sigmoid(self.fc2(x))

        return x



if __name__ == '__main__':

    #set parameters
    num_epochs = 5
    num_mfccs = 13
    hidden1_size = 100
    hidden2_size = 20
    out_size = 1

    # set other constants
    sr = 22050
    hop_length = 512
    song_dur = 120

    # create neural network
    pause_net = PauseNet2(num_mfccs, hidden1_size, hidden2_size, out_size)
    optimizer = optim.Adam(pause_net.parameters(), lr=0.001)
    loss_func = nn.BCELoss()

    # Pack the label and .wav file
    iso_labels, reg_wavs, iso_wavs = zip_label_wav()
    song_idxs = list(range(len(iso_labels)))

    for epoch in range(num_epochs):
        #shuffle songs for training each epoch to prevent overfitting
        random.shuffle(song_idxs)

        total_loss = 0
        for song, labels in zip(iso_wavs, iso_labels):

            # sample window of 2048 and hop size of 512 samples
            mfccs = librosa.feature.mfcc(y=song, n_mfcc=num_mfccs) #(num_mfccs, 5168)
            mfccs = torch.unsqueeze(torch.FloatTensor(mfccs.T), 0)

            #breaks down mfccs into their time intervals
            audio_length = len(song) / sr # in seconds
            step = hop_length / sr # in seconds
            intervals_s = np.arange(0, audio_length, step)

            optimizer.zero_grad()
            pred = pause_net.forward(mfccs)

            pred = torch.squeeze(torch.squeeze(pred, 0), -1)

            # nonpause proper label is 0, pause proper label is 1
            loss = loss_func(pred, labels)
            total_loss += loss

            loss.backward()
            optimizer.step()

        print("Epoch %i Total Loss: %.3f" % (epoch, total_loss))
