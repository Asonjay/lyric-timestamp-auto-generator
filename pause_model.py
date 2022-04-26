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
class PauseNet1(nn.Module):
  def __init__(self, num_mfccs, hid, out):
    super().__init__()
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(num_mfccs, hid)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.fc2 = nn.Linear(hid, hid)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    self.fc3 = nn.Linear(hid, hid)
    torch.nn.init.xavier_uniform_(self.fc3.weight)
    self.fc4 = nn.Linear(hid, hid)
    torch.nn.init.xavier_uniform_(self.fc4.weight)
    self.fc5 = nn.Linear(hid, out)
    torch.nn.init.xavier_uniform_(self.fc5.weight)


  # pass forward for nn
  def forward(self, data):
    x=self.relu(self.fc1(data))
    x=self.relu(self.fc2(x))
    x=self.relu(self.fc3(x))
    x=self.relu(self.fc4(x))
    x=self.sigmoid(self.fc5(x))

    return x


if __name__ == '__main__':
    #set parameters
    num_epochs = 5
    num_mfccs = 13
    hidden_size = 30
    out_size = 1

    # set other constants
    sr = 22050
    hop_length = 512
    song_dur = 120

    # create neural network
    pause_net = PauseNet1(num_mfccs, hidden_size, out_size)
    optimizer = optim.Adam(pause_net.parameters(), lr=0.001)
    loss_func = nn.BCELoss()


    # Pack the label and .wav file
    iso_labels, iso_wavs, reg_wavs = zip_label_wav()
    song_idxs = list(range(len(iso_labels)))

    for epoch in range(num_epochs):
        #shuffle songs for training each epoch to prevent overfitting
        random.shuffle(song_idxs)

        total_loss = 0
        for i in song_idxs:
            song = iso_wavs[i]
            label = iso_labels[i]

            # sample window of 2048 and hop size of 512 samples
            mfccs = librosa.feature.mfcc(y=song, n_mfcc=num_mfccs) #(num_mfccs, 5168)
            mfccs = torch.FloatTensor(mfccs.T)

            #breaks down mfccs into their time intervals
            audio_length = len(song) / sr # in seconds
            step = hop_length / sr # in seconds
            intervals_s = np.arange(0, audio_length, step)


            optimizer.zero_grad()
            pred = pause_net.forward(mfccs)
            pred = torch.squeeze(pred, -1)

            # nonpause proper label is 0, pause proper label is 1
            loss = loss_func(pred, label)
            total_loss += loss

            loss.backward()
            optimizer.step()

        print("Epoch %i Total Loss: %.3f" % (epoch, total_loss))
