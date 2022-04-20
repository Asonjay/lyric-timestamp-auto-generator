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
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(num_mfccs, hid1)
    torch.nn.init.xavier_uniform_(self.fc1.weight)
    self.fc2 = nn.Linear(hid1, hid2)
    torch.nn.init.xavier_uniform_(self.fc2.weight)
    self.fc3 = nn.Linear(hid2, out)
    torch.nn.init.xavier_uniform_(self.fc3.weight)


  # pass forward for nn
  def forward(self, x):
    x=self.relu(self.fc1(x))
    x=self.relu(self.fc2(x))
    x=self.sigmoid(self.fc3(x))

    return x


if __name__ == '__main__':
    #set parameters
    num_epochs = 30
    num_mfccs = 20
    hidden1_size = 100
    hidden2_size = 25
    out_size = 1

    # set other constants
    sr = 22050
    hop_length = 512
    song_dur = 120

    # create neural network
    pause_net = PauseNet1(num_mfccs, hidden1_size, hidden2_size, out_size)
    optimizer = optim.Adam(pause_net.parameters(), lr=0.0005)
    loss_func = nn.BCELoss()

    # get data
    generate_lyrics_label()
    # Pack the label and .wav file
    iso_labels, iso_wavs, reg_wavs = zip_label_wav()

    for epoch in range(num_epochs):

        total_loss = 0
        for song, labels in zip(reg_wavs, iso_labels):

            # sample window of 2048 and hop size of 512 samples
            mfccs = librosa.feature.mfcc(y=song, n_mfcc=num_mfccs) #(num_mfccs, 5168)

            #breaks down mfccs into their time intervals
            audio_length = len(song) / sr # in seconds
            step = hop_length / sr # in seconds
            intervals_s = np.arange(0, audio_length, step)

            # get each second and sample
            for label in labels:
                #allows me to easily find the cutoff
                intervals_s -= 1
                #isolate a single second
                sec_interval = np.where(intervals_s < 0)[0]
                # indexes a single second of song from MFCCs
                song_sec = np.take(mfccs, sec_interval, axis=1)

                #average the values over a second
                inp = torch.from_numpy(np.mean(song_sec, axis=1))

                label_tensor = torch.tensor([label])

                optimizer.zero_grad()
                pred = pause_net.forward(inp)

                # nonpause proper label is 0, pause proper label is 1
                loss = loss_func(pred, label_tensor)
                total_loss += loss

                loss.backward()
                optimizer.step()

        print("Epoch %i Total Loss: %.3f" % (epoch, total_loss))
