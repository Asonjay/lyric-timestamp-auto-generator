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
    self.fc3 = nn.Linear(hid, out)
    torch.nn.init.xavier_uniform_(self.fc3.weight)


  # pass forward for nn
  def forward(self, data):
    x=self.relu(self.fc1(data))
    x=self.relu(self.fc2(x))
    x=self.sigmoid(self.fc3(x))

    return x


def model_accuracy(net, eval_wavs, eval_labels):
    print('===== Evaluating =====')
    #set parameters
    sr = 22050
    hop_length = 512

    pred_seq = []
    for song in eval_wavs:
        mfccs = librosa.feature.mfcc(y=song, n_mfcc=num_mfccs) #(num_mfccs, 5168)
        mfccs = torch.FloatTensor(mfccs.T)

        pred = pause_net.forward(mfccs)
        pred = torch.squeeze(pred, -1)

        pred = torch.where(pred >= 0.22, 1, 0)
        pred_seq.append(pred.tolist())
    labels = np.array(pred_seq)

    # Eval
    for predictions, golds in zip(labels, eval_labels):
        num_correct = 0
        num_pos_correct = 0
        num_pred = 0
        num_gold = 0
        num_total = 0
        if len(golds) != len(predictions):
            raise Exception("Mismatched gold/pred lengths: %i / %i" % (len(golds), len(predictions)))
        for idx in range(0, len(golds)):
            gold = golds[idx]
            prediction = predictions[idx]
            if prediction == gold:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if gold == 1:
                num_gold += 1
            if prediction == 1 and gold == 1:
                num_pos_correct += 1
            num_total += 1
        acc = float(num_correct) / num_total
        output_str = "Accuracy: %i / %i = %f" % (num_correct, num_total, acc)
        prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
        rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec > 0 and rec > 0 else 0.0
        output_str += ";\nPrecision (fraction of predicted positives that are correct): %i / %i = %f" % (num_pos_correct, num_pred, prec)
        output_str += ";\nRecall (fraction of true positives predicted correctly): %i / %i = %f" % (num_pos_correct, num_gold, rec)
        output_str += ";\nF1 (harmonic mean of precision and recall): %f;\n" % f1
        print(output_str)
        print("++++++++++++++++++++++++++++++++++++")


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
    iso_labels, reg_wavs, iso_wavs = zip_label_wav("train")
    song_idxs = list(range(len(iso_labels)))
    test_iso_labels, test_reg_wavs, test_iso_wavs = zip_label_wav("dev")

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

            """#breaks down mfccs into their time intervals
            audio_length = len(song) / sr # in seconds
            step = hop_length / sr # in seconds
            intervals_s = np.arange(0, audio_length, step)"""

            optimizer.zero_grad()
            pred = pause_net.forward(mfccs)
            pred = torch.squeeze(pred, -1)

            # nonpause proper label is 0, pause proper label is 1
            loss = loss_func(pred, label)
            total_loss += loss

            loss.backward()
            optimizer.step()

        print("Epoch %i Total Loss: %.3f" % (epoch, total_loss))

    model_accuracy(pause_net, iso_wavs, iso_labels)
    model_accuracy(pause_net, test_iso_wavs, test_iso_labels)
