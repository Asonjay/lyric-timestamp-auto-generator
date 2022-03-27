#vocal.py
#author: JDW 3/6/22

import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import librosa.display

"""
Code & Usage: 
- To get Librosa as a module, I used 'pip install librosa', after launching my terminal as administrator
- Implementation pulled from: https://librosa.org/librosa_gallery/auto_examples/plot_vocal_separation.html

"""

if __name__ == '__main__':

    # load then compute the spectrogram magnitude and phase
    y, sr = librosa.load('data/sample/countryRoads.wav', duration=120)
    S_full, phase = librosa.magphase(librosa.stft(y))

    S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)

    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                               margin_i * (S_full - S_filter),
                               power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                               margin_v * S_filter,
                               power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full

    """ This code will produce plots based on the seperated vocals:"""

    idx = slice(*librosa.time_to_frames([15, 20], sr=sr))
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='log', sr=sr)
    plt.title('Full spectrum')
    plt.colorbar()

    plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                            y_axis='log', sr=sr)
    plt.title('Background')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                            y_axis='log', x_axis='time', sr=sr)
    plt.title('Foreground')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    

    foreground_audio = librosa.istft(S_foreground)
    soundfile.write('data/results/foreground.WAV', foreground_audio, sr)

    
