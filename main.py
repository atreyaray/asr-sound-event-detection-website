import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from audio import AudioUtil
from model import AudioClassifier
import torchaudio

# streamlit run main.py


def load_model(model_path, map_location):
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    return model


def sounds(prediction) :
    sound_label = ["Dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insects (flying)", "Sheep", "Crow"
                    ,"Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm"
                    ,"Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking, sipping"
                    , "Door knock", "Mouse click", "Keyboard typing", "Door, wood creaks", "Can opening", "washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking"
                    , "Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"]
    s = dict(zip(range(50), sound_label))
    return s[prediction]

def endtoend(model, audiofile):
    audio = AudioUtil.open(audiofile)
    rechannel = AudioUtil.rechannel(audio, 1) #change the channel
    resamp = AudioUtil.resample(rechannel, 44100)

    padded = AudioUtil.resize_aud(resamp, 5000)
    shifted = AudioUtil.time_shift(padded, 0.4)
    sgram = AudioUtil.spectro_gram(shifted, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
    input_loader = torch.utils.data.DataLoader(aug_sgram, batch_size=16, shuffle=False)
    
    for input in input_loader:
        #print(input.shape)
        input = input.reshape([-1,1,64,430]) #change the channel
        #print(input.shape)
        input_m, input_s = input.mean(), input.std()
        input = (input - input_m) / input_s
        #print(input.shape)
        output = model(input)
        _, prediction = torch.max(output,1)
        prediction = prediction.numpy()[0]
    return prediction


'''
# Sound Audio Detection

'''


audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # save audio file to disk
    with open("audio.wav", "wb") as f:
        f.write(audio_bytes)


    # convert bytes to numpy array
    audio = np.frombuffer(audio_bytes, dtype=np.int16)

    # plot audio waveform
    # use librosa to plot spectrogram 
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(audio)
    ax[0].set(title="Audio Waveform")
    ax[0].label_outer()
    ax[1].specgram(audio, Fs=44100)
    ax[1].set(title="Audio Spectrogram")
    ax[1].label_outer()

    # call end to end
    # get the absolute path of the current directory 
    dir = os.path.dirname(__file__)
    model = load_model(dir + '/cnn-200.pt', map_location=torch.device('cpu'))
    s = endtoend(model, "audio.wav")
    st.write(f"I think this is the sound of a {sounds(s)}")

    st.pyplot(fig)


    