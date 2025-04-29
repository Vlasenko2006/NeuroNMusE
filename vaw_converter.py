#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:31:05 2025

@author: andrey
"""


import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from pydub import AudioSegment
import soundfile as sf



def mp3_to_mel_spectrogram(mp3_file, sr=2 * 16000, n_mels= 2 * 128):
    # Load the .mp3 file
    y, sr = librosa.load(mp3_file, sr=sr)
    
    # Compute the Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Optional: Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel-Spectrogram")
    plt.show()
    
    # Return the Mel-spectrogram as a numpy array
    return mel_spectrogram_db

# Example usage
mel_spectrogram = mp3_to_mel_spectrogram("example.mp3")
np.save("example_spectrogram.npy", mel_spectrogram)  # Save for NN input

def mel_spectrogram_to_mp3(mel_spectrogram, sr=2*16000, output_mp3_file="output.mp3"):
    """
    Converts a Mel-spectrogram back to an MP3 file with improved quality using Griffin-Lim.
    
    Parameters:
    - mel_spectrogram: The Mel-spectrogram as a numpy array.
    - sr: The sampling rate of the audio (default = 16000 Hz).
    - output_mp3_file: The name of the output MP3 file.
    """
    # Convert the Mel-spectrogram back to a linear-frequency spectrogram
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=sr)

    # Use Griffin-Lim algorithm to estimate the phase and reconstruct the waveform
    waveform = librosa.griffinlim(linear_spectrogram)

    # Save the reconstructed waveform as a .wav file
    sf.write("output.wav", waveform, sr)

    # Convert the .wav file to MP3 using pydub
    audio = AudioSegment.from_wav("output.wav")
    audio.export(output_mp3_file, format="mp3")

    print(f"Reconstructed MP3 saved as: {output_mp3_file}")
# Example usage
mel_spectrogram = np.load("example_spectrogram.npy")  # Load NN output
mel_spectrogram_to_mp3(mel_spectrogram, output_mp3_file="example_output.mp3")