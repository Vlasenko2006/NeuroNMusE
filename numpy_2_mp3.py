#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 10:25:45 2025

@author: andreyvlasenko
"""

import os
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np



# Ensure ffmpeg and ffprobe are configured
os.environ["PATH"] += os.pathsep + "/usr/local/bin"
AudioSegment.converter = "/usr/local/bin/ffmpeg"
AudioSegment.ffprobe = "/usr/local/bin/ffprobe"

def mp3_to_numpy(mp3_file, target_sr=12000):
    """
    Converts an MP3 file to a NumPy array.
    Parameters:
    - mp3_file: Path to the input MP3 file.
    - target_sr: Target sample rate for the waveform (default = 16 kHz).
    Returns:
    - waveform: NumPy array of audio data.
    - sample_rate: Sample rate of the audio.
    """
    # Load the MP3 file as a waveform
    waveform, sample_rate = librosa.load(mp3_file, sr=target_sr, mono=False)
    print(f"Converted MP3 to NumPy array: {waveform.shape}, Sample Rate: {sample_rate}")
    return waveform, sample_rate

def numpy_to_mp3(waveform, sample_rate, output_mp3_file="output.mp3"):
    """
    Converts a NumPy array back to an MP3 file.
    Parameters:
    - waveform: NumPy array of audio data.
    - sample_rate: Sample rate of the waveform.
    - output_mp3_file: Path to the output MP3 file.
    """
    # Save the NumPy array as a .wav file
    temp_wav_file = "temp_output.wav"
    sf.write(temp_wav_file, waveform.T, sample_rate)  # Transpose if stereo

    # Convert the .wav file to .mp3 using pydub
    audio = AudioSegment.from_wav(temp_wav_file)
    audio.export(output_mp3_file, format="mp3")
    print(f"Converted NumPy array to MP3: {output_mp3_file}")

    # Remove the temporary .wav file
    os.remove(temp_wav_file)


# Paths
sample_rate = 12000
path = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_output/"
path_output = "/Users/andreyvlasenko/tst/Music_NN/Lets_Rock/NN_music_output"

# Ensure the output directory exists
os.makedirs(path_output, exist_ok=True)

# Process all .npy files
for file in os.listdir(path):
    if file.endswith(".npy"):
        file_path = os.path.join(path, file)
        print(f"Processing file: {file_path}")
        
        # Load the NumPy array
        array = np.load(file_path)
        
        # Subtract 1 and convert to float32
        array = array - 1.0
        array = array.astype(np.float32)
        
        # Generate output MP3 file path
        output_mp3_file = os.path.join(path_output, file.replace(".npy", ".mp3"))
        
        # Convert to MP3
        numpy_to_mp3(array, sample_rate, output_mp3_file)

print("Processing complete. MP3 files saved to:", path_output)