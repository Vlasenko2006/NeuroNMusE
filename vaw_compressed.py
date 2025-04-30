import os
os.environ["PATH"] += os.pathsep + "/usr/local/bin"

from pydub import AudioSegment

# Ensure ffmpeg is properly configured
AudioSegment.converter = "/usr/local/bin/ffmpeg"  # Change this to the path of ffmpeg on your system
AudioSegment.ffprobe = "/usr/local/bin/ffprobe"

def mp3_to_wav(mp3_file, output_wav_file="output.wav", target_sample_rate=16000, target_channels=1):
    """
    Converts an MP3 file to a compressed WAV file.
    Parameters:
    - mp3_file: Path to the input MP3 file.
    - output_wav_file: Path to the output WAV file.
    - target_sample_rate: Desired sample rate (default = 16000 Hz).
    - target_channels: Number of audio channels (default = 1 for mono).
    """
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file)
    
    # Set target properties for compression
    audio = audio.set_frame_rate(target_sample_rate)  # Resample
    audio = audio.set_channels(target_channels)      # Convert to mono if necessary
    
    # Export as WAV
    audio.export(output_wav_file, format="wav")
    print(f"Converted and compressed MP3 to WAV: {output_wav_file}")

def wav_to_mp3(wav_file, output_mp3_file="output.mp3"):
    """
    Converts a WAV file back to an MP3 file.
    """
    # Load the WAV file
    audio = AudioSegment.from_wav(wav_file)
    
    # Export as MP3
    audio.export(output_mp3_file, format="mp3")
    print(f"Converted WAV to MP3: {output_mp3_file}")

# Example usage
if __name__ == "__main__":
    path = "/Volumes/Music_Video_Foto/Musik/LATINOS"
    
    # Input MP3 file
    example_mp3 = os.path.join(path, "Desperado.mp3")
    
    # Step 1: Convert MP3 to a compressed WAV
    mp3_to_wav(example_mp3, output_wav_file="Desperado_compressed.wav", target_sample_rate=16000, target_channels=1)
    
    # Step 2: Convert WAV back to MP3
    wav_to_mp3("Desperado_compressed.wav", output_mp3_file="Desperado_converted.mp3")