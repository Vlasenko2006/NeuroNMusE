import librosa
import pretty_midi
from pydub import AudioSegment

def mp3_to_midi(mp3_file, midi_file="output.mid"):
    """
    Converts an MP3 file to a MIDI file.
    Note: This is a simplified implementation and works best with clear melodies.
    """
    # Step 1: Load the MP3 file and extract the audio signal
    y, sr = librosa.load(mp3_file)

    # Step 2: Perform pitch tracking to extract notes
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Step 3: Create a PrettyMIDI object and add notes
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Iterate over pitches and add them as MIDI notes
    for time_idx, pitch in enumerate(pitches):
        for frequency_idx, magnitude in enumerate(pitch):
            if magnitude > 0:  # Only consider strong magnitudes
                note_number = librosa.hz_to_midi(frequency_idx)
                if 0 <= note_number <= 127:  # MIDI note range
                    note = pretty_midi.Note(velocity=100, pitch=int(note_number),
                                            start=time_idx / sr, end=(time_idx + 1) / sr)
                    instrument.notes.append(note)

    midi.instruments.append(instrument)

    # Save the MIDI file
    midi.write(midi_file)
    print(f"MIDI file saved as: {midi_file}")

def midi_to_mp3(midi_file, output_mp3_file="output.mp3", soundfont="soundfont.sf2"):
    """
    Converts a MIDI file to an MP3 file using a soundfont.
    Requires FluidSynth installed on the system.
    """
    from midi2audio import FluidSynth

    # Step 1: Convert MIDI to WAV using FluidSynth
    fs = FluidSynth(soundfont)
    wav_file = "output.wav"
    fs.midi_to_audio(midi_file, wav_file)

    # Step 2: Convert WAV to MP3 using pydub
    audio = AudioSegment.from_wav(wav_file)
    audio.export(output_mp3_file, format="mp3")

    print(f"MP3 file saved as: {output_mp3_file}")

# Example Usage
mp3_file = "example.mp3"
midi_file = "example.mid"
output_mp3_file = "example_output.mp3"

# Convert MP3 to MIDI
mp3_to_midi(mp3_file, midi_file=midi_file)

# Convert MIDI back to MP3
midi_to_mp3(midi_file, output_mp3_file=output_mp3_file, soundfont="soundfont.sf2")