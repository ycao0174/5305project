import soundfile as sf
import numpy as np

# Function to mix two audio files with a given volume ratio
def mix_audio(file1, file1_volume, file2, file2_volume, output_file):
    data1, samplerate1 = sf.read(file1)
    data2, samplerate2 = sf.read(file2)

    # Ensure both files have the same samplerate
    if samplerate1 != samplerate2:
        raise ValueError("Sample rates of the two files do not match!")

    # # Find the length of the longer file
    # max_length = max(len(data1), len(data2))
    #
    # # Pad the shorter file with zeros
    # if len(data1) < max_length:
    #     pad_length = max_length - len(data1)
    #     data1 = np.pad(data1, (0, pad_length), 'constant')
    # elif len(data2) < max_length:
    #     pad_length = max_length - len(data2)
    #     data2 = np.pad(data2, (0, pad_length), 'constant')
    #
    # # Normalize both tracks to have the same loudness level
    # data1_normalized = data1 / np.max(np.abs(data1))
    # data2_normalized = data2 / np.max(np.abs(data2))
    #
    # # Scale audio data by the specified volume levels
    # mixed_audio = (data1_normalized * file1_volume) + (data2_normalized * file2_volume)
    # Find the length of the shorter file
    min_length = min(len(data1), len(data2))

    # Crop the longer file to the length of the shorter file
    data1 = data1[:min_length]
    data2 = data2[:min_length]

    # Normalize both tracks to have the same loudness level
    data1_normalized = data1 / np.max(np.abs(data1))
    data2_normalized = data2 / np.max(np.abs(data2))

    # Scale audio data by the specified volume levels
    mixed_audio = (data1_normalized * file1_volume) + (data2_normalized * file2_volume)

    # Prevent clipping by normalizing the mixed audio
    mixed_audio /= np.max(np.abs(mixed_audio))

    # Save the mixed audio to a file
    sf.write(output_file, mixed_audio, samplerate1)

# Define the file names
file1 = 'sa1.wav'
file2 = 'si19.wav'

# Mix the files with the specified volume ratios
# mix_audio(file1, 1.5, file2, 0.5, 'AW_louder_than_AM.wav') # AW is much louder than AM
# mix_audio(file1, 1.1, file2, 0.9, 'AW_bit_louder_than_AM.wav') # AW is a bit louder than AM
# mix_audio(file1, 1.0, file2, 1.0, 'AW_AM_balanced.wav') # AW and AM sounds are balanced
# mix_audio(file1, 0.9, file2, 1.1, 'AM_bit_louder_than_AW.wav') # AM is a bit louder than AW
# mix_audio(file1, 0.5, file2, 1.5, 'AM_louder_than_AW.wav') # AM is much louder than AW
mix_audio(file1, 1.0, file2, 1.0, 'timit.wav')