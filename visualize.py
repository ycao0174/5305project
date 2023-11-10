import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Function to plot waveform and spectrogram
def plot_audio(file, title, ax_waveform, ax_spectrogram):
    # Load audio file
    data, samplerate = sf.read(file)
    # Plot waveform
    librosa.display.waveshow(data, sr=samplerate, ax=ax_waveform)
    ax_waveform.set_title(title + ' Waveform')
    ax_waveform.set_xlabel('Time')
    ax_waveform.set_ylabel('Amplitude')

    # Plot spectrogram
    S = librosa.feature.melspectrogram(y=data, sr=samplerate)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=samplerate, ax=ax_spectrogram)
    ax_spectrogram.set_title(title + ' Spectrogram')
    fig.colorbar(img, ax=ax_spectrogram, format='%+2.0f dB')

# Create subplots
fig, axs = plt.subplots(4, 2, figsize=(15, 10))

# Original separated data
plot_audio('mix_1.wav', 'Original Mix 1', axs[0, 0], axs[0, 1])
plot_audio('mix_2.wav', 'Original Mix 2', axs[1, 0], axs[1, 1])

# Predicted separated data
plot_audio('predict_1.wav', 'Predicted Mix 1', axs[2, 0], axs[2, 1])
plot_audio('predict_2.wav', 'Predicted Mix 2', axs[3, 0], axs[3, 1])

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('audio_comparison.png')
plt.show()
