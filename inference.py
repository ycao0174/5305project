import os
import soundfile as sf
import torch
import yaml
import argparse
import look2hear.models
import librosa
import torchaudio
import time
# New imports for recording
import sounddevice as sd

parser = argparse.ArgumentParser()
parser.add_argument("--conf_dir", default="configs/tdanet.yml", help="Configuration file path")
parser.add_argument("--audio_path", required=True, help="Path to the input audio file")
parser.add_argument("--output_dir", required=True, help="Directory to save the separated audio files")
parser.add_argument("--record_duration", type=int, help="Duration to record audio in seconds", default=5)

# New function to record audio
def record_audio(duration, sample_rate=16000, channels=1, filename="recorded_audio.wav"):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()  # Wait until the recording is finished
    sf.write(filename, recording, sample_rate)  # Save the recording
    print(f"Recording finished and saved to {filename}")

def preprocess_audio(audio_path, sample_rate=16000):
    # Load audio
    waveform, current_sample_rate = sf.read(audio_path, dtype='float32')

    # If audio has two channels, take the first one assuming it is mono
    if len(waveform.shape) == 2:
        waveform = waveform[:, 0]

    # Check if resampling is needed
    if current_sample_rate != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=current_sample_rate, target_sr=sample_rate)

    # Convert to tensor and add batch dimension
    waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    return waveform_tensor, sample_rate


def main(config, audio_path, output_dir, record_duration):
    # Check if the audio file exists or if the recording is requested
    if not os.path.isfile(audio_path) and record_duration:
        record_audio(record_duration, sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
                     filename=audio_path)

    # Load the pre-trained model
    model = getattr(look2hear.models, config["train_conf"]["audionet"]["audionet_name"]).from_pretrain(
        "JusperLee/TDANetBest-4ms-LRS2",
        sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"],
        **config["train_conf"]["audionet"]["audionet_config"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Preprocess the audio
    preprocessed_waveform, sample_rate = preprocess_audio(audio_path,
                                                          sample_rate=config["train_conf"]["datamodule"]["data_config"]["sample_rate"])

    # Perform separation and time it
    start_time = time.time()
    with torch.no_grad():
        est_sources = model(preprocessed_waveform.to(device))
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Audio separation took {elapsed_time:.2f} seconds.")

    # Post-processing: convert the tensor to audio waveform and save the files
    est_sources_np = est_sources.squeeze(0).cpu().numpy()
    for i, est_source in enumerate(est_sources_np):
        # Ensure the output is mono for saving
        if len(est_source.shape) > 1:
            est_source = est_source.mean(axis=0)
        sf.write(os.path.join(output_dir, f"separated_{i + 1}.wav"), est_source, sample_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load configuration
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    arg_dic["train_conf"] = train_conf

    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Call main function with the new record_duration argument
    main(arg_dic, args.audio_path, args.output_dir, args.record_duration)
