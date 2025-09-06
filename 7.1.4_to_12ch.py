import numpy as np
from pydub import AudioSegment
import os

def main(wav_file_path, wav_file_name):
    audio = AudioSegment.from_wav(wav_file_path + wav_file_name + ".wav")

    sample_rate = audio.frame_rate
    channels = audio.channels
    samples = np.array(audio.get_array_of_samples())

    samples = samples.reshape((-1, channels))

    print(f"Audio Info: {channels} channels, Sample Rate: {sample_rate} Hz, Total Frames: {len(samples)}")

    if channels != 12 and channels != 14:
        print(f"Error: Expected 12/14 channels, but got {channels}")
        return

    channel_names = [
        'L',    # FL  - Front Left
        'R',    # FR  - Front Right
        'C',    # FC  - Front Center
        'LFE',  # LFE - Low Frequency
        'Lss',  # SL  - Side Left
        'Rss',  # SR  - Side Right
        'Lsr',  # BL  - Back Left
        'Rsr',  # BR  - Back Right
        'Ltf',  # TFL - Top Front Left
        'Rtf',  # TFR - Top Front Right
        'Ltr',  # TrL - Top Rear Left
        'Rtr'   # TrR - Top Rear Right
    ]

    output_dir = os.path.join(wav_file_path, wav_file_name)
    os.makedirs(output_dir, exist_ok=True)

    # Export first 12 channels as mono
    for i in range(12):
        channel_data = samples[:, i].astype(np.int16)
        mono_audio = AudioSegment(
            channel_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        # filename = f"channel_{i+1}_{channel_names[i]}.wav"
        filename = f"{channel_names[i]}.wav"
        mono_audio.export(os.path.join(output_dir, filename), format="wav")
        print(f"Saved: {filename}")

    if channels == 14:
        # Export channel 13 and 14 as stereo
        stereo_data = samples[:, 12:14].astype(np.int16)
        stereo_audio = AudioSegment(
            stereo_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=2
        )
        stereo_filename = "binaural.wav"
        stereo_audio.export(os.path.join(output_dir, stereo_filename), format="wav")
        print(f"Saved: {stereo_filename}")

if __name__ == "__main__":
    # wav_name = "beautiful_poeple_halo"
    # wav_name = "bed_no_breakfast_wd"
    # wav_name = "bliss"
    # wav_name = "drop_dead_gorgeous"
    # wav_name = "est2"
    # wav_name = "exes"
    # wav_name = "for_life"
    # wav_name = "red_lights"
    # wav_name = "stars_will_again"
    # wav_name = "tears"
    # wav_name = "victory_lap"
    wav_name = "bedwd"
    # wav_path = "Audio-7.1.4/"
    wav_path = "Audio-Filter-7.1.4/"
    main(wav_path, wav_name)