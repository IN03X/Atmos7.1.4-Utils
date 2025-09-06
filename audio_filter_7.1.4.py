import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.io import wavfile
import pdb

def load_mono_wav_file(file_path):
    sample_rate, data = wavfile.read(file_path)
    if len(data.shape) > 1:
        data = data[:, 0]  
    return sample_rate, data


def compute_stft(data, sample_rate, nperseg=512):
    f, t, Zxx = stft(data, fs=sample_rate, nperseg=nperseg)
    return f, t, Zxx

def calculate_percentile_range(data, lower_percentile=10, upper_percentile=90):
    flat_data = np.abs(data).flatten()
    vmin = np.percentile(flat_data, lower_percentile)
    vmax = np.percentile(flat_data, upper_percentile)
    return vmin, vmax

def plot_stft(f, t, Zxx, title, ax, vmin=None, vmax=None , cmap='jet'):
    magnitude = np.abs(Zxx)
    eps = 1e-10
    pcm = ax.pcolormesh(t, f, 20 * np.log10(magnitude+ eps), shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    return pcm


def process_and_plot(mono_path, channel_paths, output_dir, output_name):
    os.makedirs(output_dir, exist_ok=True)
    
    sr_mono, data_mono = load_mono_wav_file(mono_path)
    f_mono, t_mono, Zxx_mono = compute_stft(data_mono, sr_mono)
    
    fig, axes = plt.subplots(len(channel_paths) + 1, 2, figsize=(15, 5 * (len(channel_paths) + 1)))
    pcm = plot_stft(f_mono, t_mono, Zxx_mono, 'Mono Signal STFT', axes[0, 0], vmin=-60, vmax=70)
    fig.colorbar(pcm, ax=axes[0,0], cax=axes[0,1])
    axes[0,1].set_ylabel('Magnitude (dB)')

    for idx, channel_path in enumerate(channel_paths):
        sr_ch, data_ch = load_mono_wav_file(channel_path)
        f_ch, t_ch, Zxx_ch = compute_stft(data_ch, sr_ch)

        # make sure the length of the two signals are the same
        min_len = min(Zxx_ch.shape[1], Zxx_mono.shape[1])
        Zxx_ch = Zxx_ch[:, :min_len]
        Zxx_mono = Zxx_mono[:, :min_len]
        f_ch = f_ch[:min_len]
        f_mono = f_mono[:min_len]
        t_ch = t_ch[:min_len]
        t_mono = t_mono[:min_len]

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(Zxx_ch, Zxx_mono)
            ratio[~np.isfinite(ratio)] = 1e-10 # Inf Situation

        row = idx + 1

        plot_stft(f_ch, t_ch, Zxx_ch, f'Ch:{os.path.basename(channel_path)}', axes[row, 0], vmin=-60, vmax=70)

        # vmin, vmax = calculate_percentile_range(ratio, 10, 99)
        plot_stft(f_ch, t_ch, ratio, f'Ch/Mono:{os.path.basename(channel_path)}', axes[row, 1], vmin=-30, vmax=20, cmap='viridis')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()
    print(f'Saved combined analysis: {output_path}')

        


if __name__ == "__main__":
    input_dir = "Audio-Filter-7.1.4/bedwd_sepr/vocals/"
    # input_dir = "Audio-Filter-7.1.4/gt5/"
    output_dir = "Audio-Filter-7.1.4/STFT_results"

    channel_names = [
        'L',    # FL  - Front Left
        # 'R',    # FR  - Front Right
        'C',    # FC  - Front Center
        # 'LFE',  # LFE - Low Frequency
        'Lss',  # SL  - Side Left
        # 'Rss',  # SR  - Side Right
        'Lsr',  # BL  - Back Left
        # 'Rsr',  # BR  - Back Right
        'Ltf',  # TFL - Top Front Left
        # 'Rtf',  # TFR - Top Front Right
        'Ltr',  # TrL - Top Rear Left
        # 'Rtr'   # TrR - Top Rear Right
    ]
    channel_paths = [input_dir + channel_name + ".wav" for channel_name in channel_names]

    mono_path = "Audio-Filter-7.1.4/bedwd_mono_sepr/vocals.wav"

    output_name = "bedwd_sepr_STFT_vocals.png"
    
    process_and_plot(mono_path, channel_paths, output_dir, output_name)