import os
import argparse
import soundfile as sf
import numpy as np

channel_order = [
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

# https://www.audiokinetic.com/zh/public-library/2024.1.7_8863/?source=Help&id=downmix_tables#tbl_mono
weights = {
    'L':   0.707,  # Front Left
    'R':   0.707,  # Front Right
    'C':   1.0,  # Front Center
    'LFE': 0.0,  # LFE - typically omitted in mono
    'Lss': 0.5,  # Side Left - slightly reduced
    'Rss': 0.5,  # Side Right
    'Lsr': 0.5,  # Back Left - lower weight
    'Rsr': 0.5,  # Back Right
    'Ltf': 0.5,  # Top Front Left
    'Rtf': 0.5,  # Top Front Right
    'Ltr': 0.354,  # Top Rear Left
    'Rtr': 0.354,  # Top Rear Right
}

def downmix_714_to_mono(input_path, output_path):
    print(f"Reading: {input_path}")
    data, sample_rate = sf.read(input_path, dtype='float32')

    if data.ndim == 1:
        print(f"[ERROR] '{input_path}' is not multi-channel.")
        return False
    if data.shape[1] != 12:
        print(f"[ERROR] Expected 12 channels, got {data.shape[1]} in '{input_path}'.")
        return False

    print("Downmixing to mono with standard weights...")
    mono = np.zeros(data.shape[0], dtype=np.float32)

    for idx, ch_name in enumerate(channel_order):
        weight = weights[ch_name]
        if weight > 0:
            mono += weight * data[:, idx]

    # Normalize by total active weight (optional, keeps level consistent)
    total_weight = sum(weights.values())
    mono /= total_weight

    sf.write(output_path, mono, sample_rate, subtype='PCM_16')
    print(f"[SUCCESS] Mono file saved: {output_path}")
    return True

if __name__ == "__main__":
    input_714_file = "Audio-Filter-7.1.4/bedwd.wav"
    output_mono_file = "Audio-Filter-7.1.4/bedwd_mono.wav"

    downmix_714_to_mono(input_714_file, output_mono_file)