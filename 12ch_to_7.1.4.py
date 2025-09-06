import os
import re
import subprocess
from collections import defaultdict

# input_dir = "Audio-Filter-7.1.4/gt5_sepr"
# output_dir = "Audio-Filter-7.1.4/gt5_sepr_7.1.4"
input_dir = "Audio-popmusic-7.1.4/popmusic-seperate-12channels"
output_dir = "Audio-popmusic-7.1.4/popsong_apps"
os.makedirs(output_dir, exist_ok=True)

ffmpeg_path = "ffmpeg"
# pattern = r'^(.+)_(drums|bass|vocals|other)_([A-Za-z]+)\.wav$'
pattern = r'^(.+)_(halo|wd)_binaural-7\.1\.4\.([A-Za-z]+)\.wav$'

groups = defaultdict(dict)

print("Scanning input files...")
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        match = re.match(pattern, filename)
        if match:
            prefix = f"{match.group(1)}_{match.group(2)}"
            channel = match.group(3)
            groups[prefix][channel] = filename
            print(f"  Found: {filename} -> Group '{prefix}', Channel '{channel}'")
        else:
            print(f"  Skipped (mismatched name): {filename}")

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

print("\nStarting merge process...")
for prefix, channels in groups.items():
    missing = [ch for ch in channel_order if ch not in channels]
    if missing:
        print(f"[ERROR] Group '{prefix}' missing channels: {missing}")
        continue

    print(f"[INFO] Merging '{prefix}.wav'...")
    cmd = [ffmpeg_path]
    for ch in channel_order:
        cmd.extend(["-i", os.path.join(input_dir, channels[ch])])

    amix = "".join(f"[{i}:a]" for i in range(12)) + "amerge=inputs=12[a]"
    output_file = os.path.join(output_dir, f"{prefix}.wav")
    cmd += [
        "-filter_complex", amix,
        "-map", "[a]",
        "-ac", "12",
        "-ar", "48000",
        "-acodec", "pcm_s16le",
        "-y",
        output_file
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[SUCCESS] Created: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed to create '{prefix}.wav': {e}")

print("\n✅ All processing completed.")