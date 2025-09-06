import torch
import os
from demucs.pretrained import get_model
from demucs.separate import load_track
import torchaudio
from demucs.apply import apply_model
import numpy as np

# def save_track(tensor, path, samplerate): # float32
#     torchaudio.save(path, tensor, samplerate)

def save_track(tensor, path, samplerate): # int16
    # Downmix to mono
    tensor_mono = torch.mean(tensor, dim=0, keepdim=True)
    # Convert to int16
    tensor_clamped = torch.clamp(tensor_mono, -1, 1)
    tensor_int16 = (tensor_clamped * 32767).to(torch.int16)
    torchaudio.save(
        path,
        tensor_int16,
        samplerate,
        encoding='PCM_S',       
        bits_per_sample=16
    )
    
model = get_model('htdemucs').cpu()

input_folder = "Audio-Filter-7.1.4/bedwd"
output_root = "Audio-Filter-7.1.4/bedwd_sepr"

os.makedirs(output_root, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        channel_name = filename.split('_')[-1].split('.')[0]
        
        audio_path = os.path.join(input_folder, filename)
        audio = load_track(
            audio_path, 
            model.audio_channels, 
            model.samplerate
        )

        audio_tensor = torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio
        audio_tensor = audio_tensor.unsqueeze(0) 

        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device='cpu', split=True)[0]
        
        for source, component in zip(sources, model.sources):
            output_name = f"{os.path.basename(input_folder)}_{component}_{channel_name}.wav"
            save_track(
                source,
                os.path.join(output_root, output_name),
                model.samplerate
            )

import shutil

for filename in os.listdir(output_root):
    if filename.endswith(".wav"):
        name = filename[:-4]  
        parts = name.split('_')
        if len(parts) >= 3:
            component = parts[1].lower()    
            channel = parts[2]     

            dest_folder = os.path.join(output_root, component)
            os.makedirs(dest_folder, exist_ok=True)

            src_path = os.path.join(output_root, filename)
            dest_path = os.path.join(dest_folder, f"{channel}.wav")

            shutil.move(src_path, dest_path)