
# from text import cleaners

# print(dir(cleaners))

# import torch
# import torch.distributed as dist

# dist.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29500", rank=0, world_size=1)
# print("NCCL Initialized successfully!")

import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt

# import os

# folder_path = './dataset/ITKTTS-IDN/utterance' 

# for filename in os.listdir(folder_path):
#     if filename.endswith('.wav') and filename[:4].isdigit():
#         old_path = os.path.join(folder_path, filename)
        
#         new_filename = f"ITKTTS001-{filename}"
#         new_path = os.path.join(folder_path, new_filename)
        
#         os.rename(old_path, new_path)
#         print(f"Renamed: {filename} -> {new_filename}")

# print("berhasil")

# import numpy as np
# import librosa
# import noisereduce as nr
# import soundfile as sf

# # Load audio
# y, sr = librosa.load("generate_1.wav", sr=None)

# # Ambil bagian awal (0.5 detik pertama) sebagai sampel noise
# noise_sample = y[:int(sr * 0.5)]

# # Lakukan noise reduction
# y_denoised = nr.reduce_noise(y=y, y_noise=noise_sample, sr=sr, prop_decrease=0.8)

# # Simpan hasil audio bersih
# sf.write("clean.wav", y_denoised, sr)

# print("Audio dengan noise reduction telah disimpan sebagai 'clean.wav'")



# input_file = 'dataset/ITKTTS-IDN/transkip.txt'     
# output_file = 'output.txt'   

# with open(input_file, 'r', encoding='utf-16le') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
#     for line in infile:
#         parts = line.strip().split(':', 1)
#         if len(parts) == 2:
#             nomor, teks = parts
#             nomor_padded = nomor.zfill(4)  
#             new_line = f"DUMMY1/LJ001-{nomor_padded}.wav|{teks.strip()}\n"
#             outfile.write(new_line)

# print("File berhasil dikonversi!")


# import os
# import glob

# # Direktori tempat file .pt berada
# directory = "dataset/ITKTTS-IDN/utterance"  # Ganti dengan path yang sesuai

# # Cari semua file .pt dalam direktori
# pt_files = glob.glob(os.path.join(directory, "*.pt"))

# # Hapus setiap file yang ditemukan
# for file in pt_files:
#     try:
#         os.remove(file)
#         print(f"Deleted: {file}")
#     except Exception as e:
#         print(f"Error deleting {file}: {e}")

# print("Done!")

# import librosa
# y, sr = librosa.load("dataset/ITKTTS-IDN/utterance/ITKTTS001-0180.wav", sr=None)
# print(f"Sample rate: {sr}, Duration: {len(y)/sr:.2f} sec")
import glob
import os

# # Path ke folder dataset
dataset_path = "dataset/ITKTTS-IDN/utterance"  # Ganti dengan path yang benar

# Cari semua file .pt dalam folder dataset
pt_files = glob.glob(os.path.join(dataset_path, "*.pt"))

# Hapus semua file .pt
for file in pt_files:
    os.remove(file)
    print(f"Deleted: {file}")


# import torch

# def load_and_fix_spectrogram(spec_filename):
#     spec = torch.load(spec_filename)

#     print("Original shape:", spec.shape)

#     # Jika tensor memiliki batch dimension (1, freq, time), squeeze
#     if len(spec.shape) == 3 and spec.shape[0] == 1:
#         spec = spec.squeeze(0)
#     elif len(spec.shape) != 2:
#         raise ValueError(f"Unexpected shape {spec.shape}, expected (freq, time)")

#     print("Final shape:", spec.shape)
#     return spec

# # Contoh pemanggilan
# spec = load_and_fix_spectrogram("DUMMY1/LJ001-0002.spec.pt")

# plt.figure(figsize=(10, 5))
# plt.imshow(spec.numpy(), aspect='auto', origin='lower', cmap='magma')
# plt.colorbar(label="Magnitude")
# plt.title("Spectrogram")
# plt.xlabel("Time Frames")
# plt.ylabel("Frequency Bins")
# plt.show()