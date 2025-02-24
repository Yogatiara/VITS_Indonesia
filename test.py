
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

import librosa
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Load semua file audio dalam folder dataset
audio_files = glob.glob("dataset/ITKTTS-IDN/utterance/*.wav")  # Ganti dengan path datasetmu
hop_length = 256
sr = 22050

file_names = []
num_frames_list = []
zero_frame_files = []  # List file dengan 0 frame

print(f"🔍 Memproses {len(audio_files)} file audio...\n")

# Loop untuk menghitung frame setiap file dengan progress bar
for file in tqdm(audio_files, desc="⏳ Menghitung jumlah frame", unit="file"):
    try:
        y, _ = librosa.load(file, sr=sr)
        num_frames = len(y) // hop_length  # Hitung jumlah frame
        file_name = file.split("/")[-1]

        if num_frames == 0:
            print(f"⚠️ WARNING: {file_name} memiliki 0 frame! Periksa file ini.")
            zero_frame_files.append(file_name)
            continue  # Skip file yang bermasalah

        file_names.append(file_name)  # Simpan nama file
        num_frames_list.append(num_frames)  # Simpan jumlah frame

        print(f"✅ {file_name}: {num_frames} frame")

    except Exception as e:
        print(f"❌ ERROR: Gagal memproses {file}. Kesalahan: {e}")

# Cek apakah ada file yang valid sebelum membuat grafik
if num_frames_list:
    # Gunakan indeks angka sebagai label X
    x_indices = np.arange(len(file_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x_indices, num_frames_list, color="skyblue")

    # Label & judul
    plt.xlabel("File Index", fontsize=12)
    plt.ylabel("Jumlah Frame", fontsize=12)
    plt.title("Distribusi Jumlah Frame per File Audio", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Tambahkan garis batas min & max frame
    plt.axhline(y=min(num_frames_list), color="red", linestyle="--", label=f"Min Frame: {min(num_frames_list)}")
    plt.axhline(y=max(num_frames_list), color="green", linestyle="--", label=f"Max Frame: {max(num_frames_list)}")

    plt.legend()

    # Simpan sebagai file PNG
    output_file = "frame_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n📊 Grafik berhasil disimpan sebagai '{output_file}' ✅")

# Laporkan file yang memiliki 0 frame di akhir proses
if zero_frame_files:
    print("\n⚠️ Daftar file dengan 0 frame:")
    for f in zero_frame_files:
        print(f"   - {f}")
    print("\n🛠️ Periksa file di atas karena mungkin rusak atau kosong.")
else:
    print("\n✅ Tidak ada file dengan 0 frame. Semua file valid! 🎉")
