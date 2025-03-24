import argparse
import logging
import librosa
import glob
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configure_path", default="configs/ITKTTS.json")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.configure_path)
    logging.getLogger('matplotlib.font_manager').disabled = True
    audio_files = glob.glob(f"{hps.data.utterance_files}/*.wav")  
    hop_length = hps.data.hop_length
    sr = hps.data.sampling_rate

    file_names = []
    num_frames_list = []
    zero_frame_files = []  

    print(f"Processing {len(audio_files)} audio files...\n")

    for file in tqdm(audio_files, desc="Counting frames", unit="file"):
        try:
            y, _ = librosa.load(file, sr=sr)
            num_frames = max(1, (len(y) - hop_length) // hop_length + 1)  # Perbaikan rumus
            file_name = file.split("/")[-1]

            if num_frames == 1:
                print(f"WARNING: {file_name} has very few frames! Please check this file.")
                zero_frame_files.append(file_name)
                continue  

            file_names.append(file_name)  
            num_frames_list.append(num_frames)  

            print(f"{file_name}: {num_frames} frames")

        except Exception as e:
            print(f"ERROR: Failed to process {file}. Error: {e}")

    if num_frames_list:
        x_indices = np.arange(len(file_names))
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.figure(figsize=(12, 6))
        plt.bar(x_indices, num_frames_list, color="skyblue")

        plt.xlabel("File Index", fontsize=12)
        plt.ylabel("Number of Frames", fontsize=12)
        plt.title("Frame Count Distribution per Audio File", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.axhline(y=min(num_frames_list), color="red", linestyle="--", label=f"Min Frames: {min(num_frames_list)}")
        plt.axhline(y=max(num_frames_list), color="green", linestyle="--", label=f"Max Frames: {max(num_frames_list)}")

        plt.legend()

        output_file = "frame_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nGraph successfully saved as '{output_file}'")

    if zero_frame_files:
        print("\nList of files with very few frames:")
        for f in zero_frame_files:
            print(f"   - {f}")
        print("\nPlease check the above files as they may be corrupted or empty.")
    else:
        print("\nNo files with 0 frames. All files are valid!")