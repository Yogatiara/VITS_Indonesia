import argparse
import os
import time
import librosa
import text
import soundfile as sf
import utils
import random
from utils import load_filepaths_and_text
from tqdm import tqdm


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--wav_files", default="dataset/ITKTTS-IDN/utterance")
  parser.add_argument("--input_transcript", default="dataset/ITKTTS-IDN/transkrip.txt")
  parser.add_argument("--output_train", default="filelists/indonesian/itktts_audio_text_train_filelist.txt")
  parser.add_argument("--output_val", default="filelists/indonesian/itktts_audio_text_val_filelist.txt")
  parser.add_argument("--configure_path", default="configs/ITKTTS.json")
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["filelists/indonesian/itktts_audio_text_train_filelist.txt", "filelists/indonesian/itktts_audio_text_val_filelist.txt"])
  parser.add_argument("--text_cleaners", nargs="+", default=["indonesian_cleaners"])


  args = parser.parse_args()
  hps = utils.get_hparams_from_file(args.configure_path)
  trimming_done = False


  if hps.data.trimming:
      folder = args.wav_files
      file_list = [f for f in os.listdir(folder) if f.endswith('.wav')]  

      with tqdm(file_list, desc="Trimming audio", unit="file", dynamic_ncols=True, leave=True) as pbar:
          for filename in pbar:
              file_path = os.path.join(folder, filename)

              y, sr = librosa.load(file_path, sr=None)
              yt, index = librosa.effects.trim(y, top_db=hps.data.top_db)  

              original_duration = librosa.get_duration(y=y, sr=sr)
              trimmed_duration = librosa.get_duration(y=yt, sr=sr)


              sf.write(file_path, yt, sr)

      print("All audio files successfully trimmed!")
      trimming_done = True




#   if not hps.data.trimming or (hps.data.trimming and trimming_done):
#       with open(args.input_transcript, 'r', encoding='utf-8') as file:
#           lines = [line.strip() for line in file.readlines() if line.strip()]
      
#       modified_lines = [f"{args.wav_files}/{line}\n" for line in lines]
      
#       random.shuffle(modified_lines)
      
#       total_lines = len(modified_lines)
#       train_count = int(total_lines * hps.data.train_ratio)
      
#       train_lines = modified_lines[:train_count]
#       eval_lines = modified_lines[train_count:]
      
#       with open(args.output_train, 'w', encoding='utf-8') as file:
#           file.writelines(train_lines)
      
#       with open(args.output_val, 'w', encoding='utf-8') as file:
#           file.writelines(eval_lines)
      
#       print("Dataset has been split and modified!")

#   time.sleep(5)
#   for filelist in args.filelists:
#     print("START:", filelist)
#     filepaths_and_text = load_filepaths_and_text(filelist)
#     for i in range(len(filepaths_and_text)):
#       original_text = filepaths_and_text[i][args.text_index]
#       cleaned_text = text._clean_text(original_text, args.text_cleaners)
      
#       filepaths_and_text[i][args.text_index] = cleaned_text

#     new_filelist = filelist + "." + args.out_extension
#     with open(new_filelist, "w", encoding="utf-8") as f:
#       f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])