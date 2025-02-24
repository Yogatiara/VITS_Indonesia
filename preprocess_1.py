import argparse
import os
import librosa
import soundfile as sf
import utils
import random

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--wav_files", default="dataset/ITKTTS-IDN/utterance")
  parser.add_argument("--input_transcript", default="dataset/ITKTTS-IDN/new_transkip.txt")
  parser.add_argument("--output_train", default="filelists/indonesian/itktts_audio_text_train_filelist.txt")
  parser.add_argument("--output_val", default="filelists/indonesian/itktts_audio_text_val_filelist.txt")
  parser.add_argument("--configure_path", default="configs/ITKTTS.json")


  args = parser.parse_args()
  hps = utils.get_hparams_from_file(args.configure_path)
  trimming_done = False

  if (hps.data.trimming):
    folder = args.wav_files
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder, filename)

            y, sr = librosa.load(file_path, sr=None)
            yt, index = librosa.effects.trim(y, top_db=hps.data.top_db)  

            original_duration = librosa.get_duration(y=y, sr=sr)
            trimmed_duration = librosa.get_duration(y=yt, sr=sr)

            print(f"{filename} - Before: {original_duration:.2f}s, After: {trimmed_duration:.2f}s")

            sf.write(file_path, yt, sr)

    print("All audio file successful trimmed!")
    trimming_done =True


  if not hps.data.trimming or (hps.data.trimming and trimming_done):
    with open(args.input_transcript, 'r', encoding='utf-8') as file:
      lines = [line.strip() + '\n' for line in file.readlines() if line.strip()]
    
    random.shuffle(lines)
    
    total_lines = len(lines)
    train_count = int(total_lines * hps.data.train_ratio)
    
    train_lines = lines[:train_count]
    eval_lines = lines[train_count:]
    
    with open(args.output_train, 'w', encoding='utf-8') as file:
        file.writelines(train_lines)
    
    with open(args.output_val, 'w', encoding='utf-8') as file:
        file.writelines(eval_lines)
    
    print("Dataset has been split")