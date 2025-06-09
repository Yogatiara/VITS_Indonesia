import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import sys
import json
import math
import torch
from pydub import AudioSegment 

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import soundfile as sf
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))


import scipy.io.wavfile as wav
import commons
import model_utils

from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
SOURCEIMAGEDIR = "./source_audio"

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def load_model(predictor, n_data,   with_duration_discriminator, hps):
 


  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda()
  _ = net_g.eval()


  if with_duration_discriminator:
    _ = model_utils.load_checkpoint(f"../logs/{predictor}/{n_data}/with discriminator/G_DP_adversarial.pth", net_g, None)
  else:
    _ = model_utils.load_checkpoint(f"../logs/{predictor}/{n_data}/G_DP.pth", net_g, None)

  return net_g


def create_audio(small_model, text, DP_adversarial_learning, SDP):
  if small_model:
    n_data = 343
  else:
    n_data = 1250

  if SDP:
    duration_predictor = 'SDP'
    hps = model_utils.get_hparams_from_file("../configs/ITKTTS.json")
  else:
    duration_predictor = 'DDP'
    hps = model_utils.get_hparams_from_file("../configs/ITKTTS_nosdp.json")


  stn_tst = get_text(f"{text if '?' in text or '!' in text else text + '.'}", hps)
  net_g = load_model(duration_predictor, n_data, DP_adversarial_learning, hps)

  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  with torch.no_grad():
      x_tst = stn_tst.cuda().unsqueeze(0)
      x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
      audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
  # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

  if not os.path.exists(SOURCEIMAGEDIR):
    os.makedirs(SOURCEIMAGEDIR)

  file_path = os.path.join(SOURCEIMAGEDIR, "generate.wav")
  sf.write(file_path, audio, hps.data.sampling_rate)

 
  wav_file = AudioSegment.from_file("./source_audio/generate.wav") 

  louder_wav_file = wav_file + 5

  louder_wav_file.export(out_f = "./source_audio/generate.wav", 
                        format = "wav")

  with sf.SoundFile(file_path) as f:
    sample_rate = f.samplerate
    frames = f.frames
    duration = frames / sample_rate
    encoding = f.subtype

  
  return {
      "sample_rate": sample_rate,
      "duration_seconds": round(duration, 2),
      "encoding": encoding,
      "audio_url": "./source_audio/generate.wav"

  }
  






