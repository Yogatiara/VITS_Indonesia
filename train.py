import os
import json
import argparse
import itertools
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import tqdm

import commons
import model_utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
  DurationDiscriminator
  )

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0
use_duration_discriminator = False



def main():
  hps = model_utils.get_hparams()

  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  print(f"-----{torch.cuda.get_device_name(0)}-----")
  n_gpus = torch.cuda.device_count()
  print(f"{'-----Single GPU-----' if n_gpus <= 1 else '-----Multiple GPU-----'}")

  if hps.train.eight_bit_optimizer:
    print("-----Using 8-bit optimizer-----")

  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = hps.train.port
  torch.cuda.empty_cache()

  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step, use_duration_discriminator
  
  if n_gpus <= 1 or rank == 0:
    logger = model_utils.get_logger(hps.model_dir)
    logger.info(hps)
    model_utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  if n_gpus > 1:
    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
  

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)

  collate_fn = TextAudioCollate()
  
  if n_gpus <= 1:
    train_loader = DataLoader(
        train_dataset, num_workers=0, shuffle=True, pin_memory=True,
        collate_fn=collate_fn, batch_size=hps.train.batch_size
    )
  else:
    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size, hps.data.audio_boundaries, 
        num_replicas=n_gpus, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, num_workers=0, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler
    )
  if n_gpus <= 1 or rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
  
  if hps.model.use_duration_discriminator:
    use_duration_discriminator = True
    print("-----Using duration predictor with discriminator-----")
    net_dur_disc = DurationDiscriminator(
      hps.model.hidden_channels,
      hps.model.hidden_channels,
      hps.model.kernel_size,
      hps.model.p_dropout,
      gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0
    ).cuda(rank)
  else:
    net_dur_disc = None
    use_duration_discriminator = False

  
  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  if hps.train.eight_bit_optimizer:
    import bitsandbytes as bnb
    optim_g = bnb.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    optim_d = bnb.optim.AdamW8bit(
        net_d.parameters(),
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    if net_dur_disc is not None:
      optim_dur_disc = bnb.optim.AdamW8bit(
        net_dur_disc.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps = hps.train.eps
    ) 
    else:
      optim_dur_disc = None

  else:
    optim_g = torch.optim.AdamW(
        net_g.parameters(), 
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate, 
        betas=hps.train.betas, 
        eps=hps.train.eps)
  
    if net_dur_disc is not None:
      optim_dur_disc = torch.optim.AdamW(
        net_dur_disc.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps = hps.train.eps
    )
    else:
      optim_dur_disc = None
    
  
  if n_gpus > 1:
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    if net_dur_disc is not None:
      net_dur_disc = DDP(net_dur_disc, device_ids=[rank])

  try:
    _, _, _, epoch_str = model_utils.load_checkpoint(model_utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = model_utils.load_checkpoint(model_utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)

    if net_dur_disc is not None:
       _, _, _, epoch_str = model_utils.load_checkpoint(model_utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"), net_dur_disc, optim_dur_disc)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  if net_dur_disc is not None:
    scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  else:
    scheduler_dur_disc = None

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if n_gpus <= 1:
      train_and_evaluate(n_gpus, rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      if rank==0:
        train_and_evaluate(n_gpus, rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
      else:
        train_and_evaluate(n_gpus, rank, epoch, hps, [net_g, net_d, net_dur_disc], [optim_g, optim_d, optim_dur_disc], [scheduler_g, scheduler_d, scheduler_dur_disc], scaler, [train_loader, None], None, None)

    scheduler_g.step()
    scheduler_d.step()
    if net_dur_disc is not None:
      scheduler_dur_disc.step()


def train_and_evaluate(n_gpus, rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):   
  net_g, net_d, net_dur_disc = nets
  optim_g, optim_d, optim_dur_disc = optims
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  if n_gpus > 1:
    train_loader.batch_sampler.set_epoch(epoch)

  global global_step

  net_g.train()
  net_d.train()

  if net_dur_disc is not None:
      net_dur_disc.train()

  if n_gpus <= 1:
    loader = tqdm.tqdm(train_loader, desc="Loading train data")
  else:
    if rank == 0:
      loader = tqdm.tqdm(train_loader, desc="Loading train data")
    else:
      loader = train_loader

  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    
    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q), (hidden_x, logw, logw_) = net_g(x, x_lengths, spec, spec_lengths)

      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat = y_hat.float()
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

        # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc

        # Duration Discriminator
      if net_dur_disc is not None:
        y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach())
        with autocast(enabled=False):
          loss_dur_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_dur_hat_r, y_dur_hat_g)
          loss_dur_disc_all = loss_dur_disc

        optim_dur_disc.zero_grad()
        scaler.scale(loss_dur_disc_all).backward()
        scaler.unscale_(optim_dur_disc)
        grad_norm_dur_disc = commons.clip_grad_value_(
          net_dur_disc.parameters(), None
        )
        scaler.step(optim_dur_disc)

    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      if net_dur_disc is not None:
        y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel

        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        if net_dur_disc is not None:
          loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
          loss_gen_all += loss_dur_gen

    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if n_gpus <= 1 or rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]

        loss_names = ['Disc Loss', 'Gen Loss', 'FM Loss', 'Mel Loss', 'Dur Loss', 'KL Loss']
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
            epoch,
            100. * batch_idx / len(train_loader)))

        # Log losses side by side
        loss_info = ' | '.join(['{}: {}'.format(name, loss.item()) for name, loss in zip(loss_names, losses)])
        logger.info(loss_info)

        # Log additional info like global step and learning rate
        logger.info('Global Step: {}, Learning Rate: {}, total loss: {}'.format(global_step, lr, loss_gen_all))
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        if net_dur_disc is not None:
          scalar_dict.update({"loss/dur_disc/total": loss_dur_disc_all, "grad_norm_dur_disc": grad_norm_dur_disc,})

        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        if net_dur_disc is not None:
          scalar_dict.update({"loss/dur_gen/{}".format(i): v for i, v in enumerate(losses_dur_gen)})
          scalar_dict.update({"loss/dur_disc_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
          scalar_dict.update({"loss/d_disc_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

        image_dict = { 
            "slice/mel_org": model_utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": model_utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": model_utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": model_utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        model_utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(n_gpus, hps, net_g, eval_loader, writer_eval)
        model_utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        model_utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

        if net_dur_disc is not None:
          model_utils.save_checkpoint(net_dur_disc, optim_dur_disc, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)))
        model_utils.remove_old_checkpoints(hps.model_dir,hps.train.boundary_sorted_ckpts, prefixes=hps.train.remove_model_ckpts)
    global_step += 1
  
  if n_gpus <= 1 or rank == 0:
    logger.info('====> Epoch: {} '.format(epoch))

 
def evaluate(n_gpus, hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        # break
      if n_gpus <= 1:
        y_hat, attn, mask, *_ = generator.infer(x, x_lengths, max_len=1000)
      else:
        y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": model_utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": model_utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    model_utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
