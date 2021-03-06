"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

import random
import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time

import sys
sys.path.append("..")
from dataloaders.vcr import VCR, VCRLoader

from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint, adjust_learning_rate

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models
#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-rationale',
    action="store_true",
    help='use rationale',
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)
parser.add_argument(
    '-plot',
    dest='plot',
    help='plot folder location',
    type=str,
)

args = parser.parse_args()
writer = SummaryWriter('runs/' + args.plot)
params = Params.from_file(args.params)
train, val = VCR.splits(mode='rationale' if args.rationale else 'answer',
                              embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                              only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True),
                              expand2obj36 = params['dataset_reader'].get('expand2obj36', False),)
NUM_THREADS = 3
torch.set_num_threads(NUM_THREADS)
NUM_GPUS = torch.cuda.device_count()
print (NUM_GPUS)
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td

num_workers = 4
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)

ARGS_RESET_EVERY = 20
print("Loading {} for {}".format(params['model'].get('type', 'WTF?'), 'rationales' if args.rationale else 'answer'), flush=True)
model = Model.from_params(vocab=train.vocab, params=params['model'])

if hasattr(model, 'detector'):
    for submodule in model.detector.backbone.modules():
        for p in submodule.parameters():
            p.requires_grad = False

model = DataParallel(model).cuda() if NUM_GPUS > 1 else model.cuda()
optimizer = Optimizer.from_params([x for x in model.named_parameters() if x[1].requires_grad],
                                  params['trainer']['optimizer'])

# lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
# scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model, optimizer, serialization_dir=args.folder)#,learning_rate_scheduler=scheduler)
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)

param_shapes = print_para(model)
num_batches = 0
global_train_loss = []
global_train_acc = []
global_val_loss = []
global_val_acc = []
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    train_results = []
    norms = []
    model.train()

    start = time.time()
    adjust_learning_rate(optimizer, epoch_num, args.rationale)
    print("Epoch", epoch_num, "learning_rate :", optimizer.param_groups[0]['lr'])
    
    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        batch = _to_gpu(batch)
        optimizer.zero_grad()
        output_dict = model(**batch)
        loss = output_dict['loss'].mean()
        if 'cnn_regularization_loss' in output_dict:
            loss += output_dict['cnn_regularization_loss'].mean()

        loss.backward()
        num_batches += 1
        #if scheduler:
            #scheduler.step_batch(num_batches)

        norms.append(
            clip_grad_norm(model.named_parameters(), max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
        )
        optimizer.step()
        
        train_results.append(pd.Series({'binary_loss': output_dict['loss'].mean().item(),
                                        'crl': output_dict[
                                            'cnn_regularization_loss'].mean().item() if 'cnn_regularization_loss' in output_dict else -1,
                                        'accuracy': (model.module if NUM_GPUS > 1 else model).get_metrics(
                                            reset=(b % ARGS_RESET_EVERY) == 0)['accuracy'],
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))
        if b % ARGS_RESET_EVERY == 0 and b > 0:
            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms:\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)

            print("time:", time.time()-start)

    epoch_stats = pd.DataFrame(train_results).mean()
    train_loss = epoch_stats['binary_loss']
    train_acc = epoch_stats['accuracy']
    writer.add_scalar('loss/train', train_loss, epoch_num)
    writer.add_scalar('accuracy/train', train_acc, epoch_num)
    global_train_loss.append(train_loss)
    global_train_acc.append(train_acc)
    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))

    val_probs = []
    val_labels = []
    val_loss_sum = 0.0
    model.eval()
    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
            output_dict = model(**batch)
            val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
            val_labels.append(batch['label'].detach().cpu().numpy())
            val_loss_sum += output_dict['loss'].mean().item() * batch['label'].shape[0]
    val_labels = np.concatenate(val_labels, 0)
    val_probs = np.concatenate(val_probs, 0)
    val_loss_avg = val_loss_sum / val_labels.shape[0]

    val_metric_per_epoch.append(float(np.mean(val_labels == val_probs.argmax(1))))
    # if scheduler:
    #     scheduler.step(val_metric_per_epoch[-1], epoch_num)

    print("Val epoch {} has acc {:.3f} and loss {:.3f}".format(epoch_num, val_metric_per_epoch[-1], val_loss_avg),
          flush=True)
    writer.add_scalar('loss/validation', val_loss_avg, epoch_num)
    writer.add_scalar('accuracy/validation',val_metric_per_epoch[-1], epoch_num)
    global_val_loss.append(val_loss_avg)
    global_val_acc.append(val_metric_per_epoch[-1])
    # if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
    #     print("Stopping at epoch {:2d}".format(epoch_num))
    #     break
    # if scheduler:
    #     save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,\
    #                     is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1),learning_rate_scheduler=scheduler)
    # else:
    save_checkpoint(model, optimizer, args.folder, epoch_num, val_metric_per_epoch,\
                        is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))


writer.close()
print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model, args.folder)
model.eval()
val_probs = []
val_labels = []
for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        output_dict = model(**batch)
        val_probs.append(output_dict['label_probs'].detach().cpu().numpy())
        val_labels.append(batch['label'].detach().cpu().numpy())
val_labels = np.concatenate(val_labels, 0)
val_probs = np.concatenate(val_probs, 0)
acc = float(np.mean(val_labels == val_probs.argmax(1)))
print("Final val accuracy is {:.3f}".format(acc))
np.save(os.path.join(args.folder, f'valpreds.npy'), val_probs)
np.save(os.path.join(args.folder, f'global_val_loss.npy'), global_val_loss)
np.save(os.path.join(args.folder, f'global_val_acc.npy'), global_val_acc)
np.save(os.path.join(args.folder, f'global_train_loss.npy'),global_train_loss )
np.save(os.path.join(args.folder, f'global_train_acc.npy'), global_train_acc)
