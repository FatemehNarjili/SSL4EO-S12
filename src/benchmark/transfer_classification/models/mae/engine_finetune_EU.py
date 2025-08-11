# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
import os
from typing import Iterable, Optional
from torch.utils.tensorboard import SummaryWriter

import torch

from timm.data import Mixup
from timm.utils import accuracy

#import util.misc as misc
#import util.lr_sched as lr_sched
from .util import misc
from .util import lr_sched

from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))


from src.benchmark.transfer_classification.classification_metrics import Metrics_Micro


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    train_metrics = Metrics_Micro([], []) # Collect true labels and predictions
    epoch_loss = 0.0
    

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        train_metrics.y_true.extend(targets.cpu().numpy()) # Collect true labels

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        #print(samples.shape,samples.dtype,targets.shape,targets.dtype)
        device =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        with  torch.amp.autocast(device):
            outputs = model(samples)
            loss = criterion(outputs, targets.long())
            train_metrics.y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy()) # Collect predictions

        loss_value = loss.item()
        epoch_loss += loss_value   # Accumulate loss for the epoch

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        elif device == "cpu":
            pass

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Calculate and log metrics
    
    precision = train_metrics.precision()
    recall = train_metrics.recall()
    f1 = train_metrics.f1()
    confusion = train_metrics.confusion_matrix()
    accuracy = train_metrics.accuracy()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result.update({ 
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": confusion,
            "accuracy": accuracy,
            'loss': epoch_loss / len(data_loader)})  # Return average loss for the

    return result

@torch.no_grad()
def evaluate(data_loader, model, device, criterion):
    #criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    val_metrics = Metrics_Micro([], []) # collect true labels and predictions
    val_loss = 0.0

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1] 
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        val_metrics.y_true.extend(target.cpu().numpy())  # Collect true labels

        # compute output
        #print(images.shape,images.dtype,target.shape,target.dtype)
        # with torch.amp.autocast('cuda'):
        device =  "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        with  torch.amp.autocast(device):
            output = model(images)
            loss = criterion(output, target)

            val_loss += loss.item()  # Accumulate loss for the validation set

            val_metrics.y_pred.extend(torch.argmax(output, dim=1).cpu().numpy()) # Collect predictions

        score = torch.sigmoid(output).detach().cpu()
        acc1 = accuracy_score(target.cpu(), torch.argmax(score,axis=1)) * 100.0
        acc5 = acc1
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        metric_logger.meters['acc5'].update(acc5, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result.update({ 
            "precision": val_metrics.precision(),
            "recall": val_metrics.recall(),
            "f1": val_metrics.f1(),
            "confusion": val_metrics.confusion_matrix(),
            "accuracy": val_metrics.accuracy(),
            'loss': val_loss / len(data_loader)})  # Return average loss for the

    return result

