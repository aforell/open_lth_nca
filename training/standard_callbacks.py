# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import torch
import losses.loss_functions as loss_functions

from datasets.base import DataLoader
from foundations import hparams
from foundations.step import Step
from platforms.platform import get_platform
from training import checkpointing

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    average_precision_score, roc_auc_score,
    confusion_matrix, brier_score_loss
)


# Standard callbacks.
def save_model(output_location, step, model, optimizer, logger):
    model.save(output_location, step)


def save_logger(output_location, step, model, optimizer, logger):
    logger.save(output_location)


def create_timekeeper_callback():
    time_of_last_call = None

    def callback(output_location, step, model, optimizer, logger):
        if get_platform().is_primary_process:
            nonlocal time_of_last_call
            t = 0.0 if time_of_last_call is None else time.time() - time_of_last_call
            print(f'Ep {step.ep}\tIt {step.it}\tTime Elapsed {t:.2f}')
            time_of_last_call = time.time()
        get_platform().barrier()

    return callback


def create_eval_callback(eval_name: str, loader: DataLoader, verbose=False):
    """This function returns a callback."""

    time_of_last_call = None
    total_time = None
    max_vram_alloc = None

    def eval_callback(output_location, step, model, optimizer, logger):
        example_count = torch.tensor(0.0).to(get_platform().torch_device)
        total_loss = torch.tensor(0.0).to(get_platform().torch_device)
        total_dice = torch.tensor(0.0).to(get_platform().torch_device)
        total_correct = torch.tensor(0.0).to(get_platform().torch_device)
        all_probs, all_labels = [], []

        def correct(labels, outputs):
            return torch.sum(torch.eq(labels, output.argmax(dim=1)))

        model.eval()

        with torch.no_grad():
            for examples, labels in loader:
                examples = examples.to(get_platform().torch_device)
                labels = labels.to(get_platform().torch_device)
                output = model(examples)

                probs = torch.softmax(output, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

                labels_size = torch.tensor(len(labels), device=get_platform().torch_device)
                example_count += labels_size
                total_loss += model.loss_criterion(output, labels) * labels_size
                dice = loss_functions.dice_score(output, labels)
                total_dice += dice * labels_size
                total_correct += correct(labels, output)

        # Share the information if distributed.
        if get_platform().is_distributed:
            torch.distributed.reduce(total_loss, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total_correct, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(example_count, 0, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.reduce(total_dice, 0, op=torch.distributed.ReduceOp.SUM)

        total_loss = total_loss.cpu().item()
        total_correct = total_correct.cpu().item()
        example_count = example_count.cpu().item()
        total_dice = total_dice.cpu().item()

        #TODO check metrics for segmentation

        # Convert to numpy
        probs = torch.cat(all_probs).numpy()
        labels = torch.cat(all_labels).numpy()
        preds = probs.argmax(axis=1)

        # Metrics
        #macro_f1 = f1_score(labels, preds, average="macro")
        #pr_auc = average_precision_score(labels, probs, average="macro")
        #roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

        # Efficiency
        n_params = sum(p.numel() for p in model.parameters())
        vram_alloc = torch.cuda.max_memory_allocated(device=get_platform().torch_device) / 1024**2 if torch.cuda.is_available() else 0
        nonlocal max_vram_alloc
        max_vram_alloc = vram_alloc if max_vram_alloc is None else max(max_vram_alloc, vram_alloc)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if get_platform().is_primary_process:
            logger.add('{}_loss'.format(eval_name), step, total_loss / example_count)
            logger.add('{}_accuracy'.format(eval_name), step, total_correct / example_count)
            logger.add('{}_examples'.format(eval_name), step, example_count)
            logger.add('{}_dice'.format(eval_name), step, total_dice / example_count)
            #logger.add('{}_macro_f1'.format(eval_name), step, macro_f1)
            #logger.add('{}_pr_auc'.format(eval_name), step, pr_auc)
            #logger.add('{}_roc_auc'.format(eval_name), step, roc_auc)
            logger.add('{}_vram_alloc'.format(eval_name), step, vram_alloc)
            logger.add('{}_max_vram_alloc'.format(eval_name), step, max_vram_alloc)
            logger.add('{}_n_params'.format(eval_name), step, n_params)

            if verbose:
                nonlocal time_of_last_call
                nonlocal total_time
                elapsed = 0 if time_of_last_call is None else time.time() - time_of_last_call
                print('{}\tep {:03d}\tit {:03d}\tloss {:.3f}\tacc {:.2f}%\tdice {:.3f}\tex {:d}\ttime {:.2f}s'.format(
                    eval_name, step.ep, step.it, total_loss/example_count, 100 * total_correct/example_count,
                    total_dice/example_count, int(example_count), elapsed))
                time_of_last_call = time.time()
                total_time = 0 if total_time is None else total_time + elapsed
                logger.add('{}_total_time'.format(eval_name), step, total_time)

    return eval_callback


# Callback frequencies. Each takes a callback as an argument and returns a new callback
# that runs only at the specified frequency.
def run_every_epoch(callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step.it != 0:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


def run_every_step(callback):
    return callback


def run_at_step(step1, callback):
    def modified_callback(output_location, step, model, optimizer, logger):
        if step != step1:
            return
        callback(output_location, step, model, optimizer, logger)
    return modified_callback


# The standard set of callbacks that should be used for a normal training run.
def standard_callbacks(training_hparams: hparams.TrainingHparams, train_set_loader: DataLoader,
                       test_set_loader: DataLoader, eval_on_train: bool = False, verbose: bool = True,
                       start_step: Step = None, evaluate_every_epoch: bool = True):
    start = start_step or Step.zero(train_set_loader.iterations_per_epoch)
    end = Step.from_str(training_hparams.training_steps, train_set_loader.iterations_per_epoch)
    test_eval_callback = create_eval_callback('test', test_set_loader, verbose=verbose)
    train_eval_callback = create_eval_callback('train', train_set_loader, verbose=verbose)

    # Basic checkpointing and state saving at the beginning and end.
    result = [
        run_at_step(start, save_model),
        run_at_step(end, save_model),
        run_at_step(end, save_logger),
        run_every_epoch(checkpointing.save_checkpoint_callback),
    ]

    # Test every epoch if requested.
    if evaluate_every_epoch: result = [run_every_epoch(test_eval_callback)] + result
    elif verbose: result.append(run_every_epoch(create_timekeeper_callback()))

    # Ensure that testing occurs at least at the beginning and end of training.
    if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, test_eval_callback)] + result
    if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, test_eval_callback)] + result

    # Do the same for the train set if requested.
    if eval_on_train:
        if evaluate_every_epoch: result = [run_every_epoch(train_eval_callback)] + result
        if start.it != 0 or not evaluate_every_epoch: result = [run_at_step(start, train_eval_callback)] + result
        if end.it != 0 or not evaluate_every_epoch: result = [run_at_step(end, train_eval_callback)] + result

    return result
