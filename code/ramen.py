# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
import math
import torch
from src.models import RobertaMLM, RobertaAdaptor, BertMLM, BertAdaptor
from src.data.loader import DataIterator
import numpy as np
from src.logger import init_logger
from opt import get_parser
import time
from src.optim import AdamInverseSquareRootWithWarmup
from apex import amp, optimizers


# util for printing time
def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs


def get_model(params):
    """return pre-trained model, adaptor, and mask_index"""
    model_path = params.src_pretrained_path
    assert 'roberta' in model_path or 'bert' in model_path

    if 'roberta' in model_path:
        mask_index = 50264
        return RobertaMLM, RobertaAdaptor, mask_index
    else:
        mask_index = 103
        return BertMLM, BertAdaptor, mask_index


def mask_input(x, word_pred, mask_index):
    """
    mask the input with a certain probability
    :param x: a tensor of size (bsize, slen)
    :param word_pred: float type, indicate the percentage of masked words
    :param mask_index: int, masking index
    :return: a masked input and the original input
    """
    bsize, slen = x.size()
    npred = math.ceil(bsize * slen * word_pred)
    # make it a multiplication of 8
    npred = (npred // 8) * 8
    # masked words to predict
    y_idx = np.random.choice(bsize * slen, npred, replace=False)
    # keep some identity words
    i_idx = np.random.choice(npred, int(0.10*npred), replace=False)
    mask = torch.zeros(slen * bsize, dtype=torch.long)
    mask[y_idx] = 1
    # identity (i.e copy)
    mask_ = mask.clone()
    mask_[y_idx[i_idx]] = 0

    mask = mask.view(bsize, slen)
    # do not predict CLS
    mask[:, 0] = 0
    y = mask * x.clone() + (mask - 1)
    # process x
    mask_ = mask_.view(bsize, slen)
    mask_[:, 0] = 0
    x.masked_fill_(mask_ == 1, mask_index)   # mask_index

    return x.cuda(), y.cuda()


def main():
    parser = get_parser()
    params = parser.parse_args()

    if not os.path.exists(params.exp_path):
        os.makedirs(params.exp_path)
    # some short-cut
    src_lg = params.src_lang
    tgt_lg = params.tgt_lang

    log_file = os.path.join(params.exp_path, f'ramen_{src_lg}-{tgt_lg}.log')

    logger = init_logger(log_file)
    logger.info(params)

    pretrained_model, adaptor, mask_index = get_model(params)
    src_model = pretrained_model.from_pretrained(params.src_pretrained_path)
    tgt_model = pretrained_model.from_pretrained(params.tgt_pretrained_path)
    model = adaptor(src_model, tgt_model)
    model = model.cuda()

    optimizer = AdamInverseSquareRootWithWarmup(model.parameters(), lr=params.lr, warmup_updates=4000)
    model, optimizer = amp.initialize(model, optimizer, opt_level=params.opt_level)

    data = DataIterator(params)

    train_loader_fr = data.get_iter(tgt_lg, 'train')
    train_loader_en = data.get_iter(src_lg, 'train')

    valid_loader_fr = data.get_iter(tgt_lg, 'valid')
    valid_loader_en = data.get_iter(src_lg, 'valid')

    def evaluate(lang, loader):
        # useful for evaluation
        model.eval()
        losses = []
        for x in loader:
            x, y = mask_input(x.squeeze(0), params.word_pred, mask_index)
            with torch.no_grad():
                loss = model(lang, x, masked_lm_labels=y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def step(lg, x, update=True):
        x, y = mask_input(x.squeeze(0), params.word_pred, mask_index)
        loss = model(lg, x, masked_lm_labels=y)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if update:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5)
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()

    n_iter = 0
    n_epoch = 0
    start_time = time.time()
    best_valid_nll = 1e8
    model_prefix = 'roberta' if 'roberta' in params.src_pretrained_path else 'bert'

    while True:
        for batch_en, batch_fr in zip(train_loader_en, train_loader_fr):
            n_iter += 1
            loss_en = step(src_lg, batch_en, False)
            loss_fr = step(tgt_lg, batch_fr, n_iter % params.grad_acc_steps == 0)

            if n_iter % 50 == 0:
                time_taken = time.time() - start_time
                hours, mins, secs = _to_hours_mins_secs(time_taken)

                cur_lr = optimizer.get_lr()
                logger.info(
                    f" Iter {n_iter:>7} - MLM-{src_lg} {loss_en:.4f} -"
                    f" MLM-{tgt_lg} {loss_fr:.4f} - lr {cur_lr:.7f}"
                    f" elapsed {int(hours)}:{int(mins)}"
                )

            if n_iter % params.epoch_size == 0:
                n_epoch += 1
                valid_src_nll = evaluate(src_lg, valid_loader_en)
                valid_tgt_nll = evaluate(tgt_lg, valid_loader_fr)
                logger.info(
                    f" Validation - Iter {n_iter} |"
                    f" MLM-{src_lg} {valid_src_nll:.4f} MLM-{tgt_lg} {valid_tgt_nll:.4f}"
                )

                avg_nll = (valid_src_nll + valid_tgt_nll) / 2
                if avg_nll < best_valid_nll:
                    best_valid_nll = avg_nll
                    logger.info(f"| Best checkpoint at epoch: {n_epoch}")

                src_model = f'{model_prefix}_{src_lg}_ep{n_epoch}'
                tgt_model = f'{model_prefix}_{tgt_lg}_ep{n_epoch}'
                src_path = os.path.join(params.exp_path, src_model)
                tgt_path = os.path.join(params.exp_path, tgt_model)

                if not os.path.exists(src_path): os.makedirs(src_path)
                if not os.path.exists(tgt_path): os.makedirs(tgt_path)

                # save both models
                logger.info(f'save ({src_lg}) model to: {src_path}')
                model.src_model.save_pretrained(src_path)
                logger.info(f'save ({tgt_lg}) model to: {tgt_path}')
                model.tgt_model.save_pretrained(tgt_path)

                if n_epoch == params.max_epoch:
                    exit()


if __name__ == '__main__':
    main()
