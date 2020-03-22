# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="RAMEN-LM")
    parser.add_argument("--exp_path", type=str, default="../experiments",
                        help="path to store experiments dir model")
    parser.add_argument("--src_pretrained_path", type=str, default="",
                        help="path to [bert|xlnet] pretrained dir model")
    parser.add_argument("--tgt_pretrained_path", type=str, default="",
                        help="path to [bert|xlnet] pretrained dir model")
    parser.add_argument("--pretrained_path", type=str, default="",
                        help="path to [bert|roberta|xlnet] pretrained dir model")
    parser.add_argument("--xnli_model", type=str, default="",
                        help="path to trained xnli model")

    parser.add_argument("--ud_model", type=str, default="",
                        help="path to trained parser")
    # masked LM params
    parser.add_argument("--bptt", type=int, default=256,
                        help="number of tokens in a sentence.")
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="% of masked words")
    # batch parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of sentences per batch")

    parser.add_argument("--fp16", action='store_true',
                        help="use float16 for training")
    parser.add_argument("--opt_level", type=str, default='O2',
                        choices=['O1', 'O2'],
                        help='optimization level')
    # data
    parser.add_argument("--data_path", type=str, default="../data/bert/binary",
                        help="Data path")
    parser.add_argument("--src_lang", type=str, default="",
                        help="source languages")
    parser.add_argument("--tgt_lang", type=str, default="",
                        help="target languages")

    parser.add_argument("--max_epoch", type=int, default=100,
                        help="max number of training epoch")
    parser.add_argument("--epoch_size", type=int, default=50000,
                        help="number of updates per epoch")
    parser.add_argument("--lr", type=float, default=0.000005,
                        help="learning rate of Adam")
    parser.add_argument("--optim", type=str, default='adam',
                        help="optimizer.")
    parser.add_argument("--grad_acc_steps", type=int, default=1,
                        help="gradient accumulation steps")
    parser.add_argument("--debug_train", action='store_true',
                        help='fast debugging model')
    parser.add_argument("--tokenizer", choices=["bert", "roberta"],
                        default="bert", type=str,
                        help="tokenizer, needed for special tokens")
    parser.add_argument("--src_model", choices=["bert", "roberta"],
                        default="bert", type=str,
                        help="source pre-trained model")
    return parser
