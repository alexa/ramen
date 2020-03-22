# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# -*- coding: utf-8 -*-
from tokenizers import BertWordPieceTokenizer
import numpy as np
import argparse
import torch

parser = argparse.ArgumentParser(description="Numerize text into numpy format")
parser.add_argument('--vocab', default='path/to/vocab.txt',
                    help="path to vocabulary file")
parser.add_argument('--merge', default='',
                    help="path to merge file")
parser.add_argument('--input', default='en.train',
                    help="path to tokenized text file")
parser.add_argument('--bin_path', default='en.train.pth',
                    help="path to binary file output")
parser.add_argument("--model", choices=["bert", "roberta"], default="bert",
                    help="source pre-trained model")
params = parser.parse_args()



if params.model == 'bert':
    CLS_TOKEN, CLS_INDEX = "[CLS]", 101
    SEP_TOKEN, SEP_INDEX = "[SEP]", 102
    UNK_TOKEN, UNK_INDEX = "[UNK]", 100
    PAD_TOKEN, PAD_INDEX = "[PAD]", 0
    MASK_TOKEN, MASK_INDEX = "[MASK]", 103
elif params.model == 'roberta':
    CLS_TOKEN, CLS_INDEX = '<s>', 0
    SEP_TOKEN, SEP_INDEX = '</s>', 2
    UNK_TOKEN, UNK_INDEX = '<unk>', 3
    PAD_TOKEN, PAD_INDEX = '<pad>', 1
    MASK_TOKEN, MASK_INDEX = '<mask>', 50264


def numerize(vocab_path, input_path, bin_path):
    tokenizer = BertWordPieceTokenizer(
        vocab_path,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        cls_token=CLS_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        lowercase=False,
        strip_accents=False)
    sentences = []
    with open(input_path, 'r') as f:
        batch_stream = []
        for i, line in enumerate(f):
            batch_stream.append(line)
            if i % 1000 == 0:
                res = tokenizer.encode_batch(batch_stream)
                batch_stream = []
                # flatten the list
                for s in res:
                    sentences.extend(s.ids[1:])
            if i % 100000 == 0:
                print(f'processed {i} lines')
    
    print('convert the data to numpy')

    # convert data to numpy format in uint16
    if tokenizer.get_vocab_size() < 1 << 16:
        sentences = np.uint16(sentences)
    else:
        assert tokenizer.get_vocab_size() < 1 << 31
        sentences = np.int32(sentences)

    # save special tokens for later processing
    sep_index = tokenizer.token_to_id(SEP_TOKEN)
    cls_index = tokenizer.token_to_id(CLS_TOKEN)
    unk_index = tokenizer.token_to_id(UNK_TOKEN)
    mask_index = tokenizer.token_to_id(MASK_TOKEN)
    pad_index = tokenizer.token_to_id(PAD_TOKEN)

    # sanity check
    assert sep_index == SEP_INDEX
    assert cls_index == CLS_INDEX
    assert unk_index == UNK_INDEX
    assert pad_index == PAD_INDEX
    assert mask_index == MASK_INDEX

    print('collect statistics')
    # collect some statistics of the dataset
    n_unks = (sentences == unk_index).sum()
    n_toks = len(sentences)
    p_unks = n_unks * 100. / n_toks
    n_seqs = (sentences == sep_index).sum()
    print(f'| {n_seqs} sentences - {n_toks} tokens - {p_unks:.2f}% unknown words')

    # print some statistics
    data = {'sentences': sentences,
            'sep_index': sep_index,
            'cls_index': cls_index,
            'unk_index': unk_index,
            'pad_index': pad_index,
            'mask_index': mask_index}

    torch.save(data, bin_path, pickle_protocol=4)


numerize(params.vocab, params.input, params.bin_path)
