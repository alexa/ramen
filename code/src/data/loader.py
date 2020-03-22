# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from __future__ import absolute_import
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MonoDataset(Dataset):
    """Monolingual DataSet: it will be a stream of text."""
    def __init__(self, bin_path, params):
        _data = torch.load(bin_path)
        sentences = _data['sentences']

        bptt = params.bptt - 1  # reserved 1 element for cls
        batch_size = params.batch_size
        self.cls_index = _data['cls_index']
        self.sep_index = _data['sep_index']
        # calculate number of sentences
        self.n_sents = (sentences == self.sep_index).sum()

        # when we have many samples, ignore some is forgiven
        n_batches = len(sentences) // (batch_size * bptt)
        taken_size = n_batches * bptt * batch_size
        self.data = sentences[:taken_size].reshape((batch_size, n_batches * bptt))

        self.bptt = bptt
        self.n_batches = n_batches
        # for padding
        self.cls_pad = torch.LongTensor([[self.cls_index]] * batch_size)

    def num_sentences(self):
        return self.n_sents

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        a = self.bptt * idx
        b = self.bptt * (idx + 1)
        x = torch.from_numpy(self.data[:, a:b].astype(np.int64))
        return torch.cat([self.cls_pad, x], 1)


class DataIterator(object):
    """
    Data Iterator for training
    """
    def __init__(self, params):
        data = load_data(params)
        self.langs = list(data.keys())
        self.iterators = {}
        for lang in data.keys():
            self.iterators[lang] = {}
            for splt, ds in data[lang].items():
                iterator = DataLoader(ds, num_workers=4, batch_size=1,
                                      shuffle=('train' == splt))
                self.iterators[lang][splt] = iterator

    def get_iter(self, lang, splt):
        return self.iterators[lang][splt]


def load_data(params):
    """ Load monolingual data
    """
    data = {}
    tgt = params.tgt_lang
    src = params.src_lang
    languages = [tgt] if not src else [src, tgt]

    for lang in languages:
        print(f'Loading ({lang})')

        data[lang] = {}
        for splt in ['train', 'valid', 'test']:
            # load data / update dictionary parameters / update data
            path = os.path.join(params.data_path, f"{lang}.{splt}.pth")
            data[lang][splt] = MonoDataset(path, params)

    print('Data Statistics:\n')
    print('-' * 19)
    for lang, v in data.items():
        for splt in v.keys():
            print(f"{lang:>2} {splt:>5} {v[splt].num_sentences():>10}")
        print('-'*19)
    return data
