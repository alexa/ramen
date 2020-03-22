# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from transformers import RobertaTokenizer, BertTokenizer
import argparse
from torch.nn.utils.rnn import pad_sequence
import torch

parser = argparse.ArgumentParser(description="preparing data for downstream task")
parser.add_argument('--vocab', type=str, default='', help='path to vocab file')
parser.add_argument('--merge', type=str, default='', help='path to merge file')
parser.add_argument('--input', type=str, default='', help='path to input text')
parser.add_argument('--output', type=str, default='', help='path to tensor output files')

params = parser.parse_args()


def make_xnli_data(params):
    """
    read text file and tensorize
    text file input has the format "premise  hypothesis  label"
    """
    if params.merge:
        tokenizer = RobertaTokenizer(params.vocab, params.merge)
    else:
        tokenizer = BertTokenizer(params.vocab, do_lower_case=False)

    xs, ys = [], []
    labels = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    pad_index = tokenizer.pad_token_id
    with open(params.input, 'r') as f:
        next(f)
        for i, line in enumerate(f):
            cols = line.rstrip().split('\t')

            if len(cols) != 3: print(f"potential error at line {i}")

            enc1 = tokenizer.encode(cols[0])
            enc2 = tokenizer.encode(cols[1])
            xs.append(enc1 + enc2[1:-2])  # [CLS] p1 ... [SEP] h1 ...hn
            ys.append(labels[cols[2]])

        # convert data to tensor
        xs = [torch.LongTensor(s) for s in xs]
        xs = pad_sequence(xs, batch_first=True, padding_value=pad_index)

        ys = torch.LongTensor(ys)
    unk_index = tokenizer.unk_token_id
    n_unks = sum([(s == unk_index).sum().item() for s in xs])
    n_toks = sum([len(s) for s in xs])
    p_unks = n_unks * 100. / n_toks
    print(f"{n_toks} tokens - {p_unks:.2f}% unknown words")
    data = {'xs': xs, 'ys': ys, 'pad_index': pad_index}

    torch.save(data, params.output)


if __name__ == "__main__":
    make_xnli_data(params)
