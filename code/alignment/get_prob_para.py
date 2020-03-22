# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
from collections import defaultdict, Counter
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser('Generate target embeddings from alignments')
parser.add_argument('--bitxt', default='', help='bi-text bpe file')
parser.add_argument('--align', default='', help='alignment file')
parser.add_argument('--save', default='', help='path to save file')
args = parser.parse_args()
print(args)


def main(args):
    print('| collecting counts')
    count = defaultdict(Counter)
    bitxt = open(args.bitxt, 'r').readlines()
    align = open(args.align, 'r').readlines()
    assert len(bitxt) == len(align)

    for line, a in zip(tqdm(bitxt), align):
        langs = line.strip().split(' ||| ')
        if len(langs) != 2: continue

        a = [tuple(map(int, x.split('-'))) for x in a.split()]
        src_toks = langs[0].split()
        tgt_toks = langs[1].split()

        for (sid, tid) in a:
            if sid >= len(src_toks) or tid >= len(tgt_toks):
                continue
            e, f = src_toks[sid], tgt_toks[tid]
            count[f][e] += 1

    # re-normalize counts to get translation probability
    print('| re-normalizing counts')
    for f, es in tqdm(count.items()):
        z = sum(es.values())
        for e, k in es.items():
            count[f][e] = k/z
    torch.save(count, args.save)


if __name__ == '__main__':
    main(args)
