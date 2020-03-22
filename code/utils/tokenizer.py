# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# -*- coding: utf-8 -*-
from tokenizers import BertWordPieceTokenizer
import argparse


parser = argparse.ArgumentParser(description="Tokenize data")
parser.add_argument('--vocab', default='path/to/vocab.txt',
                    help="path to vocabulary file")
parser.add_argument('--merge', default='',
                    help="path to merge file")
parser.add_argument('--input', default='en.train',
                    help="path to tokenized text file")
parser.add_argument('--output', default='en.train.tok',
                    help="path to tokenized output")
parser.add_argument("--model", choices=["bert", "roberta"], default="bert",
                    help="source pre-trained model")
params = parser.parse_args()


def chunks(lst, n=1000):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main():
    if params.model == 'bert':
        CLS_TOKEN = '[CLS]'
        SEP_TOKEN = "[SEP]"
        UNK_TOKEN = "[UNK]"
        PAD_TOKEN = "[PAD]"
        MASK_TOKEN = "[MASK]"
    elif params.model == 'roberta':
        CLS_TOKEN = '<s>'
        SEP_TOKEN = '</s>'
        UNK_TOKEN = '<unk>'
        PAD_TOKEN = '<pad>'
        MASK_TOKEN = '<mask>'

    tokenizer = BertWordPieceTokenizer(
        params.vocab,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        cls_token=CLS_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        lowercase=False,
        strip_accents=False)

    with open(params.input, 'r') as reader, open(params.output, 'w') as writer:
        sentences = reader.readlines()
        for batch in chunks(sentences, 1000):
            encoded = tokenizer.encode_batch(batch)
            sentences = map(lambda x: ' '.join(x.tokens[1:-1]), encoded)
            writer.write('\n'.join(sentences) + '\n')


if __name__ == '__main__':
    main()



