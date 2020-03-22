# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import argparse
from tokenizers import BertWordPieceTokenizer
import os

parser = argparse.ArgumentParser(description="Learn BPE vocabulary")
parser.add_argument('--input', type=str,
                    help='input text file')
parser.add_argument('--out_dir', type=str, default='bert_vocab',
                    help='output directory.')
parser.add_argument('--lg', type=str, default='vi',
                    help='target language.')
parser.add_argument('--vocab_size', type=int, default=30000,
                    help='vocabulary size')
parser.add_argument("--model", choices=["bert", "roberta"],
                    default="bert", type=str,
                    help="source pre-trained model")
params = parser.parse_args()

if not os.path.exists(params.out_dir):
    os.makedirs(params.out_dir)


# set special tokens and their indices
# if transferring from other model,
# these values need to be set differently depending on each model's tokenizer
# note that here we use word-piece tokenizer for the target language.

assert params.model in ['bert', 'roberta']
if params.model == 'bert':
    CLS_TOKEN, CLS_INDEX = '[CLS]', 101
    SEP_TOKEN, SEP_INDEX = '[SEP]', 102
    UNK_TOKEN, UNK_INDEX = '[UNK]', 100
    PAD_TOKEN, PAD_INDEX = '[PAD]', 0
    MASK_TOKEN, MASK_INDEX = '[MASK]', 103
else:
    CLS_TOKEN, CLS_INDEX = '<s>', 0
    SEP_TOKEN, SEP_INDEX = '</s>', 2
    UNK_TOKEN, UNK_INDEX = '<unk>', 3
    PAD_TOKEN, PAD_INDEX = '<pad>', 1
    MASK_TOKEN, MASK_INDEX = '<mask>', 50264

special_word2id = {
    CLS_TOKEN: CLS_INDEX,
    SEP_TOKEN: SEP_INDEX,
    UNK_TOKEN: UNK_INDEX,
    PAD_TOKEN: PAD_INDEX,
    MASK_TOKEN: MASK_INDEX
}

special_id2word = {i: w for w, i in special_word2id.items()}

tokenizer = BertWordPieceTokenizer(
    unk_token=UNK_TOKEN,
    sep_token=SEP_TOKEN,
    cls_token=CLS_TOKEN,
    pad_token=PAD_TOKEN,
    mask_token=MASK_TOKEN,
    strip_accents=False,
    lowercase=False
)

# make sure that the vocab is large enough to cover special indices
assert params.vocab_size > max(CLS_INDEX, SEP_INDEX, UNK_INDEX, PAD_INDEX, MASK_INDEX)
tokenizer.train(params.input, vocab_size=params.vocab_size, min_frequency=5)
tokenizer.save(params.out_dir, params.lg)

# insert special words to the correct position
vocab_file = os.path.join(params.out_dir, params.lg + '-vocab.txt')
with open(vocab_file, 'r') as f:
    words = [w.rstrip() for w in f]
    new_words = words[len(special_word2id):]
    for i in sorted(special_id2word.keys()):
        new_words.insert(i, special_id2word[i])

# overwrite the vocab file
with open(vocab_file, 'w') as f:
    for w in new_words:
        f.write(w + '\n')



