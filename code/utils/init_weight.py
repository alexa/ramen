# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import torch
import argparse
import torch.nn as nn
from transformers import BertTokenizer, RobertaTokenizer
from tokenizers import BertWordPieceTokenizer
import numpy as np
parser = argparse.ArgumentParser('generate target embeddings from alignments')
parser.add_argument('--tgt_vocab', default='', help='target vocabulary file')
parser.add_argument('--src_vocab', default='', help='path to source vocabulary')
parser.add_argument('--src_merge', default='', help='path to source merge file')
parser.add_argument('--src_model', default='pytorch.bin', help='source pre-trained file')
parser.add_argument('--prob', default='', help='word translation probability')
parser.add_argument('--tgt_model', default='', help='save the target model')
params = parser.parse_args()
print(params)

if 'roberta' in params.src_model:
    CLS_TOKEN, CLS_INDEX = '<s>', 0
    SEP_TOKEN, SEP_INDEX = '</s>', 2
    UNK_TOKEN, UNK_INDEX = '<unk>', 3
    PAD_TOKEN, PAD_INDEX = '<pad>', 1
    MASK_TOKEN, MASK_INDEX = '<mask>', 50264

    MAP = {
        'word_embeddings': 'roberta.embeddings.word_embeddings.weight',
        'output_weight': 'lm_head.decoder.weight',
        'output_bias': 'lm_head.bias'
    }

else:
    CLS_TOKEN, CLS_INDEX = "[CLS]", 101
    SEP_TOKEN, SEP_INDEX = "[SEP]", 102
    UNK_TOKEN, UNK_INDEX = "[UNK]", 100
    PAD_TOKEN, PAD_INDEX = "[PAD]", 0
    MASK_TOKEN, MASK_INDEX = "[MASK]", 102

    MAP = {
        'word_embeddings': 'bert.embeddings.word_embeddings.weight',
        'output_weight': 'cls.predictions.decoder.weight',
        'output_bias': 'cls.predictions.bias'
    }


def guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=None):
    emb_dim = src_embs.size(1)
    num_tgt = tgt_tokenizer.get_vocab_size()

    # init with zero
    tgt_embs = src_embs.new_empty(num_tgt, emb_dim)
    tgt_bias = src_bias.new_zeros(num_tgt)
    nn.init.normal_(tgt_embs, mean=0, std=emb_dim ** -0.5)

    # copy over embeddings of special words
    for i in src_tokenizer.all_special_ids:
        tgt_embs[i] = src_embs[i]
        tgt_bias[i] = src_bias[i]

    # initialize randomly
    if prob is None:
        print('| INITIALIZE EMBEDDINGS AND BIAS RANDOMLY')
        return tgt_embs, tgt_bias


    num_src_per_tgt = np.array([len(x) for x in prob.values()]).mean()
    print(f'| # aligned src / tgt: {num_src_per_tgt:.5}')

    for t, ws in prob.items():
        if not tgt_tokenizer.token_to_id(t): continue

        px, ix = [], []
        for e, p in ws.items():
            # get index of the source word e
            j = src_tokenizer.convert_tokens_to_ids(e)
            ix.append(j)
            px.append(p)
        px = torch.tensor(px).to(src_embs.device)
        # get index of target word t
        ti = tgt_tokenizer.token_to_id(t)
        tgt_embs[ti] = px @ src_embs[ix]
        tgt_bias[ti] = px.dot(src_bias[ix])

    return tgt_embs, tgt_bias


def init_tgt(params):
    """
    Initialize the parameters of the target model
    """
    prob = None
    if params.prob:
        print(' | load word translation probs!')
        prob = torch.load(params.prob)

    print(f'| load English pre-trained model: {params.src_model}')
    model = torch.load(params.src_model)
    if 'roberta' in params.src_model:
        assert params.src_merge, "merge file should be provided!"
        src_tokenizer = RobertaTokenizer(params.src_vocab, params.src_merge)
    else:
        # note that we do not lowercase here
        src_tokenizer = BertTokenizer(params.src_vocab, do_lower_case=False)

    # get English word-embeddings and bias
    src_embs = model[MAP['word_embeddings']]
    src_bias = model[MAP['output_bias']]

    # initialize target tokenizer, we always use BertWordPieceTokenizer for the target language
    tgt_tokenizer = BertWordPieceTokenizer(
        params.tgt_vocab,
        unk_token=UNK_TOKEN,
        sep_token=SEP_TOKEN,
        cls_token=CLS_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
        lowercase=False,
        strip_accents=False
    )

    tgt_embs, tgt_bias = guess(src_embs, src_bias, tgt_tokenizer, src_tokenizer, prob=prob)

    # checksum for debugging purpose
    print(' checksum src | embeddings {:.5f} - bias {:.5f}'.format(
        src_embs.norm().item(), src_bias.norm().item()))
    model[MAP['word_embeddings']] = tgt_embs
    model[MAP['output_bias']] = tgt_bias
    model[MAP['output_weight']] = model[MAP['word_embeddings']]
    print(' checksum tgt | embeddings {:.5f} - bias {:.5f}'.format(
        model[MAP['word_embeddings']].norm().item(),
        model[MAP['output_bias']].norm().item()))

    # save the model
    torch.save(model, params.tgt_model)


if __name__ == '__main__':
    init_tgt(params)
