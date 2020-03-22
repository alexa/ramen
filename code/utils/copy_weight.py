# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import torch
import argparse

parser = argparse.ArgumentParser(description="Copy weights")
parser.add_argument("--src", type=str, default="",
                    help="path to source model")
parser.add_argument("--tgt", type=str, default="",
                    help="path to target model")
parser.add_argument("--f", type=str, default="",
                    help="from trained target weights")
args = parser.parse_args()
print(args)

W_MAPS = {'roberta': ('encoder.embeddings.word_embeddings.weight',
                      'roberta.embeddings.word_embeddings.weight'),
          'bert': ('encoder.embeddings.word_embeddings.weight',
                   'bert.embeddings.word_embeddings.weight')}

# load tuned source model (tuned for XNLI or UD Parsing)
tuned_src = torch.load(args.src)
# load target pre-trained model
f_weights = torch.load(args.f)

model = 'roberta' if 'roberta' in args.src else 'bert'
m = W_MAPS[model]
emb_src = tuned_src[m[0]]
emb_tgt = f_weights[m[1]].to(emb_src.device).to(emb_src.dtype)
# copy over
tuned_src[m[0]] = emb_tgt.data

# save model
print('save the model to: {}'.format(args.tgt))
torch.save(tuned_src, args.tgt)
