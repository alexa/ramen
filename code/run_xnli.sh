#! /usr/bin/env bash

set -x;
lg=vi
ep=ep1
gpuid=7
export CUDA_VISIBLE_DEVICES=$gpuid; python xnli.py --batch_size 32 --pretrained_path ../checkpoints/$lg/bert_base_mono/bert_en_$ep --src_lang en --exp_path ../checkpoints/$lg/bert_base_mono --data_path ../data/bert/xnli/ --max_epoch 5 --epoch_size 5000 --lr 0.00001 --grad_acc_steps 2 --xnli_model ../checkpoints/$lg/bert_base_mono/xnli.en.pth

export CUDA_VISIBLE_DEVICES=$gpuid; python utils/copy_weight.py --src ../checkpoints/$lg/bert_base_mono/xnli.en.pth --f ../checkpoints/$lg/bert_base_mono/bert_${lg}_${ep}/pytorch_model.bin --tgt ../checkpoints/$lg/bert_base_mono/xnli.$lg.pth

export CUDA_VISIBLE_DEVICES=$gpuid; python xnli.py --pretrained_path ../checkpoints/$lg/bert_base_mono/bert_${lg}_${ep} --xnli_model ../checkpoints/$lg/bert_base_mono/xnli.$lg.pth --data_path ../data/bert/xnli --tgt_lang $lg
