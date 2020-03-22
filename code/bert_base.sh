#!/usr/bin/env bash
set -x;
lg=vi; gpuid=5

V=32000
DATA=../data
MONO_PATH=$DATA/mono
VOCAB_PATH=$DATA/bert/vocab
ALIGN_PATH=$DATA/bert/align
PARA_PATH=$DATA/para
PRETRAINED_PATH=$DATA/bert/bert-base/
TGT_INIT=../checkpoints/$lg/bert-base
EXP_PATH=../checkpoints/$lg/bert_base_mono
PROB_PATH=../data/bert/probs

N_UPDATES=20000
MAX_EPOCH=6
python utils/init_weight.py --src_vocab $VOCAB_PATH/bert-base-cased-vocab.txt --src_model $PRETRAINED_PATH/pytorch_model.bin --prob $PROB_PATH/probs.mono.en-$lg.pth --tgt_model $TGT_INIT/pytorch_model.bin --tgt_vocab $VOCAB_PATH/$lg-vocab.txt
# copy config file
cp $PRETRAINED_PATH/bert-base-cased-config.json $TGT_INIT/config.json
export CUDA_VISIBLE_DEVICES=${gpuid}; python ramen.py --lr 0.0001 --tgt_lang ${lg} --src_lang en --batch_size 72 --bptt 256 --src_pretrained_path $PRETRAINED_PATH --tgt_pretrained_path $TGT_INIT  --data_path $DATA/bert/binary --epoch_size $N_UPDATES --max_epoch $MAX_EPOCH --fp16 --exp_path ${EXP_PATH} --grad_acc_steps 1
