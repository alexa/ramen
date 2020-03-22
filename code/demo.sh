#! /usr/bin/env bash

set -x;

lg=ar
V=32000
DATA=../data
MONO_PATH=$DATA/mono
VOCAB_PATH=$DATA/bert/vocab
ALIGN_PATH=$DATA/bert/align
PARA_PATH=$DATA/para
PRETRAINED_PATH=$DATA/bert/bert-base/
TGT_INIT=../checkpoints/$lg/bert-base
EXP_PATH=../checkpoints/$lg/bert_base_para
PROB_PATH=$DATA/bert/probs

mkdir -p $PROB_PATH
mkdir -p $VOCAB_PATH
mkdir -p $ALIGN_PATH
mkdir -p $EXP_PATH
mkdir -p $TGT_INIT


# install fast_align
if [ ! -d "tools" ]
then
  mkdir tools
  cd tools
  git clone https://github.com/clab/fast_align.git
  cd fast_align
  mkdir build
  cd build
  cmake ..
  make
  cd ../..
fi

# prepare vocab
echo "preparing vocabulary for $lg";
shuf $MONO_PATH/$lg.all | head -2000000 > $MONO_PATH/$lg.2M
python utils/learn_vocab.py --input $MONO_PATH/$lg.2M --out_dir $VOCAB_PATH --lg $lg --vocab_size $V --model bert
rm $MONO_PATH/$lg.2M

echo "preparing parallel data for fast_align"
python utils/tokenizer.py --vocab $VOCAB_PATH/$lg-vocab.txt --input $PARA_PATH/en-$lg.$lg --output $ALIGN_PATH/en-$lg.$lg
python utils/tokenizer.py --vocab $VOCAB_PATH/bert-base-cased-vocab.txt --input $PARA_PATH/en-$lg.en --output $ALIGN_PATH/en-$lg.en

:|paste -d ' ||| ' $ALIGN_PATH/en-$lg.en - - - - $ALIGN_PATH/en-$lg.$lg > $ALIGN_PATH/text.en-$lg
rm $ALIGN_PATH/en-$lg.en
rm $ALIGN_PATH/en-$lg.$lg

tools/fast_align/build/fast_align -i $ALIGN_PATH/text.en-$lg -d -o -v -I 10 > $ALIGN_PATH/forward.en-$lg
tools/fast_align/build/fast_align -i $ALIGN_PATH/text.en-$lg -d -o -v -r -I 10 > $ALIGN_PATH/reverse.en-$lg
tools/fast_align/build/atools -i $ALIGN_PATH/forward.en-$lg -j $ALIGN_PATH/reverse.en-$lg -c grow-diag-final-and > $ALIGN_PATH/align.en-$lg

echo "estimate word translation probabilities"
python alignment/get_prob_para.py --bitxt $ALIGN_PATH/text.en-$lg --align $ALIGN_PATH/align.en-$lg --save $PROB_PATH/probs.para.en-$lg.pth

echo "initialize target model"
python utils/init_weight.py --src_vocab $VOCAB_PATH/bert-base-cased-vocab.txt --src_model $PRETRAINED_PATH/pytorch_model.bin --prob $ALIGN_PATH/para.en-$lg.pth --tgt_model $TGT_INIT/pytorch_model.bin --tgt_vocab $VOCAB_PATH/$lg-vocab.txt
# copy config file
cp $PRETRAINED_PATH/bert-base-cased-config.json $TGT_INIT/config.json
