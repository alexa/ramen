# runing example for en-vi
set -x
VEC_PATH=../../data/vecs
ALIGN=../tools/fastText/alignment/align.py
VOCAB_PATH=../../data/bert/vocab
PROB_PATH=../../data/bert/probs

mkdir -p $PROB_PATH
N=100000
for lg in en vi; do
  wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.${lg}.300.vec.gz -P $VEC_PATH
  gunzip $VEC_PATH/cc.${lg}.300.vec.gz
  # get wordlist
  head -$N $VEC_PATH/cc.$lg.300.vec > $VEC_PATH/cc.$lg.300.vec.$N
  cut -f1 -d ' ' $VEC_PATH/cc.$lg.300.vec.$N > $VEC_PATH/wordlist.$lg
done

lg=vi

# note that it's important to  have wordlist.en as the first argument
python get_common_words.py $VEC_PATH/wordlist.en $VEC_PATH/wordlist.$lg > $VEC_PATH/wordlist.en-$lg

python $ALIGN --src_emb $VEC_PATH/cc.$lg.300.vec --tgt_emb $VEC_PATH/cc.en.300.vec --dico_train $VEC_PATH/wordlist.en-${lg} --dico_test $VEC_PATH/wordlist.en-$lg --output $VEC_PATH/aligned.$lg.vec  --niter 10 --maxload $N

python get_prob_vect.py --src_aligned_vec $VEC_PATH/cc.en.300.vec --src_vocab $VOCAB_PATH/bert-base-cased-vocab.txt --topn 50000 --tgt_aligned_vec $VEC_PATH/aligned.$lg.vec --tgt_vocab $VOCAB_PATH/$lg-vocab.txt --save $PROB_PATH/probs.mono.en-$lg.pth
