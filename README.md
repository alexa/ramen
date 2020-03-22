## RAMEN
A software for transferring pretrained English models to foreign languages. For technical details is provided in the paper [From English To Foreign Languages: Transferring Pre-trained Language Models](https://arxiv.org/abs/2002.07306)

If you use this code for your research, please cite
```
@article{trnke2020_ramen,
       author = {{Tran}, Ke},
        title = "{From English To Foreign Languages: Transferring Pre-trained Language Models}",
      journal = {arXiv e-prints},
         year = 2020,
        month = feb,
          eid = {arXiv:2002.07306},
        pages = {arXiv:2002.07306},
archivePrefix = {arXiv},
       eprint = {2002.07306},
 primaryClass = {cs.CL},
}

```
## Requirements
- [transformers](https://github.com/huggingface/transformers) with can be installed via `pip install transformers`
- [tokenizes](https://github.com/huggingface/tokenizers) via `pip install tokenizers`

## Data Processing

Data pre-processing consists of several steps.

### Tokenization
- learning BPE code for target languages
- I used [fastBPE](https://github.com/glample/fastBPE) in the experiments, but any tokenizers can be used. In `demo.sh` HuggingFace's [tokenizers](https://github.com/huggingface/tokenizers) is used.

```
$ python src/misc/tokenizer.py --input data/mono/vi.2M --model bert --vocab_size 32000
```


### Tensorize Data
```
$ python utils/numerize.py --vocab /path/to/vocab --input vi.train --bin_path vi.train.pth
```

### Estimating sub-word translation probabilities

#### With Parallel data

See `demo.sh` script

#### With fastText vectors

See `alignment/example_mono.sh`




### Training RAMEN
See `bert_base.sh` script

### Evaluation
Evaluation using XNLI task can be found in `run_xnli.sh` script

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
