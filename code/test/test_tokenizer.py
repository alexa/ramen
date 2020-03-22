# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


# We use tokenizer from transformers and tokenizers exchangeably
# it is important to ensure the correctness of both tokenizers
import unittest
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer


class TestExchangeability(unittest.TestCase):

    def test_en(self):
        tfm = BertTokenizer('data/bert-base-cased-vocab.txt', do_lower_case=False)
        tok = BertWordPieceTokenizer('data/bert-base-cased-vocab.txt', lowercase=False, strip_accents=False)
        with open('data/toy.en', 'r') as f:
            for line in f:
                enc1 = tok.encode(line)
                enc2 = tfm.encode(line)
                self.assertListEqual(enc1.ids, enc2)

    def test_vi(self):
        tfm = BertTokenizer('data/vi-vocab.txt', do_lower_case=False)
        tok = BertWordPieceTokenizer('data/vi-vocab.txt', lowercase=False, strip_accents=False)
        with open('data/toy.vi', 'r') as f:
            for line in f:
                enc1 = tok.encode(line)
                enc2 = tfm.encode(line)
                self.assertListEqual(enc1.ids, enc2)


if __name__ == '__main__':
    unittest.main()
