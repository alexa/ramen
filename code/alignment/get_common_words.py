# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
from sys import argv

def wordlist(fname):
    words = [x.rstrip() for x in open(fname, 'r').readlines()]
    return words[1:]

words1 = wordlist(argv[1])
words2 = wordlist(argv[2])
common = set(words1) & set(words2)

# for robustness, we only use top 5000 words
candidates = []
for w in words1:
    if w in common:
        candidates.append(w)
    if len(candidates) > 5000:
        break

for w in common:
    print(f"{w} {w}")
