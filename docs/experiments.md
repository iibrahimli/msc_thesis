# Experiments


## Experiment 1

Aims to aims to reproduce Figure 22 (a) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Trains NanoGPT, vanilla Transformer, and a Universal Transformer on addition task. Training set consists of examples involving operands with 1 and 3 digits, and the test set is 100 examples each of 1, 2, 3, and 4 digit examples. All 3 models generalize to the same number of digits as training (1 and 3) and fail on 2 and 4 digit examples. Examples are padded with a `$` symbol in the beginning and end, and the answers are reversed. 

> Note: Since NanoGPT is a decoder-only model, and the others are encoder-decoder models the data is fed a bit differently: NanoGPT simply gets a continuous sequence of `SEQ_LENGTH` concatenated example lines, while enc-dec models get padded source and target sequences (each much shorter than `SEQ_LENGTH`).


## Experiment 2

Aims to aims to reproduce Figure 22 (b) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Same 3 models as Experiment 1 are trained on 1-7 digit addition and tested on 8 digit addition. The examples are plain formatted (i.e. answers are not reversed). Other details are the same as Experiment 1.