# Experiments


## Experiment 1

Aims to aims to reproduce Figure 22 (a) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Trains NanoGPT, vanilla Transformer, and a Universal Transformer on addition task. Training set consists of examples involving operands with 1 and 3 digits, and the test set is 100 examples each of 1, 2, 3, and 4 digit examples. All 3 models generalize to the same number of digits as training (1 and 3) and fail on 2 and 4 digit examples. Examples are padded with a `$` symbol in the beginning and end, and the answers are reversed. 

> Note: Since NanoGPT is a decoder-only model, and the others are encoder-decoder models the data is fed a bit differently: NanoGPT simply gets a continuous sequence of `SEQ_LENGTH` concatenated example lines, while enc-dec models get padded source and target sequences (each much shorter than `SEQ_LENGTH`).


## Experiment 2

Aims to aims to reproduce Figure 22 (b) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Same 3 models as Experiment 1 are trained on 1-7 digit addition and tested on 8 digit addition. The examples are plain formatted (i.e. answers are not reversed). Other details are the same as Experiment 1.


## Experiment 3

Go simpler: train the models on 3x3 digit addition on ~1M samples. No out of distribution test set. Just see if they can perfect adding 3 digit numbers.


## Experiment 4

As Experiment 3, but with 7x7 digit addition.


## Experiment 5

Same as experiment 3, except the answers are padded with a leading `0`, so that all answers are 4 digits long (still reversed).


## Experiment 6

As Experiment 5, but with 7x7 digit addition.


## Experiment 7

Like Experiment 2, but with operands zero-padded to 8 digits, answers padded to 9 digits, and not reversing the answers for enc-dec models.


## Experiment 8

Training set: 1-9 digit addition EXCEPT 8 digit addition. Test on 1-10 digit addition. Padding operands to 10 digits, answers to 11 digits, and not reversing the answers for enc-dec models.