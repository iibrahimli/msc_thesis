# Experiments

## [Experiment 1](../arithmetic_lm/conf/experiment/1)
Aims to aims to reproduce Figure 22 (a) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Trains NanoGPT, vanilla Transformer, and a Universal Transformer on addition task. Training set consists of examples involving operands with 1 and 3 digits, and the test set is 100 examples each of 1, 2, 3, and 4 digit examples. All 3 models generalize to the same number of digits as training (1 and 3) and fail on 2 and 4 digit examples. Examples are padded with a `$` symbol in the beginning and end, and the answers are reversed. 

> Note: Since NanoGPT is a decoder-only model, and the others are encoder-decoder models the data is fed a bit differently: NanoGPT simply gets a continuous sequence of `SEQ_LENGTH` concatenated example lines, while enc-dec models get padded source and target sequences (each much shorter than `SEQ_LENGTH`).

## [Experiment 2](../arithmetic_lm/conf/experiment/2)
Aims to aims to reproduce Figure 22 (b) from the [[Lee et al. 2023] Teaching Arithmetic to Small Transformers](https://arxiv.org/abs/2307.03381). Same 3 models as Experiment 1 are trained on 1-7 digit addition and tested on 8 digit addition. The examples are plain formatted (i.e. answers are not reversed). Other details are the same as Experiment 1.

## [Experiment 3](../arithmetic_lm/conf/experiment/3)
Go simpler: train the models on 3x3 digit addition on ~1M samples. No out of distribution test set. Just see if they can perfect adding 3 digit numbers.

## [Experiment 4](../arithmetic_lm/conf/experiment/4)
As Experiment 3, but with 7x7 digit addition.

## [Experiment 5](../arithmetic_lm/conf/experiment/5)
Same as experiment 3, except the answers are padded with a leading `0`, so that all answers are 4 digits long (still reversed).

## [Experiment 6](../arithmetic_lm/conf/experiment/6)
As Experiment 5, but with 7x7 digit addition.

## [Experiment 7](../arithmetic_lm/conf/experiment/7)
Like Experiment 2, but with operands zero-padded to 8 digits, answers padded to 9 digits, and not reversing the answers for enc-dec models.

## [Experiment 8](../arithmetic_lm/conf/experiment/8)
Training set: 1-9 digit addition EXCEPT 8 digit addition. Test on 1-10 digit addition. Padding operands to 10 digits, answers to 11 digits, and not reversing the answers for enc-dec models.

## [Experiment 9](../arithmetic_lm/conf/experiment/9)
Go back, no zero padding. Like experiment 8, but no zero padding, but with fixed filler tokens instead. In our case filler tokens are 10 dots.

## [Experiment 10](../arithmetic_lm/conf/experiment/10)
Train UT-dec and Trans-dec on 8 digits, evaluate on 1-8 to see if it can generalize to a lower number of digits. Dataset: [add_generalize_to_lower.yaml](../arithmetic_lm/conf/data/add_generalize_to_lower.yaml)

## [Experiment 11](../arithmetic_lm/conf/experiment/11)
Like Experiment 10, but trained on 7 and 8 digits, evaluated on 1-8. Dataset: [add_generalize_to_lower_7-8.yaml](../arithmetic_lm/conf/data/add_generalize_to_lower_7-8.yaml)

## [Experiment 12](../arithmetic_lm/conf/experiment/12)
Train on 1M 1x1-50x50 digit addition examples except 1x1-5x5, 20x20, and 45x45. Test on a different set of excluded pairs and 51x51. Dataset: [add_high_n_digit_variation.yaml](../arithmetic_lm/conf/data/add_high_n_digit_variation.yaml)

## [Experiment 13](../arithmetic_lm/conf/experiment/13)
Train on 1M 1x1-19x19 excluding 18x18, test on 1x1-20x20 (18 digits are for in-between OOD, 20 longer OOD generalization). Dataset: [add_generalize_to_longer_19.yaml](../arithmetic_lm/conf/data/add_generalize_to_longer_19.yaml)

## [Experiment 14](../arithmetic_lm/conf/experiment/14)
Curriculum learning, start training with 1x1, then add 2x2, 3x3, etc. Since this is the first curriculum training, the stages are manually started/resumed.

## [Experiment 15](../arithmetic_lm/conf/experiment/15)
Like Experiment 13, but with chain of thought.

## [Experiment 16](../arithmetic_lm/conf/experiment/16)
Pretrain on a character matching task, 5M training examples of 1-15 mixed operand length, test on in-distribution and OOD of 13-17 chars. Lowercase letters and numbers are used. Dataset: [matching_v1.yaml](../arithmetic_lm/conf/data/matching_v1.yaml)

## [Experiment 17](../arithmetic_lm/conf/experiment/17)
Step back to a simpler problem: string length in range 1-20 and test on in-distribution 1-20 and OOD lengths 21-30 and 31-40. e.g. prompt "$somestringhere=" to answer "14$". Dataset: [strlen_v1_1M.yaml](../arithmetic_lm/conf/data/strlen_v1_1M.yaml). 1M training examples.

## [Experiment 18](../arithmetic_lm/conf/experiment/18)
String character retrieval by index. 2M training examples of 1-25 string length, test on in-distribution 1-25 and OOD 25-30 and 31-40 and 31-40. e.g. prompt "$somestringhere[2]" to answer "m$". Dataset: [strindex_v1_2M.yaml](../arithmetic_lm/conf/data/strindex_v1_2M.yaml)

## [Experiment 19](../arithmetic_lm/conf/experiment/19)
Like string length, but more primitve. Instead of outputting a number, output the same number of dots as string length, e.g. prompt "$somestringhere=" to answer "..............$". Dataset: [strlen_v2_1M.yaml](../arithmetic_lm/conf/data/strlen_v2_1M.yaml). 1M training examples. Train on length 1-10, test on in-dist 1-10, and OOD 11-15 and 16-20.

## [Experiment 20](../arithmetic_lm/conf/experiment/20)
Like Experiment 16, but matching only digits instead of characters, and with abacus embeddings. Dataset: [matching_v2.yaml](../arithmetic_lm/conf/data/matching_digits.yaml)

## [Experiment 21](../arithmetic_lm/conf/experiment/21)
Like Experiment 13 (addition-generalize-to-longer) but with no carries in training set. Dataset: [add_generalize_to_longer_20_nocarry.yaml](../arithmetic_lm/conf/data/add_generalize_to_longer_20_nocarry.yaml)

## [Experiment 22](../arithmetic_lm/conf/experiment/22)
Same as generalize_to_longer_19, but using a 2M samples in train dataset instead of 1M. Dataset: [add_generalize_to_longer_19_2M.yaml](../arithmetic_lm/conf/data/add_generalize_to_longer_19_2M.yaml)

## [Experiment 23](../arithmetic_lm/conf/experiment/23)
Train on 1-7 and 9 digit addition, test on 8 and 10-13 digit addition. Try different training dataset sizes: 100K, 1M, 10M. Dataset: [add_generalize_to_longer_mini.yaml](../arithmetic_lm/conf/data/add_generalize_to_longer_mini.yaml)

## [Experiment 24](../arithmetic_lm/conf/experiment/24)
As in add_generalize_to_longer_mini, but with multitask setup.
- Subtasks:
  - `rev` operand reversing: `$123+456=` -> `321+654$`
  - `ali` digit alignment: `$123+456=` -> `1+4,2+5,3+6$`
  - `mad` digit-wise modular addition (ignore carries): `$345+678=` -> `913$`
  - `car` detecting carries (c for carry, . otherwise): `$234+678=` -> `.cc$`
    - (full) `add` addition: `$123+456=` -> `579$`
- Use flags to specify tasks:
  - prefix the prompt with task name
  - `rev$123+456=` -> `321+654$`
  - full addition task add: `add$123+456=` -> `579$`

## [Experiment 25](../arithmetic_lm/conf/experiment/25)
Looped transformer decoder on addition 100K.

## [Experiment 26](../arithmetic_lm/conf/experiment/26)
Different pos encodings on gen-to-longer-mini.

## [Experiment 27](../arithmetic_lm/conf/experiment/27)
Compositional learning with sub-task training data across data and model scales.
Questions to answer:
- how much does sub-task data (digit alignment, carry detection, digit-wise modular addition, reversing) help to learn multi-digit integer addition?
- how does this contribution change with respect to model and dataset sizes?
Model: transformer decoder with 6 layers and 8 heads
Data format: not reversed operands and answers, operands have random spaces added (ratio 0.5, means if seq length is 10, up to 5 random spaces added)
Batch size 128
32k steps
Dataset: 1-7 + 9 digit train, 8, 10, 11 digits OOD
What varies:
embedding dimension d_model in range from 64 to 1536 in increments of 64 (=24)
repeat for 4 train dataset sizes: 10k, 100k, 1M, 10M samples
repeat for add-only and mix training data (same size, e.g. 10k addition or 2k addition + 8k other tasks)
repeat for 3 seeds each

## [Experiment 28](../arithmetic_lm/conf/experiment/28)
The models can usually close a gap of 1 digit between trained digits (e.g. trained on 1-7 and 9, 8 digit OOD works good). Similar to generalize-to-longer-mini, but use a dataset where training is on 1-7 and 11 digits, see if a larger gap (8,9,10) can also be closed. Test on OOD: 8, 9, 10, 12, 13.

## [Experiment 29](../arithmetic_lm/conf/experiment/29)
Different pos encodings on gen-to-longer-mini. Absolute, absolute + random spaces, abacus, fire + index hints, abs + index hints.