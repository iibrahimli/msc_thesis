The config structure:

```yaml
# Experiment 1: Train on 1 and 3 digit addition, test on 1-4 digit addition

# section for data
data:
  # name of the dataset
  name: "Addition 1-3 digits"
  format:
    pad: $ # padding token before and after example, set to null for no padding
    reverse_ans: true # reverse the answer, e.g. 12+34=46 -> 12+34=64
    encdec: false # use encoder-decoder format, e.g. src='12+34' tgt='=46' if true otherwise regular language modeling for decoder-only model 
  train:
    # path to the training data
    - "data/experiment_1/train_add_1-3digit.txt"
  test:
    # paths to the test data
    1digit: "data/experiment_1/test_add_1digit.txt"
    2digit: "data/experiment_1/test_add_2digit.txt"
    3digit: "data/experiment_1/test_add_3digit.txt"
    4digit: "data/experiment_1/test_add_4digit.txt"

model:
  name: "NanoGPT"
  args:
    # model parameters, depends on the model
    context_len: 256
    n_embd: 384
    n_head: 6
    n_layer: 6
    dropout: 0.1

training:
  batch_size: 256
  lr: 0.001
  weight_decay: 0.1
  warmup_iters: 100
  max_iters: 10000
  num_dl_workers: 4
  val_ratio: 0.1  # ratio of training data to use for validation
  val_interval: 100  # validate (and test) every X iterations
  limit_val_batches: null # null = no limit, val and test on all batches
  devices: [6]  # list of GPU ids to use

sampling:
  temperature: 0.8
  top_k: 1

wandb:
  enable: true
  entity: "compositional-generalization-ut"
  project: "addition-1-3-digit"
  run_name: "transformer_6layers"
  grad_log_interval: 500  # log gradients every X iterations
```


Values can be overridden by command line arguments, e.g. `model.args.context_len=512` will override the context_len parameter in the config file.