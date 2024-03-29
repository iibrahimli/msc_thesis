## Dependencies

The code has been tested on Python 3.12. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Generating datasets

To generate the datasets, run the following command:

```bash
# addition
python -m arithmetic_lm.dataset.generate_addition
```

which will generate the datasets in the `data/` directory in the project root.

## Training

The training script uses [Hydra](hydra.cc/docs) and can be run as follows:

```bash
python -m arithmetic_lm.train +experiment=<experiment_name>
```

e.g.

```bash
# first, generate the datasets if you haven't already using the command above
# to train NanoGPT on 1-3 digit addition task
python -m arithmetic_lm.train +experiment=1/exp1_nanogpt
```

The parameters can be overridden by using CLI arguments, using the dotted key of the parameter in the YAML config file as such:

```bash
# override the batch size to 128
python -m arithmetic_lm.train +experiment=1/exp1_nanogpt training.batch_size=128

# another useful example: use only specified GPUs (id 0 and 3 as seen in nvidia-smi output)
python -m arithmetic_lm.train +experiment=1/exp1_nanogpt training.devices=[0,3]
```

Note that the key and value are separated by an `=` sign, and no spaces are used.

> Run the training script with `--help` to see all the available configuration options:
> ```bash
> python -m arithmetic_lm.train --help
> ```
