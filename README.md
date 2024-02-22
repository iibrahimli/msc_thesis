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
