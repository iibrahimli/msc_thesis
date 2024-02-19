## Dependencies

Dependencies are managed using [Poetry](https://python-poetry.org/). To install the dependencies, run the following command:

```bash
poetry install --with dev
```

## Generating datasets

To generate the datasets, run the following command:

```bash
# addition
poetry run python -m arithmetic_lm.dataset.generate_addition
```

which will generate the datasets in the `data/addition` directory in the project root.
