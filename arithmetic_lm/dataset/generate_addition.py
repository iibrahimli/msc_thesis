"""
Generate balanced dataset for addition task
"""

import math
import random
from pathlib import Path

from loguru import logger

from ..constants import DATA_DIR
from ..utils import set_seed


def num_carry_ops(a: int, b: int) -> int:
    """Number of carry operations in addition of a and b"""

    def digit_sum(n: int):
        return sum(map(int, str(n)))

    return int((digit_sum(a) + digit_sum(b) - digit_sum(a + b)) / 9)


def k_notation(number: int) -> str:
    """Convert number to k notation"""
    return f"{number // 1000}k" if number >= 1000 else f"{number}"


# for train set
def generate_balanced(
    filepath: str | Path, num_examples: int = 10_000, num_digits: int = 3
) -> None:
    """Generate addition problems with balanced number of digits and carries"""
    set_seed(42)

    num_digit_2 = int(900 * num_examples / 10000)
    num_digit_list = [100, num_digit_2, num_examples - 100 - num_digit_2]

    logger.info(f"Number of examples for each digit: {num_digit_list}")

    # create a list of number of carries - we target each number of carries to have the same number of examples
    target_num_carry_examples = math.ceil(num_examples / (num_digits + 1))
    num_carry_list = [0 for _ in range(num_digits + 1)]

    logger.info(f"Target number of carry examples: {target_num_carry_examples}")

    with open(filepath, "w") as f:
        num_example = 0

        # generate all 1 digit examples
        for a in range(10):
            for b in range(10):
                c = a + b
                f.write(f"{a}+{b}={c}\n")
                num_example += 1
                num_carry = num_carry_ops(a, b)
                num_carry_list[num_carry] += 1

        for num_digit in range(2, num_digits + 1):
            num_digit_example = 0

            while (
                num_digit_example < num_digit_list[num_digit - 1]
                and num_example < num_examples
            ):
                # generate a random number between 0 and 10^(i+1) - 1
                a = random.randint(0, 10 ** (num_digit) - 1)
                b = random.randint(0, 10 ** (num_digit) - 1)
                c = a + b

                # count number of carries in c
                num_carry = num_carry_ops(a, b)
                if num_carry_list[num_carry] < target_num_carry_examples:
                    # write the example to file
                    f.write(f"{a}+{b}={c}\n")

                    # increment num_carry_list[num_carry]
                    num_carry_list[num_carry] += 1
                    num_digit_example += 1
                    num_example += 1
                else:
                    continue

    logger.info(f"Number of carries for each digit: {num_carry_list}")


# for test set
def generate_uniform_exclude(
    filepath: str | Path,
    exclude: set[str],
    num_digits: int,
    num_examples: int,
) -> None:
    """
    Uniformly sample addition problems, excluding the given examples and without answer
    """
    set_seed(42)

    c = 0
    max_val = 10**num_digits - 1
    with open(filepath, "w") as f:
        while c < num_examples:

            a = random.randint(0, max_val)
            b = random.randint(0, max_val)

            example = f"{a}+{b}="
            ans = f"{a+b}\n"

            if (example + ans) in exclude:
                continue

            f.write(example + "\n")
            c += 1


# for smaller datasets
def create_subset_dataset(
    input_filepath: str | Path, output_filepath: str | Path, num_samples: int
) -> None:
    """Generate a subset of given dataset, given we have 10k dataset"""
    set_seed(42)

    with open(input_filepath, "r") as f:
        lines = f.readlines()

    # select the first 100 lines (1-digit examples)
    selected_lines = lines[:100]
    selected_lines += random.sample(lines[100:1000], int(900 * num_samples / 10000))
    selected_lines += random.sample(
        lines[1000:], num_samples - 100 - int(900 * num_samples / 10000)
    )

    with open(output_filepath, "w") as f:
        f.writelines(selected_lines)


def generate_addition(out_dir: str | Path, num_digits: int, num_examples: int):
    """
    Generate dataset for addition task. Writes 2 files:
    - add_Ndigit_Mk_bal.txt: M (thousand) balanced addition problems
    - add_Ndigit_Mk_test.txt: M thousand uniformly sampled without overlap with train set
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    kn = k_notation(num_examples)
    train = out_dir / f"add_{num_digits}digit_{kn}_bal.txt"
    test = out_dir / f"add_{num_digits}digit_{kn}_test.txt"

    logger.info(f"Generating {num_examples} samples to {train}")
    generate_balanced(train, num_digits=3, num_examples=num_examples)

    # create set of excluded train examples for generating test set
    with open(train, "r") as f:
        lines = f.readlines()
        exclude = set(lines)

    logger.info(f"Generating {num_examples} samples to {test}")
    generate_uniform_exclude(
        test, exclude=exclude, num_digits=3, num_examples=num_examples
    )


# Generate all datasets
if __name__ == "__main__":

    # generate 10k 3-digit addition problems
    generate_addition(DATA_DIR / "add_3digit", num_digits=3, num_examples=10_000)

    # generate smaller datasets
    for num_examples in [1000, 2000, 5000]:
        kn = k_notation(num_examples)
        outfile = DATA_DIR / "add_3digit" / f"add_3digit_{kn}_bal.txt"
        logger.info(f"Generating {num_examples} subset to {outfile}")
        create_subset_dataset(
            input_filepath=DATA_DIR / "add_3digit" / f"add_3digit_10k_bal.txt",
            output_filepath=outfile,
            num_samples=num_examples,
        )
