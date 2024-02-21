"""
Generate balanced dataset for addition task
"""

import math
import random
from pathlib import Path

from loguru import logger

from .. import formatting
from ..constants import DATA_DIR
from ..utils import set_seed

FMT_STR = formatting.PLAIN_FORMAT_STR
OPERATOR = "+"


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
    filepath: str | Path,
    num_examples: int = 10_000,
    num_digits: int = 3,
    seed: int = 42,
) -> None:
    """Generate addition problems with balanced number of digits (<= num_digits) and carries"""
    set_seed(seed)

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
                f.write(FMT_STR.format(a=a, op=OPERATOR, b=b, ans=c))
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
                    f.write(FMT_STR.format(a=a, op=OPERATOR, b=b, ans=c))

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
    seed: int = 42,
) -> None:
    """
    Uniformly sample addition problems, excluding the given examples and without answer
    for <= num_digit digit numbers
    """
    set_seed(seed)

    c = 0
    max_val = 10**num_digits - 1
    with open(filepath, "w") as f:
        while c < num_examples:

            a = random.randint(0, max_val)
            b = random.randint(0, max_val)

            example_with_ans = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=c)

            if example_with_ans in exclude:
                continue

            # remove answer and write to file
            example = example_with_ans.split("=")[0] + "=\n"
            f.write(example)
            c += 1


# for N digit tasks
def generate_only_digit(
    filepath: str | Path,
    num_digits: int,
    num_examples: int,
    include_ans: bool = True,
    seed: int = 42,
):
    """Generate addition problems with given number of digits"""
    set_seed(seed)

    # if <= 2 digits and enough num_examples, generate all possible examples
    if num_digits <= 2 and num_examples >= 100**num_digits:
        with open(filepath, "w") as f:
            for a in range(10 ** (num_digits - 1), 10**num_digits):
                for b in range(10 ** (num_digits - 1), 10**num_digits):
                    c = a + b
                    example = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=c)
                    if not include_ans:
                        example = example.split("=")[0] + "=\n"
                    f.write(example)
        return

    # otherwise, generate uniformly sampled examples
    min_val = 10 ** (num_digits - 1)
    max_val = 10**num_digits - 1
    with open(filepath, "w") as f:
        count = 0
        while count < num_examples:
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
            c = a + b
            example = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=c)
            if not include_ans:
                example = example.split("=")[0] + "=\n"
            f.write(example)
            count += 1


# for smaller datasets
def create_subset_dataset(
    input_filepath: str | Path,
    output_filepath: str | Path,
    num_samples: int,
    seed: int = 42,
) -> None:
    """Generate a subset of given dataset, given we have 10k dataset"""
    set_seed(seed)

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
    - add_Ndigit_Mk_bal.txt: M (thousand) balanced addition problems up to N digits
    - add_Ndigit_Mk_test.txt: M thousand uniformly sampled without overlap with train set
    """

    out_dir = Path(out_dir)
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


def main():

    # generate 10k 3-digit addition problems
    # this is the main, largest training dataset, smaller train datasets
    # are subsets of this, and test datasets are generated to not have
    # overlap with this dataset
    generate_addition(DATA_DIR / "add_3digit_bal", num_digits=3, num_examples=10_000)

    # generate smaller train datasets
    for num_examples in [1000, 2000, 5000]:
        kn = k_notation(num_examples)
        outfile = DATA_DIR / "add_3digit_bal" / f"add_3digit_{kn}_bal.txt"
        logger.info(f"Generating {num_examples} subset to {outfile}")
        create_subset_dataset(
            input_filepath=DATA_DIR / "add_3digit_bal" / "add_3digit_10k_bal.txt",
            output_filepath=outfile,
            num_samples=num_examples,
        )

    # generate smaller test datasets
    for num_examples in [1000]:
        kn = k_notation(num_examples)
        outfile = DATA_DIR / "add_3digit_bal" / f"add_3digit_{kn}_test.txt"
        with open(DATA_DIR / "add_3digit_bal" / "add_3digit_10k_bal.txt", "r") as f:
            lines = f.readlines()
            exclude = set(lines)
        logger.info(f"Generating {num_examples} uniform test dataset to {outfile}")
        generate_uniform_exclude(
            outfile,
            exclude=exclude,
            num_digits=3,
            num_examples=num_examples,
            seed=num_examples,
        )

    # generate N-digit datasets
    for num_digits in [1, 2, 3, 4, 5, 6, 7]:
        n_examples = 10_000
        kn = k_notation(n_examples)
        test = False  # whether to include answer in the dataset (test = no answer)
        suffix = "_test" if test else ""
        subdir = DATA_DIR / "add_digits"
        subdir.mkdir(parents=True, exist_ok=True)
        outfile = subdir / f"add_only{num_digits}digit_{kn}{suffix}.txt"
        logger.info(f"Generating {n_examples} {num_digits}-digit dataset to {outfile}")
        generate_only_digit(
            outfile,
            num_digits=num_digits,
            num_examples=n_examples,
            include_ans=not test,
        )


if __name__ == "__main__":
    main()
