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


def get_set_from_file(filepath: str | Path) -> set[str]:
    with open(filepath, "r") as f:
        return set(f.readlines())


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
                ans = str(a + b)
                f.write(FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans))
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
                ans = str(a + b)

                # count number of carries in ans
                num_carry = num_carry_ops(a, b)
                if num_carry_list[num_carry] < target_num_carry_examples:
                    # write the example to file
                    f.write(FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans))

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
            ans = str(a + b)

            example_with_ans = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans)

            if example_with_ans in exclude:
                continue

            # example = example_with_ans.split("=")[0] + "=\n"
            f.write(example_with_ans)
            c += 1


# for N digit tasks
def generate_only_digit(
    filepath: str | Path,
    num_digits: int,
    num_examples: int,
    exclude: set[str] = None,
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
                    ans = str(a + b)
                    example = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans)
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
            ans = str(a + b)
            example = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans)
            if exclude and example in exclude:
                continue
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


def generate_experiment_1(out_dir: str | Path):
    """
    Generate data for Experiment 1 (Trying to replicate length generalization,
    Figure 22(a) from Lee 2023 "Teaching Arithmetic to Small Transformers"):
     1. generate 1,2,3 digit balanced 10k examples dataset `add_1-2-3digit_10k_bal.txt`
     2. from that, remove examples featuring 2 digit operands to get `train_add_1-3digit.txt`
     3. generate {1,2,3,4} digit test datasets of 100 examples each `test_Xdigit_100.txt`
    (the test data should have no overlap with train file)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating data for experiment {out_dir}")

    # 1. generate balanced dataset
    bal_path = out_dir / "add_1-2-3digit_10k_bal.txt"
    n_bal_examples = 10_000
    logger.info(f"Generating {bal_path}")
    generate_balanced(
        filepath=bal_path,
        num_examples=n_bal_examples,
        num_digits=3,
    )

    def _does_not_feature_2_digit_operand(line: str) -> bool:
        a, bc = line.split(OPERATOR)
        b, _ = bc.split("=")
        return len(a.strip()) != 2 and len(b.strip()) != 2

    # 2. remove 2 digit examples
    logger.info("Removing 2 digit examples")
    with open(bal_path, "r") as f:
        lines = f.readlines()
    lines = list(filter(_does_not_feature_2_digit_operand, lines))
    with open(out_dir / "train_add_1-3digit.txt", "w") as f:
        f.writelines(lines)

    # 3. generate test datasets
    n_test_examples = 100
    for n_digits in (1, 2, 3, 4):
        digit_path = out_dir / f"test_{n_digits}digit_{n_test_examples}.txt"
        logger.info(f"Generating {digit_path}")
        generate_only_digit(digit_path, num_digits=n_digits, num_examples=100)


def main():
    generate_experiment_1(DATA_DIR / "experiment_1")


if __name__ == "__main__":
    main()
