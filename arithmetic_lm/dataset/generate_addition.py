"""
Generate balanced dataset for addition task
"""

import math
import random
from pathlib import Path

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


def n_possible_examples(num_digits: int) -> int:
    """Number of possible examples for given number of digits exactly"""
    # exlude the case where either start with 0
    return 100**num_digits - 99


# for train set
def generate_balanced(
    filepath: str | Path,
    num_examples: dict[int, int],  # digit -> num_examples
    exclude: set[str] = None,
    balance_carries: bool = True,
    seed: int = 42,
) -> None:
    """Generate addition problems with balanced number of carries"""
    set_seed(seed)

    if 1 in num_examples:
        assert (
            num_examples[1] == 100
        ), f"Expected 100 examples for 1 digit for coverage, got {num_examples[1]}"

    for i, n in num_examples.items():
        assert n > 0, f"Expected positive number of examples for {i} digit"
        if i > 1:
            assert n <= n_possible_examples(
                i
            ), f"Can't generate more than {n_possible_examples(i)} examples for {i} digit (requested {n})"

    # for each digit, we target each number of carries (0...N) to have the same number of examples
    num_carries = {i: [0] * (i + 1) for i in num_examples}
    num_target_carries = {i: math.ceil(n / (i + 1)) for i, n in num_examples.items()}

    with open(filepath, "w") as f:

        for num_digit, num_digit_examples in num_examples.items():

            if num_digit == 1 or num_digit_examples == n_possible_examples(num_digit):
                print(
                    f"Generating all possible {num_digit} digit examples ({num_digit_examples})"
                )
                # generate all examples
                min_val = 10 ** (num_digit - 1) if num_digit > 1 else 0
                max_val = 10**num_digit - 1
                for a in range(min_val, max_val + 1):
                    for b in range(min_val, max_val + 1):
                        ans = str(a + b)
                        example_str = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans)
                        if exclude and example_str in exclude:
                            continue
                        f.write(example_str)
                        # increment the number of examples for this carry
                        num_carry = num_carry_ops(a, b)
                        num_carries[num_digit][num_carry] += 1
                continue

            print(f"Using random sampling for {num_digit} digit examples")

            num_current_digit_examples = 0
            generated_examples = set()

            while num_current_digit_examples < num_digit_examples:
                a = random.randint(10 ** (num_digit - 1), 10**num_digit - 1)
                b = random.randint(10 ** (num_digit - 1), 10**num_digit - 1)
                ans = str(a + b)

                # count number of carries in ans
                num_carry = num_carry_ops(a, b)

                if balance_carries:
                    # check if we have enough examples for this carry
                    if (
                        num_carries[num_digit][num_carry]
                        >= num_target_carries[num_digit]
                    ):
                        # enough carries for this digit
                        continue

                example_str = FMT_STR.format(a=a, op=OPERATOR, b=b, ans=ans)

                if exclude and example_str in exclude:
                    continue

                if example_str in generated_examples:
                    continue
                generated_examples.add(example_str)

                # write the example to file
                f.write(example_str)

                # increment the number of examples for this carry
                num_carries[num_digit][num_carry] += 1
                num_current_digit_examples += 1

    carry_summary = {
        i: f"{num_carries[i]} target={num_target_carries[i]}" for i in num_examples
    }
    print(f"Number of carries for each digit / target: {carry_summary}")


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
            start = 10 ** (num_digits - 1) if num_digits > 1 else 0
            end = 10**num_digits
            for a in range(start, end):
                for b in range(start, end):
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
    print()
    print(f" > Generating data for Experiment 1 to {out_dir}")

    # 1. generate balanced dataset
    bal_path = out_dir / "add_1-2-3digit_10k_bal.txt"
    n_bal_examples = 10_000
    print(f"Generating {bal_path}")
    generate_balanced(
        filepath=bal_path,
        num_examples={
            1: 100,
            2: 900,
            3: n_bal_examples - 100 - 900,
        },
    )

    def _does_not_feature_2_digit_operand(line: str) -> bool:
        a, bc = line.split(OPERATOR)
        b, _ = bc.split("=")
        return len(a.strip()) != 2 and len(b.strip()) != 2

    # 2. remove 2 digit examples
    print("Removing 2 digit examples")
    with open(bal_path, "r") as f:
        lines = f.readlines()
    lines = list(filter(_does_not_feature_2_digit_operand, lines))
    with open(out_dir / "train_add_1-3digit.txt", "w") as f:
        f.writelines(lines)

    # 3. generate test datasets
    n_test_examples = 100
    for n_digits in (1, 2, 3, 4):
        digit_path = out_dir / f"test_add_{n_digits}digit_{n_test_examples}.txt"
        print(f"Generating {digit_path}")
        generate_only_digit(digit_path, num_digits=n_digits, num_examples=100)


def generate_experiment_2(out_dir: str | Path):
    """
    Generate data for Experiment 2 (Trying to replicate length generalization,
    Figure 22(b) from Lee 2023 "Teaching Arithmetic to Small Transformers"),
    assuming Experiment 1 data is already generated in the out_dir:
     1. generate 1-7 digit 100k examples train dataset `train_add_1-7digit.txt`
     2. generate {5,6,7,8} digit test datasets of 100 examples each `test_Xdigit_100.txt`
    (the test data should have no overlap with train file)
    """

    out_dir = Path(out_dir)
    assert out_dir.exists(), "Expected experiment 1 data to already exist"
    print()
    print(f" > Generating data for Experiment 2 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_1-7digit.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples={
            1: 100,
            2: 9901,
            3: 16650,
            4: 16650,
            5: 16650,
            6: 16650,
            7: 16650,
        },  # total 100k
    )

    # 2. generate 5-8 digit test datasets (1-4 generated in Experiment 1)
    n_test_examples = 100
    for n_digits in (5, 6, 7, 8):
        digit_path = out_dir / f"test_add_{n_digits}digit_{n_test_examples}.txt"
        print(f"Generating {digit_path}")
        generate_only_digit(
            digit_path,
            num_digits=n_digits,
            num_examples=100,
            exclude=get_set_from_file(train_path),
            seed=n_digits,  # different seed to avoid overlap
        )


def generate_experiment_3(out_dir: str | Path):
    """
    Experiment 3: 800k examples of 3x3 digit addition problems for training,
    200k examples of 3x3 digit addition problems for testing (no overlap)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 3 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_3x3digit_800k.txt"
    print(f"Generating {train_path}")
    generate_only_digit(train_path, num_digits=3, num_examples=800_000)

    # 2. generate test dataset
    test_path = out_dir / "test_add_3x3digit_200k.txt"
    print(f"Generating {test_path}")
    generate_only_digit(
        test_path,
        num_digits=3,
        num_examples=200_000,
        exclude=get_set_from_file(train_path),
    )


def generate_experiment_4(out_dir: str | Path):
    """
    Experiment 4: Like Experiment 3, but with 7x7 digit addition problems,
    800k examples for training and 200k examples for testing (no overlap)
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 4 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_7x7digit_800k.txt"
    print(f"Generating {train_path}")
    generate_only_digit(train_path, num_digits=7, num_examples=800_000)

    # 2. generate test dataset
    test_path = out_dir / "test_add_7x7digit_200k.txt"
    print(f"Generating {test_path}")
    generate_only_digit(
        test_path,
        num_digits=7,
        num_examples=200_000,
        exclude=get_set_from_file(train_path),
    )


def generate_experiment_8(out_dir: str | Path):
    """
    Experiment 8: ~1M examples of 1 to 9 digit addition problems for training
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 8 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_1-9digit_except8_1M.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples={
            1: 100,
            2: 9901,  # exactly all possible 2 digit examples
        }
        | {i: 100_000 for i in [3, 4, 5, 6, 7, 9]},  # NOTE no 8 digit examples
    )

    # 2. generate test datasets
    n_test_examples = 100
    excluded = get_set_from_file(train_path)
    for n_digits in range(1, 11):
        digit_path = out_dir / f"test_add_{n_digits}digit_{n_test_examples}.txt"
        print(f"Generating {digit_path}")
        generate_only_digit(
            digit_path,
            num_digits=n_digits,
            num_examples=100,
            exclude=(
                excluded if n_digits > 2 else None
            ),  # not enough samples to exclude in 1-2 digits
            seed=n_digits,  # different seed to avoid overlap
        )


def generate_experiment_10(out_dir: str | Path):
    """
    Experiment 10: ~1M examples of 8 digit addition problems for training,
    100 examples of 1-8 for testing
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 10 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_8digit_1M.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples={8: 1_000_000},
    )

    # 2. generate test datasets
    n_test_examples = 100
    excluded = get_set_from_file(train_path)
    for n_digits in range(1, 9):
        digit_path = out_dir / f"test_add_{n_digits}digit_{n_test_examples}.txt"
        print(f"Generating {digit_path}")
        generate_only_digit(
            digit_path,
            num_digits=n_digits,
            num_examples=100,
            exclude=(
                excluded if n_digits > 2 else None
            ),  # not enough samples to exclude in 1-2 digits
            seed=n_digits,  # different seed to avoid overlap
        )


def generate_experiment_11(out_dir: str | Path):
    """
    Experiment 11: ~1M examples of 7 and 8 digit addition problems for training,
    100 examples of 1-8 for testing
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 11 to {out_dir}")

    # 1. generate train dataset
    train_path = out_dir / "train_add_7-8digit_1M.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples={7: 500_000, 8: 500_000},
    )

    # 2. generate test datasets
    n_test_examples = 100
    excluded = get_set_from_file(train_path)
    for n_digits in range(1, 9):
        digit_path = out_dir / f"test_add_{n_digits}digit_{n_test_examples}.txt"
        print(f"Generating {digit_path}")
        generate_only_digit(
            digit_path,
            num_digits=n_digits,
            num_examples=100,
            exclude=(
                excluded if n_digits > 2 else None
            ),  # not enough samples to exclude in 1-2 digits
            seed=n_digits,  # different seed to avoid overlap
        )


def generate_experiment_12(out_dir: str | Path):
    """
    Experiment 12: train on 1M 1x1-50x50 digit addition examples
    except 1x1-5x5, 20x20, and 45x45. Test on a different set of excluded
    pairs and 51x51
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 12 to {out_dir}")

    # out of distribution
    out_dist = {1, 2, 3, 4, 5, 20, 45, 51}
    in_dist = set(range(1, 51)) - out_dist
    train_num_examples = {i: 1_000_000 // len(in_dist) for i in in_dist}

    # generate train dataset
    train_path = out_dir / "train_add_6-50digit_except_20-45_1M.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples=train_num_examples,
        balance_carries=False,  # too slow for large digit numbers, TODO: optimize
    )

    # train examples excluded from test
    excluded = get_set_from_file(train_path)

    # generate test dataset with trained digit lengths (in distribution)
    in_distribution_test_path = out_dir / "test_add_in_distribution_2000.txt"
    print(f"Generating {in_distribution_test_path}")
    generate_balanced(
        filepath=in_distribution_test_path,
        num_examples={i: 2000 / len(in_dist) for i in in_dist},
        exclude=excluded,
        balance_carries=False,  # too slow for large digit numbers, TODO: optimize
    )

    # generate test dataset with out of distribution digit lengths
    for i in out_dist:
        out_dist_test_path = out_dir / f"test_add_ood_{i}digit_100.txt"
        print(f"Generating {out_dist_test_path}")
        generate_only_digit(
            out_dist_test_path,
            num_digits=i,
            num_examples=100,
            exclude=excluded,
            seed=i,
        )


def generate_experiment_13(out_dir: str | Path):
    """
    Train on 1M 1x1-19x19 excluding 18x18, test on 1x1-20x20 (18 digits are for
    in-between OOD, 20 longer OOD generalization).
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 13 to {out_dir}")

    # out of distribution
    out_dist = {18, 20}
    in_dist = set(range(1, 20)) - out_dist
    train_num_examples = {i: 999_000 // len(in_dist) for i in in_dist} | {
        1: 100,
        2: 9901,
    }

    # generate train dataset
    train_path = out_dir / "train_add_1-19_except18_1M.txt"
    print(f"Generating {train_path}")
    generate_balanced(
        filepath=train_path,
        num_examples=train_num_examples,
        balance_carries=False,  # too slow for large digit numbers, TODO: optimize
    )

    # train examples excluded from test
    excluded = get_set_from_file(train_path)

    # generate test dataset with trained digit lengths (in distribution)
    in_distribution_test_path = out_dir / "test_add_in_distribution_2000.txt"
    print(f"Generating {in_distribution_test_path}")
    generate_balanced(
        filepath=in_distribution_test_path,
        num_examples={
            i: 2000 / len(in_dist) for i in in_dist if i not in (1, 2)
        },  # exclude 1 and 2 since they're fully covered in training dataset
        exclude=excluded,
        balance_carries=False,  # too slow for large digit numbers, TODO: optimize
    )

    # generate test dataset with out of distribution digit lengths
    for i in out_dist:
        out_dist_test_path = out_dir / f"test_add_ood_{i}digit_100.txt"
        print(f"Generating {out_dist_test_path}")
        generate_only_digit(
            out_dist_test_path,
            num_digits=i,
            num_examples=100,
            exclude=excluded,
            seed=i,
        )


def main():
    # generate_experiment_1(DATA_DIR / "addition")
    # generate_experiment_2(DATA_DIR / "addition")
    # generate_experiment_3(DATA_DIR / "addition")
    # generate_experiment_4(DATA_DIR / "addition")
    # generate_experiment_8(DATA_DIR / "addition" / "exp_8")
    # generate_experiment_10(DATA_DIR / "addition" / "exp_10")
    # generate_experiment_11(DATA_DIR / "addition" / "exp_11")
    # generate_experiment_12(DATA_DIR / "addition" / "exp_12")
    generate_experiment_13(DATA_DIR / "addition" / "exp_13")


if __name__ == "__main__":
    main()
