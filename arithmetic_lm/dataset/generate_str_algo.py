"""
Generate datasets for string algorithms experiments
"""

import random
import string
from pathlib import Path

from ..constants import DATA_DIR

CHARS = list(string.ascii_lowercase + string.digits)


def get_set_from_file(filepath: str | Path) -> set[str]:
    with open(filepath, "r") as f:
        return set(f.readlines())


def build_matching_example(a: str, b: str) -> str:
    res = f"{a}+{b}="
    res += "".join([f"{ca}+{cb}," for ca, cb in zip(a[::-1], b[::-1])])[:-1]
    return res


def generate_experiment_16(out_dir: str | Path):
    """
    Generate 5M training samples of 1-15 chars, and testing samples of in-dist
    and 16 chars, 2000 samples each
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 16 to {out_dir}")

    # generate train dataset
    n_train = 5_000_000
    max_train_len = 15
    train_path = out_dir / "train_match_1-15chars_5M.txt"
    print(f"Generating {train_path}")
    with open(train_path, "w") as f:
        for _ in range(n_train):
            a = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            b = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            example = build_matching_example(a, b) + "\n"
            f.write(example)

    excluded = get_set_from_file(train_path)

    # generate test datasets
    n_test = 2000
    # in dist
    test_path = out_dir / "test_match_in_dist_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        i = 0
        while i < n_test:
            a = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            b = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            example = build_matching_example(a, b) + "\n"
            if example in excluded:
                continue
            f.write(example)
            i += 1

    # 13-17 chars
    test_path = out_dir / "test_match_ood_13-17chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for _ in range(n_test):
            a = "".join(random.choices(CHARS, k=random.randint(13, 17)))
            b = "".join(random.choices(CHARS, k=random.randint(13, 17)))
            example = build_matching_example(a, b) + "\n"
            f.write(example)


def generate_experiment_17(out_dir: str | Path):
    """
    String length, train 1-20 symbols, test 1-20 in-dist, 21-30 out-of-dist
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 17 to {out_dir}")

    # generate train dataset
    n_train = 1_000_000
    max_train_len = 20
    train_path = out_dir / "train_strlen_1-20chars_1M.txt"
    print(f"Generating {train_path}")
    with open(train_path, "w") as f:
        for _ in range(n_train):
            a = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            f.write(f"{a}={len(a)}\n")

    excluded = get_set_from_file(train_path)

    # generate test datasets
    n_test = 2000
    # in dist
    test_path = out_dir / "test_strlen_in_dist_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        i = 0
        while i < n_test:
            a = "".join(random.choices(CHARS, k=random.randint(1, max_train_len)))
            example = f"{a}={len(a)}\n"
            if example in excluded:
                continue
            f.write(example)
            i += 1

    # 21-30 chars out of dist
    test_path = out_dir / "test_strlen_ood_21-30chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for _ in range(n_test):
            a = "".join(random.choices(CHARS, k=random.randint(21, 30)))
            f.write(f"{a}={len(a)}\n")

    # 31-40 chars out of dist
    test_path = out_dir / "test_strlen_ood_31-40chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for _ in range(n_test):
            a = "".join(random.choices(CHARS, k=random.randint(31, 40)))
            f.write(f"{a}={len(a)}\n")


def main():
    # generate_experiment_16(DATA_DIR / "matching" / "exp_16")
    generate_experiment_17(DATA_DIR / "strlen" / "exp_17")


if __name__ == "__main__":
    main()
