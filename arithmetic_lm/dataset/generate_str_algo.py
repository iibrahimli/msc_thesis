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


def single_str_generator(
    min_len: int,
    max_len: int,
    n_examples: int,
    ans_gen_func: callable,
    prompt_gen_func: callable = None,
    prompt_ans_sep: str = "=",
    exclude: set[str] | None = None,
):
    """Generator for simple string functions"""

    count = 0
    while count < n_examples:
        s = "".join(random.choices(CHARS, k=random.randint(min_len, max_len)))
        prompt = s if prompt_gen_func is None else prompt_gen_func(s)
        ans = ans_gen_func(prompt)
        example = f"{prompt}{prompt_ans_sep}{ans}\n"
        if exclude is not None and example in exclude:
            continue
        yield example
        count += 1


def generate_strlen_v1_1M(out_dir: str | Path):
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
        for example in single_str_generator(
            1, max_train_len, n_train, lambda a: len(a)
        ):
            f.write(example)

    excluded = get_set_from_file(train_path)

    # generate test datasets
    n_test = 2000
    # in dist
    test_path = out_dir / "test_strlen_in_dist_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(
            1, max_train_len, n_test, lambda a: len(a), exclude=excluded
        ):
            f.write(example)

    # 21-30 chars out of dist
    test_path = out_dir / "test_strlen_ood_21-30chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(21, 30, n_test, lambda a: len(a)):
            f.write(example)

    # 31-40 chars out of dist
    test_path = out_dir / "test_strlen_ood_31-40chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(31, 40, n_test, lambda a: len(a)):
            f.write(example)


def generate_strindex_v1(out_dir: str | Path):
    """
    String character retrieval by index. 2M training examples of 1-25 string length,
    test on in-distribution 1-25 and OOD 25-30 and 31-40. e.g. prompt
    "somestringhere[2]" to answer "m".
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f" > Generating data for Experiment 18 to {out_dir}")

    def prompt_func(s: str) -> str:
        idx = random.randint(0, len(s) - 1)
        return f"{s}[{idx}]"

    def ans_func(prompt: str) -> str:
        s, idx = prompt.split("[")
        idx = int(idx[:-1])
        return s[idx]

    # generate train dataset
    n_train = 2_000_000
    max_train_len = 25
    train_path = out_dir / "train_strindex_1-25chars_2M.txt"
    print(f"Generating {train_path}")
    with open(train_path, "w") as f:
        for example in single_str_generator(
            1,
            max_train_len,
            n_train,
            ans_gen_func=ans_func,
            prompt_gen_func=prompt_func,
        ):
            f.write(example)

    excluded = get_set_from_file(train_path)

    # generate test datasets
    n_test = 2000
    # in dist
    test_path = out_dir / "test_strindex_in_dist_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(
            1,
            max_train_len,
            n_test,
            ans_gen_func=ans_func,
            prompt_gen_func=prompt_func,
            exclude=excluded,
        ):
            f.write(example)

    # 25-30 chars out of dist
    test_path = out_dir / "test_strindex_ood_25-30chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(
            25, 30, n_test, ans_func, prompt_gen_func=prompt_func
        ):
            f.write(example)

    # 31-40 chars out of dist
    test_path = out_dir / "test_strindex_ood_31-40chars_2000.txt"
    print(f"Generating {test_path}")
    with open(test_path, "w") as f:
        for example in single_str_generator(
            31, 40, n_test, ans_func, prompt_gen_func=prompt_func
        ):
            f.write(example)


def main():
    # generate_experiment_16(DATA_DIR / "matching" / "exp_16")
    generate_strlen_v1_1M(DATA_DIR / "strlen_v1" / "1M")
    generate_strindex_v1(DATA_DIR / "strindex_v1" / "2M")


if __name__ == "__main__":
    main()
