"""See how much two datasets overlap."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate overlap between datasets.")
    parser.add_argument("train", help="path to train dataset")
    parser.add_argument("test", help="path to test dataset")
    args = parser.parse_args()

    with open(args.train, "r") as f:
        train_lines = f.readlines()
    with open(args.test, "r") as f:
        test_lines = f.readlines()

    train_set = set(train_lines)
    test_set = set(test_lines)

    n_overlap = len(train_set & test_set)

    print(f"Train set: {len(train_set)} lines")
    print(f"Test set: {len(test_set)} lines")
    print(f"Overlap: {n_overlap} lines")
    print(
        f"Overlap %: {n_overlap / len(test_set) * 100:.2f}% of test set, "
        f"{n_overlap / len(train_set) * 100:.2f}% of train set"
    )


if __name__ == "__main__":
    main()
