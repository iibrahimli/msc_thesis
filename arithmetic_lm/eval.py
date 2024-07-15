"""
Evaluate a model checkpoint on generated arithmetic expressions.
"""

import argparse
import os
import random
from pathlib import Path
from pprint import pprint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from tqdm import tqdm

from arithmetic_lm.constants import EVAL_DIR
from arithmetic_lm.eval_utils import eval_sample_numeric
from arithmetic_lm.formatting import format_line
from arithmetic_lm.model import generate, load_model
from arithmetic_lm.tokenizer import CharTokenizer, Tokenizer


def evaluate(
    exp_name: str,
    model_name: str,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    data_format: dict,
    samples_per_case: int = 10,
    samples_per_case_off_diag: int = None,
    min_digits: int = 1,
    max_digits: int = 100,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> None:
    """
    Evaluate on long addition examples
    """

    wrong_examples = []

    if samples_per_case_off_diag is None:
        samples_per_case_off_diag = samples_per_case

    eval_dir = EVAL_DIR / exp_name / model_name
    eval_dir.mkdir(exist_ok=True, parents=True)

    reverse_ans = data_format["reverse_ans"]

    print(
        f"\nStarting evaluation, {samples_per_case} samples per case, "
        f"min_digits={min_digits}, max_digits={max_digits}"
    )

    acc = np.zeros(
        (max_digits - min_digits + 1, max_digits - min_digits + 1),
        dtype=np.float32,
    )

    for i in tqdm(range(min_digits, max_digits + 1), desc=f"i", position=0):
        for j in tqdm(
            range(min_digits, max_digits + 1), desc="j", position=1, leave=False
        ):
            wrong_example_saved = False
            n_samples = samples_per_case if i == j else samples_per_case_off_diag
            for _ in range(n_samples):
                # generate
                a = random.randint(10 ** (i - 1), 10**i - 1)
                b = random.randint(10 ** (j - 1), 10**j - 1)
                real_ans = str(a + b)

                # format
                prompt = f"{a}+{b}={real_ans}"
                prompt = format_line(prompt, **data_format)
                # remove answer from prompt
                prompt = prompt[: prompt.index("=") + 1]

                # tokenize
                tokens = torch.tensor(tokenizer.encode(prompt)).to(device)

                # generate
                stop_token = tokenizer.encode("$")[0]
                generated = generate(
                    model,
                    tokens,
                    max_new_tokens=len(real_ans) + 2,
                    stop_token=stop_token,
                )

                # decode
                pred_ans = tokenizer.decode(generated.cpu().detach()).strip()

                # reverse ans if needed
                if reverse_ans:
                    real_ans = real_ans[::-1]

                if verbose:
                    print(f"{prompt} -> {pred_ans} ({real_ans})")

                # check answer
                correct = eval_sample_numeric(pred_ans, real_ans)
                acc[i - min_digits, j - min_digits] += int(correct)

                # save incorrect samples (max of 1 per case)
                if not correct and not wrong_example_saved:
                    wrong_examples.append(
                        f"[{i}, {j}] {prompt} -> {pred_ans} ({real_ans})"
                    )
                    wrong_example_saved = True

    acc[i - min_digits, j - min_digits] /= samples_per_case

    print(f"Mean accuracy: {acc.mean()}")

    # save acc array
    acc_path = eval_dir / f"accuracy_{min_digits}-{max_digits}digits.npy"
    np.save(acc_path, acc)
    print(f"Saved accuracy array to {acc_path}")

    # save wrong examples
    wrong_examples_path = (
        eval_dir / f"wrong_examples_{min_digits}-{max_digits}digits.txt"
    )

    # plot
    plt.figure(figsize=(12, 8))
    plt.imshow(acc, cmap="viridis", interpolation="none")
    plt.colorbar()

    # flip y axis
    plt.gca().invert_yaxis()

    plt.title(f"Accuracy of {model_name} ({exp_name})")
    plt.xlabel("Number of digits in first operand")
    plt.ylabel("Number of digits in second operand")

    # set ticks
    plt.xticks(
        np.arange(max_digits - min_digits + 1), np.arange(min_digits, max_digits + 1)
    )
    plt.yticks(
        np.arange(max_digits - min_digits + 1), np.arange(min_digits, max_digits + 1)
    )

    # use max n locator
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    # rotate labels
    plt.xticks(rotation=90)
    plot_path = eval_dir / f"accuracy_{min_digits}-{max_digits}digits.png"
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")
    # plt.show()


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="checkpoint to evaluate"
    )
    parser.add_argument(
        "--min_digits", type=int, default=1, help="minimum number of digits"
    )
    parser.add_argument(
        "--max_digits", type=int, default=100, help="maximum number of digits"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="number of samples to generate per case"
    )
    parser.add_argument(
        "--n_off_diag",
        type=int,
        default=None,
        help="number of samples to generate per off-diagonal case (len(a) != len(b))",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to use (cpu or cuda), default is CPU",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print more information during evaluation",
    )

    args = parser.parse_args()

    # set random seed
    random.seed(args.seed)

    # set device
    device = torch.device(args.device)
    print(f"Device: {device}")

    # extract model name, directory that contains the .ckpt file
    model_name = Path(args.ckpt).parent.name
    exp_name = Path(args.ckpt).parent.parent.name
    print(f"Model name: {model_name}")
    print(f"Experiment name: {exp_name}")

    # load model
    model, hparams = load_model(args.ckpt, map_location=device)
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.ckpt}")
    print("\nModel hyperparameters:")
    pprint(hparams["model_hparams"])

    # load tokenizer
    tokenizer = CharTokenizer()

    # evaluate
    data_format_params = hparams["extra_hparams"]["data_format"]
    # remove "encdec" key from data_format_params
    data_format_params.pop("encdec", None)
    print("\nData format parameters:")
    pprint(data_format_params)

    evaluate(
        exp_name,
        model_name,
        model,
        tokenizer,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        data_format=data_format_params,
        samples_per_case=args.n,
        samples_per_case_off_diag=args.n_off_diag,
        device=device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
