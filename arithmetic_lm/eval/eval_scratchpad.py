import argparse
import math
import multiprocessing as mp
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from Levenshtein import distance
from tqdm import tqdm

from arithmetic_lm.constants import CHECKPOINTS_DIR, PLOTS_DIR
from arithmetic_lm.formatting import scratchpad_addition
from arithmetic_lm.model import generate, load_model
from arithmetic_lm.tokenizer import CharTokenizer
from arithmetic_lm.utils import get_carry_str, set_seed

warnings.filterwarnings("ignore")

# plt.style.use("../figure.mplstyle")


def eval_scratchpad_example(true: str, pred: str) -> dict:
    """
    Evaluate a single scratchpad example, return dict with evaluation metrics.
    """
    # remove $
    true = true.replace("$", "")
    pred = pred.replace("$", "")

    accuracy = 1 if true == pred else 0
    edit_dist = distance(true, pred)

    try:
        # split the reverse part
        true_rev = true.split(";")[0]
        pred_rev = pred.split(";")[0]
        # remove spaces
        true_rev = true_rev.replace(" ", "")
        pred_rev = pred_rev.replace(" ", "")
        rev_edit_dist = distance(true_rev, pred_rev)

        # split the partial results
        true_partials = ";".join(true.split(";")[1:]).split("|")[0]
        pred_partials = ";".join(pred.split(";")[1:]).split("|")[0]
        partials_edit_dist = distance(true_partials, pred_partials)

        # split the answer
        true_ans = true.split("|")[1]
        pred_ans = pred.split("|")[1]
        answer_accuracy = 1 if true_ans == pred_ans else 0
        answer_edit_dist = distance(true_ans, pred_ans)

        # check if the answer is correct wrt partials
        answer_from_partials = "".join([p[-5] for p in pred_partials.split(";")])[::-1]
        if pred_partials[-1] == "1":
            answer_from_partials = "1" + answer_from_partials
        answer_accuracy_wrt_partials = int(answer_from_partials == true_ans)

    except Exception as e:
        # format error, return -1 for distance
        rev_edit_dist = -1
        partials_edit_dist = -1
        answer_accuracy_wrt_partials = 0
        answer_accuracy = 0
        answer_edit_dist = -1

    return {
        "accuracy": accuracy,
        "edit_dist": edit_dist,
        "reverse_edit_dist": rev_edit_dist,
        "partials_edit_dist": partials_edit_dist,
        "answer_accuracy_wrt_partials": answer_accuracy_wrt_partials,
        "answer_accuracy": answer_accuracy,
        "answer_edit_dist": answer_edit_dist,
    }


def evaluate_config(config, model, tokenizer, device, reverse_ops, reverse_ans):
    n_digits = config["n_digits"]
    n_samples = config["n_samples"]

    results = []
    stop_token = tokenizer.encode("$")[0]

    for _ in tqdm(range(n_samples), desc=f"Evaluating {n_digits} digits"):
        a = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)
        b = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)

        a = str(a)
        b = str(b)
        if reverse_ops:
            a = a[::-1]
            b = b[::-1]

        true_ans = scratchpad_addition(a, b)

        prompt = f"${a}+{b}="
        prompt_idx = torch.tensor(tokenizer.encode(prompt, return_tensors=True)).to(
            device
        )

        carry_str = get_carry_str(a, b, reverse=not reverse_ops)

        pred_tensor = generate(
            model,
            idx=prompt_idx,
            max_new_tokens=512,
            stop_token=stop_token,
        )
        pred_ans = tokenizer.decode(pred_tensor[0])

        result = eval_scratchpad_example(true_ans, pred_ans)
        result["n_digits"] = n_digits
        result["carry_ratio"] = (carry_str.count("c") + carry_str.count("C")) / len(
            carry_str
        )
        results.append(result)

    return pd.DataFrame(results)


def worker_init(ckpt_path, device):
    global model, tokenizer, hparams
    model, hparams = load_model(ckpt_path, map_location=device, strict=False)
    model.to(device)
    model.eval()
    tokenizer = CharTokenizer()


def worker(config, device):
    return evaluate_config(
        config,
        model,
        tokenizer,
        device,
        hparams["extra_hparams"]["data_format"]["reverse_ops"],
        hparams["extra_hparams"]["data_format"]["reverse_ans"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)"
    )
    ap.add_argument(
        "--ckpt_path", type=str, default=None, help="Path to the model checkpoint"
    )
    ap.add_argument(
        "--n_samples",
        type=int,
        default=1337,
        help="Number of samples (per digit) to evaluate",
    )
    ap.add_argument(
        "--n_processes",
        type=int,
        default=1,
        help="Number of processes to use for evaluation",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = ap.parse_args()

    # Set random seed
    set_seed(args.seed)

    if "cuda" in args.device.lower() and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")
    print(f"Using device: {args.device}")

    model_name = Path(args.ckpt_path).parent.name
    print(f"Model name: {model_name}")

    eval_configs = [
        {"n_digits": 17, "n_samples": args.n_samples},
        {"n_digits": 18, "n_samples": args.n_samples},
        {"n_digits": 19, "n_samples": args.n_samples},
        {"n_digits": 20, "n_samples": args.n_samples},
        {"n_digits": 21, "n_samples": args.n_samples},
    ]

    # Initialize the pool with 2 processes and the global variables
    pool = mp.Pool(
        processes=args.n_processes,
        initializer=worker_init,
        initargs=(args.ckpt_path, args.device),
    )

    # Use pool.starmap to process configurations in order, 2 at a time
    results_list = pool.starmap(
        worker, [(config, args.device) for config in eval_configs]
    )

    # Combine results
    results = pd.concat(results_list, ignore_index=True)

    # save results
    results_path = (
        PLOTS_DIR
        / "gen_to_longer_rand_spaces"
        / f"{model_name}_scratchpad_eval_violin.csv"
    )
    results.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")

    # Close the pool
    pool.close()
    pool.join()

    # Plot results
    metrics = [
        "accuracy",
        "edit_dist",
        "reverse_edit_dist",
        "partials_edit_dist",
        "answer_accuracy_wrt_partials",
        "answer_accuracy",
        "answer_edit_dist",
    ]
    names = {
        "accuracy": "Total accuracy\n(scratchpad and answer)",
        "edit_dist": "Total edit distance\n(Levenshtein)",
        "reverse_edit_dist": "Reversing operands\nedit distance",
        "partials_edit_dist": "Partial results\nedit distance",
        "answer_accuracy_wrt_partials": "Answer accuracy\nwrt scratchpad",
        "answer_accuracy": "Answer accuracy",
        "answer_edit_dist": "Answer edit distance",
    }

    total_plots = len(metrics) + 1
    nrows = 2
    ncols = math.ceil(total_plots / 2)
    scale = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale))

    axs = axs.flatten()

    # plot metrics
    for i, metric in enumerate(metrics):
        sorted_n_digits = sorted(results["n_digits"].unique())
        grouped_data = [
            results[results["n_digits"] == n_digits][metric].values
            for n_digits in sorted_n_digits
        ]

        parts = axs[i].violinplot(
            grouped_data, showmeans=True, showextrema=True, showmedians=True
        )

        for pc in parts["bodies"]:
            pc.set_facecolor("lightblue")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        parts["cmeans"].set_color("red")
        parts["cmedians"].set_color("blue")

        axs[i].set_xlabel("Number of Digits")
        axs[i].set_ylabel(names[metric])
        axs[i].set_title(names[metric])
        axs[i].set_xticks(range(1, len(grouped_data) + 1))
        axs[i].set_xticklabels(sorted_n_digits)

        if "accuracy" in metric:
            axs[i].set_ylim(0, 1)
        else:
            axs[i].set_ylim(0, None)

    # plot accuracy vs carry ratio as scatter plot
    data = results[results["answer_accuracy"] != -1]
    axs[-1].scatter(data["carry_ratio"], data["answer_accuracy"], alpha=0.5)
    axs[-1].set_xlabel("Carry Ratio")
    axs[-1].set_ylabel("Answer Accuracy")
    axs[-1].set_title("Answer Accuracy vs Carry Ratio")

    fig.suptitle(f"Scratchpad Intermediate Steps Evaluation ({args.n_samples} samples)")

    plt.subplots_adjust(wspace=0.5)

    fig.tight_layout()
    fig_path = (
        PLOTS_DIR
        / "gen_to_longer_rand_spaces"
        / f"{model_name}_scratchpad_eval_violin.png"
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    main()
