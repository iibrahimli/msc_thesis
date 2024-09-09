import argparse
import os
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
from arithmetic_lm.utils import get_carry_str

warnings.filterwarnings("ignore")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint",
    )
    args = ap.parse_args()

    DEVICE = args.device
    if "cuda" in DEVICE.lower() and not torch.cuda.is_available():
        DEVICE = "cpu"
        print("CUDA not available, using CPU")
    print(f"Using device: {DEVICE}")

    tokenizer = CharTokenizer()
    stop_token = tokenizer.encode("$")[0]

    # load model
    # ckpt_path = "checkpoints/addition-generalize-to-longer/trans_dec_6layers_768embd_4head_scratchpad_randsp0.5_ansloss/step64755-train_loss0.0001-val_loss0.0000.ckpt"
    # model_name = "trans_dec_6layers_768embd_4head_scratchpad_randsp0.5_ansloss"
    model_name = Path(args.ckpt_path).parent.name
    print(f"Model name: {model_name}")

    model, hparams = load_model(args.ckpt_path, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    print(hparams["extra_hparams"]["data_format"])
    reverse_ops = hparams["extra_hparams"]["data_format"]["reverse_ops"]
    reverse_ans = hparams["extra_hparams"]["data_format"]["reverse_ans"]

    # Define the evaluation configurations
    eval_configs = [
        {"n_digits": 17, "n_samples": 2},
        {"n_digits": 18, "n_samples": 2},
        # {"n_digits": 19, "n_samples": 2},
        # {"n_digits": 20, "n_samples": 2},
        # {"n_digits": 21, "n_samples": 2},
    ]

    # Initialize results DataFrame
    results = pd.DataFrame()

    # Evaluate on generated samples
    for config in eval_configs:
        n_digits = config["n_digits"]
        n_samples = config["n_samples"]

        print(f"Evaluating {n_digits} digits, {n_samples} samples")

        for _ in tqdm(range(n_samples)):
            a = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)
            b = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)

            a = str(a)
            b = str(b)
            if reverse_ops:
                a = a[::-1]
                b = b[::-1]

            true_ans = scratchpad_addition(a, b)
            # if reverse_ans:
            #     true_ans = true_ans[::-1]

            prompt = f"${a}+{b}="
            prompt_idx = torch.tensor(tokenizer.encode(prompt, return_tensors=True)).to(
                DEVICE
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
            result["n_carry"] = carry_str.count("c") + carry_str.count("C")
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)

    print(results.head())

    # Calculate aggregated results (ignore rows with -1)
    aggregated_results = (
        results[results != -1].groupby("n_digits").agg(["mean", "min", "max"])
    )

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

    fig, axs = plt.subplots(1, len(metrics) + 1, figsize=(24, 4))

    # plot metrics
    for i, metric in enumerate(metrics):
        data = aggregated_results[metric]
        x = range(len(data))

        axs[i].bar(
            x,
            data["mean"],
            width=0.4,
            yerr=[data["mean"] - data["min"], data["max"] - data["mean"]],
            capsize=5,
        )
        axs[i].set_xlabel("Number of Digits")
        axs[i].set_ylabel(names[metric])
        axs[i].set_title(names[metric])
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(data.index)

        if "accuracy" in metric:
            axs[i].set_ylim(0, 1)
        else:
            axs[i].set_ylim(0, None)

    # plot accuracy vs carry
    data = results.groupby("n_carry")["accuracy"].mean()
    mins = results.groupby("n_carry")["accuracy"].min()
    maxs = results.groupby("n_carry")["accuracy"].max()
    x = range(len(data))
    axs[-1].bar(x, data, width=0.4, yerr=[data - mins, maxs - data], capsize=5)
    axs[-1].set_xlabel("Number of Carry Operations")
    axs[-1].set_ylabel("Total accuracy\n(scratchpad and answer)")
    axs[-1].set_title("Total accuracy vs.\nnumber of carries")
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(data.index)
    axs[-1].set_ylim(0, 1)

    fig.suptitle(f"Scratchpad Evaluation on model {model_name}")

    plt.subplots_adjust(wspace=0.5)

    fig.tight_layout()
    plt.savefig(
        PLOTS_DIR / "gen_to_longer_rand_spaces" / f"{model_name}_scratchpad_eval.png",
        dpi=300,
        bbox_inches="tight",
    )
    # plt.show()


if __name__ == "__main__":
    main()
