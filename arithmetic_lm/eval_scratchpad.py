# %% [markdown]
# # Mistake analysis

# %%
import os
import random
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from Levenshtein import distance
from tqdm import tqdm

from arithmetic_lm.constants import CHECKPOINTS_DIR, PLOTS_DIR
from arithmetic_lm.formatting import scratchpad_addition
from arithmetic_lm.interp import plot_attn_maps
from arithmetic_lm.model import (
    TransformerDecoder,
    find_latest_ckpt,
    generate,
    load_model,
)
from arithmetic_lm.tokenizer import CharTokenizer

warnings.filterwarnings("ignore")

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# %%
tokenizer = CharTokenizer()
stop_token = tokenizer.encode("$")[0]

# %%
# load model
model_name = "trans_dec_6layers_768embd_4head_scratchpad_randsp0.5_ansloss"
ckpt_path = "checkpoints//addition-generalize-to-longer/trans_dec_6layers_768embd_4head_scratchpad_randsp0.5_ansloss/step20501-train_loss0.0056-val_loss0.0002.ckpt"
model, hparams = load_model(ckpt_path, map_location=DEVICE)
model.to(DEVICE)
model.eval()

# %%
print(hparams["extra_hparams"]["data_format"])
reverse_ops = hparams["extra_hparams"]["data_format"]["reverse_ops"]
reverse_ans = hparams["extra_hparams"]["data_format"]["reverse_ans"]


# %%
def get_carry_str(a: str, b: str, reverse: bool = False) -> str:
    """
    given a and b (non-reversed), return the carry string
    which contains:
        '.' if there is no carry at that position,
        'c' if there is current generated carry, but no carry from previous position
        'p' if there is carry from previous position, but no current generated carry
        'C' if there is both current generated carry and carry from previous position
    """

    carries = []
    carry = 0

    if reverse:
        a = a[::-1]
        b = b[::-1]

    for aa, bb in zip(a, b):
        aa = int(aa)
        bb = int(bb)
        s = aa + bb + carry
        if s >= 10:
            if carry == 1:
                carries.append("C")
            else:
                carries.append("c")
            carry = 1
        else:
            if carry == 1:
                carries.append("p")
            else:
                carries.append(".")
            carry = 0

    if carry == 1:
        carries.append("p")

    res = "".join(carries)

    if reverse:
        res = res[::-1]

    return res


# %%
def generate_with_shifted_pe(model, shift=-1, **generate_kwargs):
    """
    generate with shifted positional encoding
    """

    from copy import deepcopy

    model_copy = deepcopy(model)

    # shift positional encoding
    model_copy.pos_encoder.pe = model.pos_encoder.pe.roll(shift, 1)

    return generate(model_copy, **generate_kwargs)


# %%
def eval_scratchpad_example(true: str, pred: str) -> dict:
    """
    Evaluate a single scratchpad example, return dict with keys:
     - accuracy: 1 if the prediction is fully correct, 0 otherwise
     - edit_dist: levenshtein distance between true and pred
     - reverse_edit_dist: distance for reversing the operands part
     - partials_edit_dist: accuracy of the partial results (how many digits are correct)
     - answer_correct: 1 if the answer is correct, 0 otherwise
     - answer_edit_dist: levenshtein distance between true and pred answers
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
        true_partials = "".join(true.split(";")[1:]).split("|")[0]
        pred_partials = "".join(pred.split(";")[1:]).split("|")[0]
        partials_edit_dist = distance(true_partials, pred_partials)

        # split the answer
        true_ans = true.split("|")[1]
        pred_ans = pred.split("|")[1]
        answer_accuracy = 1 if true_ans == pred_ans else 0
        answer_edit_dist = distance(true_ans, pred_ans)
    except Exception as e:
        print(e)
        # format error, return -1 for distance
        rev_edit_dist = -1
        partials_edit_dist = -1
        answer_accuracy = 0
        answer_edit_dist = -1

    return {
        "accuracy": accuracy,
        "edit_dist": edit_dist,
        "reverse_edit_dist": rev_edit_dist,
        "partials_edit_dist": partials_edit_dist,
        "answer_accuracy": answer_accuracy,
        "answer_edit_dist": answer_edit_dist,
    }


# %%
scratchpad_eval_res = [
    {"n_digits": 18, "n_samples": 1000},
    {"n_digits": 19, "n_samples": 1000},
    {"n_digits": 20, "n_samples": 1000},
    {"n_digits": 21, "n_samples": 1000},
]


# Initialize result storage
results = {
    config["n_digits"]: {
        metric: []
        for metric in [
            "accuracy",
            "edit_dist",
            "reverse_edit_dist",
            "partials_edit_dist",
            "answer_accuracy",
            "answer_edit_dist",
        ]
    }
    for config in scratchpad_eval_res
}

# Evaluate on generated samples
for config in scratchpad_eval_res:
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

        prompt = f"${a}+{b}="
        prompt_idx = torch.tensor(tokenizer.encode(prompt, return_tensors=True)).to(
            DEVICE
        )

        pred_tensor = generate(
            model,
            idx=prompt_idx,
            max_new_tokens=256,
            stop_token=stop_token,
        )
        pred_ans = tokenizer.decode(pred_tensor[0])

        result = eval_scratchpad_example(true_ans, pred_ans)

        print(true_ans, pred_ans, result)

        if result["answer_edit_dist"] == -1:
            continue
        for metric in results[n_digits]:
            results[n_digits][metric].append(result[metric])

# Calculate mean and standard deviation
aggregated_results = {
    metric: {"mean": [], "std": []}
    for metric in [
        "accuracy",
        "edit_dist",
        "reverse_edit_dist",
        "partials_edit_dist",
        "answer_accuracy",
        "answer_edit_dist",
    ]
}

for n_digits, metrics in results.items():
    for metric, values in metrics.items():
        aggregated_results[metric]["mean"].append(np.mean(values))
        aggregated_results[metric]["std"].append(np.std(values))

# %%
# Plot results
x = np.arange(len(scratchpad_eval_res))  # the label locations
width = 0.4  # the width of the bars

names = {
    "accuracy": "Total accuracy",
    "edit_dist": "Total edit distance\n(Levenshtein)",
    "reverse_edit_dist": "Reversing operands\nedit distance",
    "partials_edit_dist": "Partial results\nedit distance",
    "answer_accuracy": "Answer accuracy",
    "answer_edit_dist": "Answer edit distance",
}

fig, axs = plt.subplots(1, len(aggregated_results), figsize=(14, 4))

for i, (metric, data) in enumerate(aggregated_results.items()):
    axs[i].bar(x, data["mean"], width, yerr=data["std"], capsize=5)
    axs[i].set_xlabel("Number of Digits")
    axs[i].set_title(names[metric])
    axs[i].set_xticks(x)
    axs[i].set_xticklabels([config["n_digits"] for config in scratchpad_eval_res])

fig.suptitle(f"Scratchpad Evaluation on model {model_name}")

fig.tight_layout()
plt.savefig(
    PLOTS_DIR / "gen_to_longer_rand_spaces" / f"{model_name}_scratchpad_eval.png"
)
plt.show()
