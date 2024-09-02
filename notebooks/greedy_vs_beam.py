# %% [markdown]
# # Mistake analysis
#
# Force models to output past end-of-sequence token (`$`) be modyfing its logit directly, or by shifting the positional encoding indices of the answer generated so far to force. Aim is to see if they can correctly predict the answer digits in OOD positions. If yes, then it could be that the position-dependent circuit decides to stop the generation by outputting the EOS token and not the addition circuit itself failing.

import os
import random

# %%
import re
import warnings
from copy import deepcopy
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import matplotlib.pyplot as plt
import numpy as np
import torch
from Levenshtein import distance
from tqdm import tqdm

from arithmetic_lm.constants import CHECKPOINTS_DIR, PLOTS_DIR
from arithmetic_lm.interp import plot_attn_maps
from arithmetic_lm.model import (
    TransformerDecoder,
    find_latest_ckpt,
    generate,
    load_model,
)
from arithmetic_lm.tokenizer import CharTokenizer

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore")

# %%
tokenizer = CharTokenizer()
stop_token = tokenizer.encode("$")[0]

# %%
# load model
model_name = "trans_dec_6layers_768embd_4head_randsp0.5_rev_ansloss"
ckpt_path = "checkpoints/addition-generalize-to-longer/trans_dec_6layers_768embd_4head_randsp0.5_rev_ansloss/step1000000-train_loss0.0002-val_loss0.0000.ckpt"
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


# %% [markdown]
# ## Greedy vs beam search

# %%
# evaluate model, see mistake distribution
beam_eval_res = [
    {
        "n_digits": 18,
        "n_samples": 100,
    },
    {
        "n_digits": 20,
        "n_samples": 100,
    },
    {
        "n_digits": 21,
        "n_samples": 100,
    },
]

methods = {
    "greedy": 0,
    "beam_5": 5,
    "beam_10": 10,
    "beam_20": 20,
}

for config in beam_eval_res:
    n_digits = config["n_digits"]
    n_samples = config["n_samples"]

    print(f"evaluating {n_digits} digits, {n_samples} samples")

    correct = {method: 0 for method in methods}
    levenshtein = {method: 0 for method in methods}

    for _ in tqdm(range(n_samples)):
        a = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)
        b = random.randint(10 ** (n_digits - 1), 10**n_digits - 1)
        true_ans = str(a + b)

        if reverse_ops:
            a = str(a)[::-1]
            b = str(b)[::-1]

        prompt = f"${a}+{b}="

        if reverse_ans:
            true_ans = true_ans[::-1]

        prompt_idx = torch.tensor(tokenizer.encode(prompt, return_tensors=True)).to(
            DEVICE
        )

        for method, n_beams in methods.items():
            pred_tensor = generate(
                model,
                idx=prompt_idx,
                max_new_tokens=25,
                stop_token=stop_token,
                n_beams=n_beams,
            )
            pred = tokenizer.decode(pred_tensor[0])

            carry_string = get_carry_str(a, b)

            pred_str = pred.replace("$", "")

            if pred_str == true_ans:
                correct[method] += 1
            else:
                levenshtein[method] += distance(pred_str, true_ans)

    config["accuracy"] = {method: correct[method] / n_samples for method in methods}
    config["levenshtein"] = {
        method: levenshtein[method] / n_samples for method in methods
    }

# %%
# Extract data
n_digits = [entry["n_digits"] for entry in beam_eval_res]
methods = list(beam_eval_res[0]["accuracy"].keys())

accuracy_data = {
    method: [entry["accuracy"][method] for entry in beam_eval_res] for method in methods
}
levenshtein_data = {
    method: [entry["levenshtein"][method] for entry in beam_eval_res]
    for method in methods
}

# Define bar width and positions
bar_width = 0.15
index = np.arange(len(n_digits))

# Plot accuracy vs. n_digits
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i, method in enumerate(methods):
    plt.bar(index + i * bar_width, accuracy_data[method], bar_width, label=method)
plt.xlabel("Number of Digits")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Number of Digits")
plt.xticks(index + bar_width * (len(methods) - 1) / 2, n_digits)
plt.legend()

# Plot levenshtein distances vs. n_digits
plt.subplot(1, 2, 2)
for i, method in enumerate(methods):
    plt.bar(index + i * bar_width, levenshtein_data[method], bar_width, label=method)
plt.xlabel("Number of Digits")
plt.ylabel("Levenshtein Distance")
plt.title("Average Levenshtein Distance vs. Number of Digits")
plt.xticks(index + bar_width * (len(methods) - 1) / 2, n_digits)
plt.legend()

plt.tight_layout()
plt.savefig(
    PLOTS_DIR
    / "gen_to_longer_rand_spaces"
    / f"{model_name}_greedy_vs_beam_decoding.png"
)
plt.show()
