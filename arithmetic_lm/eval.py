"""Model evaluation script"""

from pathlib import Path

import lightning as L
import torch

from arithmetic_lm.constants import CHECKPOINTS_DIR, DATA_DIR
from arithmetic_lm.dataset import ArithmeticEvalDataset
from arithmetic_lm.model.nanogpt import LightningNanoGPT
from arithmetic_lm.tokenizer import CharTokenizer
from arithmetic_lm.utils import set_seed

SEQ_LEN = 256
BATCH_SIZE = 128
N_LAYERS = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2
LR = 0.001
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 0.1
WARMUP_ITERS = 100
MAX_ITERS = 5000
NUM_DL_WORKERS = 4

DEVICES = [0]  # only use one GPU


def evaluate(test_dataset: str | Path, ckpt_path: str | Path):
    set_seed(42)

    # tokenizer
    tokenizer = CharTokenizer()

    # test dataset
    test_ds = ArithmeticEvalDataset(test_dataset, tokenizer=tokenizer, seq_len=SEQ_LEN)

    print("test:", len(test_ds), "examples")

    test_dl = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1,
    )

    lmodel = LightningNanoGPT.load_from_checkpoint(ckpt_path)
    print("Loaded model from", ckpt_path)

    for i, batch in enumerate(test_dl):
        if i == 1:
            break
        for prompt_ids, ans_ids in batch:
            prompt = tokenizer.decode(prompt_ids.squeeze().tolist())
            answer = tokenizer.decode(ans_ids.squeeze().tolist())
            pred_answer = lmodel.generate(prompt_ids, max_new_tokens=ans_ids.numel())
            pred_answer = tokenizer.decode(pred_answer.squeeze().tolist())
            print(f"Prompt: {prompt}")
            print(f"Answer: {answer}")
            print(f"Pred Answer: {pred_answer}")


if __name__ == "__main__":
    ckpt_path = (
        CHECKPOINTS_DIR
        / "nanogpt_add_3digit_10k_bal_with_lr_sched/step=10-train_loss=73.6065-val_loss=79.6504.ckpt"
    )
    evaluate(
        test_dataset=DATA_DIR / "add_3digit_bal" / "add_3digit_10k_test.txt",
        ckpt_path=ckpt_path,
    )
