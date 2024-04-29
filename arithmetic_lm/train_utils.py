"""Training utility functions"""

import math
import random

import lightning as L
import torch

import wandb
from arithmetic_lm.dataset.generate_addition import num_carry_ops
from arithmetic_lm.eval_utils import eval_sample
from arithmetic_lm.formatting import split_operands_and_op


def lr_cosine_annealing_with_warmup(
    it: int,
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float = None,
):
    if min_lr is None:
        min_lr = learning_rate / 10

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr

    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

    return min_lr + coeff * (learning_rate - min_lr)


class SampleCallback(L.Callback):
    """Sample from the model and log to wandb"""

    def __init__(self, n_samples: int = 10, **gen_kwargs):
        super().__init__()
        self.n_samples = n_samples
        self.gen_kwargs = gen_kwargs

    def _log(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        dsets: list[str],
        prompts: list[torch.Tensor],
        answers: list[torch.Tensor],
    ):
        cols = ["dataset", "num_carries", "prompt", "answer", "pred_answer", "correct"]
        rows = []

        for dset, prompt, ans in zip(dsets, prompts, answers):
            pred_ans = pl_module.generate(
                prompt, **self.gen_kwargs, max_new_tokens=ans.numel() + 1
            )

            prompt_str = repr(pl_module.tokenizer.decode(prompt.squeeze().tolist()))
            pred_ans_str = repr(pl_module.tokenizer.decode(pred_ans.squeeze().tolist()))
            ans_str = repr(pl_module.tokenizer.decode(ans.squeeze().tolist()))

            a, op, b = split_operands_and_op(prompt_str)
            assert (
                op.strip() == "+"
            ), f"Computing carries only supported for + for now, got op={op}"
            num_carries = num_carry_ops(int(a), int(b))

            rows.append(
                [
                    dset,
                    num_carries,
                    prompt_str,
                    ans_str,
                    pred_ans_str,
                    eval_sample(pred_ans_str, ans_str),
                ]
            )

        trainer.logger.experiment.log(
            {"samples": wandb.Table(columns=cols, data=rows)},
            # {"samples": wandb.Html(out)},
            step=trainer.global_step,
        )

    def on_validation_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ):

        ds_labels = []
        prompts = []
        answers = []

        # save whether module is in train/eval
        m_training = pl_module.training
        pl_module.eval()

        for ds_name, test_ds in zip(
            pl_module.test_dataloader_names, trainer.datamodule.test_ds_list
        ):
            test_idxs = random.sample(range(len(test_ds)), self.n_samples)
            for idx in test_idxs:
                prompt, ans = test_ds[idx]
                ds_labels.append(f"test_{ds_name}")
                prompts.append(prompt.to(pl_module.device))
                answers.append(ans.to(pl_module.device))

        self._log(trainer, pl_module, ds_labels, prompts, answers)

        # restore module training state
        pl_module.train(m_training)
