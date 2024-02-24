"""Training utility functions"""

import math
import random

import lightning as L
import torch

import wandb
from arithmetic_lm.eval_utils import eval_sample


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

    def __init__(self, n_samples: int = 5, **gen_kwargs):
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
        cols = ["dataset", "prompt", "answer", "pred_answer", "correct"]
        rows = []

        # save whether module is in train/eval
        m_training = pl_module.training
        pl_module.eval()
        for dset, prompt, ans in zip(dsets, prompts, answers):
            pred_ans = pl_module.generate(
                prompt, **self.gen_kwargs, max_new_tokens=ans.numel()
            )

            pred_ans_str = pl_module.tokenizer.decode(pred_ans.squeeze().tolist())
            ans_str = pl_module.tokenizer.decode(ans.squeeze().tolist())

            rows.append(
                [
                    dset,
                    pl_module.tokenizer.decode(prompt.squeeze().tolist()),
                    ans_str,
                    pred_ans_str,
                    eval_sample(pred_ans_str, ans_str),
                ]
            )
        pl_module.train(m_training)

        # generate monospace html for samples
        out = "<pre>"
        out += f"{'dataset':^14}|{'prompt':^15}|{'answer':^12}|{'pred_answer':^12}|{'correct':^3}\n"
        for row in rows:
            correct = " " if row[4] else "-"
            out += f"{row[0]:^14}{row[1]:^15}{row[2]:^12}{row[3]:^12}|{correct:^3}\n"
        out += "</pre>"

        trainer.logger.experiment.log(
            # {"samples": wandb.Table(columns=cols, data=rows)},
            {"samples": wandb.Html(out)},
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

        # train
        train_seq = trainer.datamodule.train_ds[
            random.randint(0, len(trainer.datamodule.train_ds) - 1)
        ]
        # decode and split by newline
        train_seq = pl_module.tokenizer.decode(train_seq.squeeze().tolist())
        # get rid of potentially incomplete lines
        train_samples = train_seq.split("\n")[1:-1]
        train_samples = random.sample(train_samples, self.n_samples)
        for sample in train_samples:
            prompt_str, ans_str = sample.split("=")
            prompt_str = prompt_str + "="
            ds_labels.append("train")
            prompts.append(
                torch.tensor(
                    pl_module.tokenizer.encode(prompt_str), device=pl_module.device
                )
            )
            answers.append(
                torch.tensor(
                    pl_module.tokenizer.encode(ans_str), device=pl_module.device
                )
            )

        # test
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
