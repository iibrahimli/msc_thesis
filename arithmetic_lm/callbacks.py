import random

import lightning as L
import matplotlib.pyplot as plt
import torch

import wandb
from arithmetic_lm.dataset.generate_addition import num_carry_ops
from arithmetic_lm.eval_utils import eval_sample_numeric
from arithmetic_lm.formatting import split_operands_and_op
from arithmetic_lm.interp import get_attn_maps_fig_for_model


class SampleCallback(L.Callback):
    """Sample from the model and log to wandb"""

    def __init__(
        self,
        n_samples: int = 10,
        eval_func: callable = eval_sample_numeric,
        **gen_kwargs,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.gen_kwargs = gen_kwargs
        self.eval_func = eval_func
        self.is_numeric = True

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

            # HACK remove random spaces from prompt if any
            prompt_str = prompt_str.replace(" ", "")

            # HACK if non-numeric addition
            if not self.is_numeric:
                num_carries = None
            elif any(c.isalpha() for c in prompt_str):
                self.is_numeric = False
                num_carries = None
            else:
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
                    self.eval_func(pred_ans_str, ans_str),
                ]
            )

        trainer.logger.experiment.log(
            {"samples": wandb.Table(columns=cols, data=rows)},
            # {"samples": wandb.Html(out)},
            # step=trainer.global_step,
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


class LogAttnMapsCallback(L.Callback):

    def __init__(self):
        super().__init__()
        self.prompts = {}

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # sample prompts from test dataloaders if not already done
        if not self.prompts:
            for ds_name, test_ds in zip(
                pl_module.test_dataloader_names, trainer.datamodule.test_ds_list
            ):
                # sample 1 per test dataset
                test_idxs = random.sample(range(len(test_ds)), 1)
                for idx in test_idxs:
                    prompt, _ = test_ds[idx]
                    self.prompts[ds_name] = prompt

        # save whether module is in train/eval
        m_training = pl_module.training
        pl_module.eval()

        figs = {}
        for ds_name, prompt in self.prompts.items():
            figs[ds_name] = get_attn_maps_fig_for_model(
                pl_module.model, pl_module.tokenizer, prompt
            )

        trainer.logger.experiment.log(
            {f"attn_maps/{ds_name}": wandb.Image(fig) for ds_name, fig in figs.items()}
        )

        # close all figures
        for fig in figs.values():
            plt.close(fig)

        # restore module training state
        pl_module.train(m_training)
