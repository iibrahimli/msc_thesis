"""Main training script"""

import os

import wandb.util

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import hydra
import lightning as L
import omegaconf
import torch

import wandb
from arithmetic_lm.constants import CHECKPOINTS_DIR, ROOT_DIR
from arithmetic_lm.dataset import (
    DATASET_CLASSES,
    ArithmeticExampleDataset,
    ArithmeticLMDataset,
    LightningArithmeticDataModule,
)
from arithmetic_lm.eval_utils import EVAL_FUNCS
from arithmetic_lm.model import MODELS
from arithmetic_lm.model.lightning_module import LightningModel
from arithmetic_lm.tokenizer import TOKENIZERS, Tokenizer
from arithmetic_lm.train_utils import LogAttnMapsCallback, SampleCallback
from arithmetic_lm.utils import set_seed


def train(
    run_name: str,
    cfg: omegaconf.DictConfig,
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    train_dataset: torch.utils.data.Dataset,
    test_data_dict: dict[str, torch.utils.data.Dataset],
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_iters: int,
    max_iters: int,
    num_dl_workers: int,
    val_ratio: float,
    val_interval: int,
    limit_val_batches: float | int,
    devices: list[int],
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str,
    grad_log_interval: int,
    gen_temp: float,
    gen_top_k: int,
    resume_ckpt_path: str | None = None,
    ckpt_weights_only: bool = False,
    eval_func: str = "numeric",
):
    """test_data_dict contains {'name': dataset}"""
    set_seed(42)

    full_resume_from_ckpt = resume_ckpt_path and not ckpt_weights_only

    # determine wandb run id
    if full_resume_from_ckpt:
        # use explicitly provided run id from cli
        wandb_run_id = cfg.wandb.get("run_id")

        if not wandb_run_id:
            # otherwise, try to get it from checkpoint
            ckpt_data = torch.load(resume_ckpt_path, map_location="cpu")
            wandb_run_id = ckpt_data["hyper_parameters"]["extra_hparams"].get(
                "wandb_run_id"
            )
            del ckpt_data

        if not wandb_run_id:
            raise ValueError(
                "Could not find wandb_run_id in checkpoint, please provide run_id manually"
            )
    else:
        # new run, generate new id
        wandb_run_id = wandb.util.generate_id()

    # test datasets
    test_ds_names = list(test_data_dict.keys())  # extract names to pass to lmodule
    test_datasets = list(test_data_dict.values())

    ldm = LightningArithmeticDataModule(
        train_dataset,
        test_datasets,
        tokenizer,
        batch_size,
        val_ratio=val_ratio,
        num_workers=num_dl_workers,
    )

    lmodel = LightningModel(
        model=model,
        tokenizer=tokenizer,
        test_dataloader_names=test_ds_names,
        lr=lr,
        weight_decay=weight_decay,
        warmup_iters=warmup_iters,
        model_hparams=omegaconf.OmegaConf.to_container(cfg.model.args, resolve=True),
        extra_hparams={
            "data_format": omegaconf.OmegaConf.to_container(
                cfg.data.format, resolve=True
            ),
            "wandb_run_id": wandb_run_id,
        },
        eval_func=EVAL_FUNCS[eval_func],
    )

    # run dir wandb_project / run_name or just run_name if no project
    run_dir = (
        CHECKPOINTS_DIR / wandb_project / run_name
        if wandb_project
        else CHECKPOINTS_DIR / run_name
    )
    run_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        save_last="link",
        dirpath=run_dir,
        filename="step{step}-train_loss{train_loss:.4f}-val_loss{val_loss:.4f}",
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_callback,
    ]

    loggers = []
    if wandb_enabled:

        # finish previous run if exists (e.g. hydra multirun)
        wandb.finish()

        wandb_logger = L.pytorch.loggers.WandbLogger(
            project=wandb_project,
            name=run_name,
            id=wandb_run_id,  # either a fresh one or previous run id
            save_dir=ROOT_DIR,
            log_model=True,
            entity=wandb_entity,
            resume="must" if full_resume_from_ckpt else "never",
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model, log_freq=grad_log_interval)

        # sampler and LR monitor callbacks
        gen_params = dict(
            temperature=gen_temp,
            top_k=gen_top_k,
            stop_token=tokenizer.encode("$")[0],
        )
        callbacks.extend(
            [
                SampleCallback(
                    n_samples=10, eval_func=EVAL_FUNCS[eval_func], **gen_params
                ),
                # TODO: re-enable after torch fix, or use odd number of heads
                # LogAttnMapsCallback(),
                L.pytorch.callbacks.LearningRateMonitor(),
            ]
        )

    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_steps=max_iters,
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=limit_val_batches,
        # limit_test_batches=N_TEST_BATCHES,
        log_every_n_steps=5,
        gradient_clip_val=1.0,
        devices=devices,
        # fast_dev_run=True,
    )

    if wandb_enabled:
        # add experiment hparams from omegaconf
        if trainer.global_rank == 0:
            wandb_logger.experiment.config.update(
                omegaconf.OmegaConf.to_container(cfg, resolve=True),
                allow_val_change=True,
            )

    # if load weights only
    if resume_ckpt_path and ckpt_weights_only:
        lmodel.load_state_dict(torch.load(resume_ckpt_path)["state_dict"])

    trainer.fit(
        lmodel,
        ldm,
        ckpt_path=(
            resume_ckpt_path if full_resume_from_ckpt else None
        ),  # if resume from full ckpt
    )


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: omegaconf.DictConfig):

    # tokenizer
    tokenizer = TOKENIZERS[cfg.tokenizer.name](**cfg.tokenizer.get("args"))

    # datasets
    train_ds_class = DATASET_CLASSES[cfg.data.get("train_ds_class")]
    if not train_ds_class:
        train_ds_class = (
            ArithmeticExampleDataset if cfg.data.format.encdec else ArithmeticLMDataset
        )
    ds_args = {
        "tokenizer": tokenizer,
        "seq_len": cfg.model.args.context_len,
        "pad": cfg.data.format.pad,
        "pad_ops_zero": cfg.data.format.pad_ops_zero,
        "pad_ans_zero": cfg.data.format.pad_ans_zero,
        "reverse_ops": cfg.data.format.reverse_ops,
        "reverse_ans": cfg.data.format.reverse_ans,
        "filler_tokens_prompt": cfg.data.format.filler_tokens_prompt,
        "filler_tokens_ans": cfg.data.format.filler_tokens_ans,
        "equal_in_prompt": not cfg.data.format.encdec,
        "chain_of_thought": cfg.data.format.get("chain_of_thought", False),
    }
    # TODO: add support for multiple train files
    train_dataset = train_ds_class(txtfile=cfg.data.train, **ds_args)
    test_data_dict = {
        n: ArithmeticExampleDataset(
            txtfile=f, limit_examples=cfg.training.limit_test_examples, **ds_args
        )
        for n, f in cfg.data.test.items()
    }
    # add a random subset of train dataset as test dataset to eval on it as well
    train_subset_ds = torch.utils.data.Subset(
        ArithmeticExampleDataset(txtfile=cfg.data.train, **ds_args),
        torch.randperm(len(train_dataset))[:100],
    )
    test_data_dict["train_subset"] = train_subset_ds

    # model
    model = MODELS[cfg.model.name](
        vocab_size=tokenizer.vocab_size, **cfg.model.get("args")
    )

    # train
    train(
        run_name=cfg.wandb.run_name,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        test_data_dict=test_data_dict,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        warmup_iters=cfg.training.warmup_iters,
        max_iters=cfg.training.max_iters,
        num_dl_workers=cfg.training.num_dl_workers,
        val_ratio=cfg.training.val_ratio,
        val_interval=cfg.training.val_interval,
        limit_val_batches=cfg.training.limit_val_batches,
        devices=cfg.training.devices,
        wandb_enabled=cfg.wandb.enabled,
        wandb_project=cfg.wandb.project,
        wandb_entity=cfg.wandb.entity,
        grad_log_interval=cfg.wandb.grad_log_interval,
        gen_temp=cfg.sampling.temp,
        gen_top_k=cfg.sampling.top_k,
        resume_ckpt_path=cfg.training.resume_ckpt_path,
        ckpt_weights_only=cfg.training.ckpt_weights_only,
        eval_func=cfg.training.eval_func,
    )


if __name__ == "__main__":
    main()
