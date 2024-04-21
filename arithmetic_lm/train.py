"""Main training script"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


import hydra
import lightning as L
import omegaconf
import torch

import wandb
from arithmetic_lm.constants import CHECKPOINTS_DIR, ROOT_DIR
from arithmetic_lm.dataset import (
    ArithmeticExampleDataset,
    ArithmeticLMDataset,
    LightningArithmeticDataModule,
)
from arithmetic_lm.model import MODELS
from arithmetic_lm.model.lightning_module import LightningModel
from arithmetic_lm.tokenizer import TOKENIZERS, Tokenizer
from arithmetic_lm.train_utils import SampleCallback
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
):
    """test_data_dict contains {'name': dataset}"""
    set_seed(42)

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
    )

    run_dir = CHECKPOINTS_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        dirpath=run_dir,
        filename="{step}-{train_loss:.4f}-{val_loss:.4f}",
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
            save_dir=ROOT_DIR,
            log_model=True,
            entity=wandb_entity,
        )
        loggers.append(wandb_logger)
        wandb_logger.watch(model, log_freq=grad_log_interval)

        # add experiment hparams from omegaconf
        wandb_logger.experiment.config.update(
            omegaconf.OmegaConf.to_container(cfg, resolve=True)
        )

        # sampler and LR monitor callbacks
        callbacks.extend(
            [
                SampleCallback(
                    n_samples=10,
                    temperature=gen_temp,
                    top_k=gen_top_k,
                    stop_token=tokenizer.encode("\n")[0],
                ),
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
    trainer.fit(lmodel, ldm)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: omegaconf.DictConfig):

    # tokenizer
    tokenizer = TOKENIZERS[cfg.tokenizer.name](**cfg.tokenizer.get("args"))

    # datasets
    train_ds_type = (
        ArithmeticExampleDataset if cfg.data.format.encdec else ArithmeticLMDataset
    )
    ds_args = {
        "tokenizer": tokenizer,
        "seq_len": cfg.model.args.context_len,
        "pad": cfg.data.format.pad,
        "reverse_ans": cfg.data.format.reverse_ans,
        "pad_ops_zero": cfg.data.format.pad_ops_zero,
        "pad_ans_zero": cfg.data.format.pad_ans_zero,
        "equal_in_prompt": not cfg.data.format.encdec,
    }
    # TODO: add support for multiple train files
    train_dataset = train_ds_type(txtfile=cfg.data.train, **ds_args)
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
    )


if __name__ == "__main__":
    main()
