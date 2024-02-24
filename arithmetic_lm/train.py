"""Main training script"""

from pathlib import Path

import lightning as L

from arithmetic_lm.constants import CHECKPOINTS_DIR, DATA_DIR, ROOT_DIR
from arithmetic_lm.dataset import (
    ArithmeticDataset,
    ArithmeticEvalDataset,
    LightningArithmeticDataModule,
)
from arithmetic_lm.model.nanogpt import LightningNanoGPT
from arithmetic_lm.tokenizer import CharTokenizer
from arithmetic_lm.train_utils import SampleCallback
from arithmetic_lm.utils import set_seed

# example formatting
PAD = "$"
REVERSE_ANS = True

# model
SEQ_LEN = 256
BATCH_SIZE = 128
N_LAYERS = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2
LR = 0.001
BETAS = (0.9, 0.99)
WEIGHT_DECAY = 0.1

# training
WARMUP_ITERS = 100
MAX_ITERS = 10_000
NUM_DL_WORKERS = 4
VAL_INTERVAL = 100
VAL_RATIO = 0.1
LIMIT_VAL_BATCHES = 5  # also test batches
DEVICES = [0]  # only use one GPU

# sampling
GEN_TEMP = 0.8
GEN_TOP_K = 1

# wandb
WANDB = True
WANDB_PROJECT = "msc-thesis-pilot"
RUN_NAME = "exp1_nanogpt_1-3digit_overfit"


def train(train_data_path: str | Path, test_data_dict: dict, run_name: str):
    """test_data_dict contains {'name': dataset}"""
    set_seed(42)

    # tokenizer
    tokenizer = CharTokenizer()

    # 10k balanced dataset
    train_val_ds = ArithmeticDataset(
        train_data_path,
        tokenizer=tokenizer,
        seq_len=SEQ_LEN,
        pad=PAD,
        reverse_ans=REVERSE_ANS,
    )

    # test datasets
    test_ds_names = list(test_data_dict.keys())  # extract names to pass to lmodule
    test_ds_paths = list(test_data_dict.values())
    test_ds_list = [
        ArithmeticEvalDataset(
            test_path,
            tokenizer=tokenizer,
            seq_len=SEQ_LEN,
            pad=PAD,
            reverse_ans=REVERSE_ANS,
        )
        for test_path in test_ds_paths
    ]

    ldm = LightningArithmeticDataModule(
        train_val_ds,
        test_ds_list,
        tokenizer,
        BATCH_SIZE,
        val_ratio=VAL_RATIO,
        num_workers=NUM_DL_WORKERS,
    )
    del train_val_ds

    lmodel = LightningNanoGPT(
        tokenizer=tokenizer,
        test_dataloader_names=test_ds_names,
        context_len=SEQ_LEN,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        vocab_size=tokenizer.vocab_size,
        dropout=DROPOUT,
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
        warmup_iters=WARMUP_ITERS,
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
        L.pytorch.callbacks.LearningRateMonitor(),
    ]

    loggers = []
    if WANDB:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project=WANDB_PROJECT, name=run_name, save_dir=ROOT_DIR, log_model=True
        )
        loggers.append(wandb_logger)

        # add experiment hparams that are not in the lightning module
        wandb_logger.experiment.config.update(
            {
                "train_dataset": train_data_path,
                "test_datasets": test_data_dict,
                "max_iters": MAX_ITERS,
                "limit_val_batches": LIMIT_VAL_BATCHES,
                "val_ratio": VAL_RATIO,
                "pad": PAD,
                "reverse_ans": REVERSE_ANS,
            }
        )

        # sampler callback
        callbacks.append(
            SampleCallback(
                n_samples=3,
                temperature=GEN_TEMP,
                top_k=GEN_TOP_K,
                stop_token=tokenizer.encode("\n"),
            )
        )

    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        max_steps=MAX_ITERS,
        val_check_interval=VAL_INTERVAL,
        check_val_every_n_epoch=None,
        limit_val_batches=LIMIT_VAL_BATCHES,
        # limit_test_batches=N_TEST_BATCHES,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        devices=DEVICES,
        # fast_dev_run=True,
    )
    trainer.fit(lmodel, ldm)


if __name__ == "__main__":
    exp_dir = DATA_DIR / "experiment_1"
    train(
        train_data_path=exp_dir / "train_add_1-3digit.txt",
        test_data_dict={
            # f"{i}digit": exp_dir / f"test_{i}digit_100.txt" for i in range(1, 4 + 1)
            "train": exp_dir
            / "train_add_1-3digit.txt",
        },
        run_name=RUN_NAME,
    )
