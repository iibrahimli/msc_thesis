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
VAL_INTERVAL = 50
# N_TEST_BATCHES = 10

WANDB = True
WANDB_PROJECT = "msc-thesis-pilot"
RUN_NAME = "nanogpt_add_3digit_10k_bal_with_lr_sched"

DEVICES = [0]  # only use one GPU


def train(train_dataset: str | Path, test_dataset: str | Path, run_name: str):
    set_seed(42)

    # tokenizer
    tokenizer = CharTokenizer()

    # 10k balanced dataset
    train_val_ds = ArithmeticDataset(
        train_dataset, tokenizer=tokenizer, seq_len=SEQ_LEN
    )

    # test dataset
    test_ds = ArithmeticEvalDataset(test_dataset, tokenizer=tokenizer, seq_len=SEQ_LEN)

    print("train + val:", len(train_val_ds), "sequences")
    print("test:", len(test_ds), "examples")

    ldm = LightningArithmeticDataModule(
        train_val_ds,
        test_ds,
        tokenizer,
        BATCH_SIZE,
        val_ratio=0.2,
        num_workers=NUM_DL_WORKERS,
    )
    del train_val_ds

    lmodel = LightningNanoGPT(
        tokenizer=tokenizer,
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

    loggers = []
    if WANDB:
        wandb_logger = (
            L.pytorch.loggers.WandbLogger(
                project=WANDB_PROJECT, name=run_name, save_dir=ROOT_DIR, log_model=True
            ),
        )
        loggers.append(wandb_logger)
        # add experiment hparams
        wandb_logger.experiment.config.update(
            {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
            }
        )

    trainer = L.Trainer(
        logger=loggers,
        callbacks=[
            checkpoint_callback,
        ],
        max_steps=MAX_ITERS,
        val_check_interval=VAL_INTERVAL,
        check_val_every_n_epoch=None,
        # limit_test_batches=N_TEST_BATCHES,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        devices=DEVICES,
        # fast_dev_run=True,
    )
    trainer.fit(lmodel, ldm)


if __name__ == "__main__":
    train(
        train_dataset=DATA_DIR / "add_3digit_bal" / "add_3digit_10k_bal.txt",
        test_dataset=DATA_DIR / "add_3digit_bal" / "add_3digit_10k_test.txt",
        run_name=RUN_NAME,
    )
