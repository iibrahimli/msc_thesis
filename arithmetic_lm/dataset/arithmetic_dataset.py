from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset

from arithmetic_lm.formatting import format_line
from arithmetic_lm.tokenizer import Tokenizer


def _format_lines(format_func: callable, lines: list[str], **kwargs) -> list[str]:
    return list(map(partial(format_line, **kwargs), lines))


class ArithmeticDataset(Dataset):
    """Concatenate lines in file and split into sequences of length seq_len + 1 (for shifted targets)."""

    def __init__(
        self,
        txtfile: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        pad: str,
        reverse_ans: bool,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(txtfile, "r") as f:
            lines = f.readlines()
        lines = _format_lines(
            format_line,
            lines,
            pad=pad,
            reverse_ans=reverse_ans,
            prepend_newline=False,
        )
        self.n_examples = len(lines)
        # merge lines into one string
        text = "".join(lines)
        tokens = self.tokenizer.encode(text)
        # make seqs of same length (truncate if necessary)
        self.seqs = [
            tokens[i : i + seq_len + 1]
            for i in range(0, len(tokens) - seq_len, seq_len)
        ]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tensor:
        # return tensors
        return torch.tensor(self.seqs[idx])


class ArithmeticEvalDataset(Dataset):
    """Dataset but instead of pure language modeling, we want to evaluate each example (line)"""

    def __init__(
        self,
        txtfile: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        pad: str,
        reverse_ans: bool,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(txtfile, "r") as f:
            lines = f.readlines()
        self.n_examples = len(lines)
        lines = _format_lines(
            format_line,
            lines,
            pad=pad,
            reverse_ans=reverse_ans,
            prepend_newline=True,  # prompt with starting \n
        )
        self.prompts = []
        self.answers = []
        for line in lines:
            prompt, ans = line.split("=")
            self.prompts.append(self.tokenizer.encode(prompt))
            self.answers.append(self.tokenizer.encode(ans))

        assert len(self.prompts) == len(
            self.answers
        ), "prompts and answers length mismatch"

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tensor:
        # return tensors
        return torch.tensor(self.prompts[idx]), torch.tensor(self.answers[idx])

    def collate_fn(
        self, batch: list[tuple[Tensor, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        """Custom collate_fn to handle multiple tensors"""
        return batch


class LightningArithmeticDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_ds: Dataset,
        test_ds: Dataset | list[Dataset],
        tokenizer: Tokenizer,
        batch_size: int,
        val_ratio: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds,
            [1 - val_ratio, val_ratio],
            generator=torch.Generator().manual_seed(seed),
        )
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        dls = [
            torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
            ),
        ]
        tds = [self.test_ds] if isinstance(self.test_ds, Dataset) else self.test_ds
        for td in tds:
            dls.append(
                torch.utils.data.DataLoader(
                    td,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    collate_fn=td.collate_fn,
                    num_workers=self.num_workers,
                )
            )
        return dls

    # def test_dataloader(self):
    #     """
    #     Test dataloader returns batches of list of (prompt, answer) where prompt
    #     and answer are tensors.
    #     """
    #     return torch.utils.data.DataLoader(
    #         self.test_ds,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         pin_memory=True,
    #         collate_fn=self.test_ds.collate_fn,
    #         num_workers=self.num_workers,
    #     )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move batch to device since tensors are wrapped in lists"""
        if isinstance(batch, list):
            return [(x.to(device), y.to(device)) for x, y in batch]
        else:
            return batch.to(device)
