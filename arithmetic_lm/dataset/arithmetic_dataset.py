from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import Dataset

from arithmetic_lm.formatting import format_line
from arithmetic_lm.tokenizer import Tokenizer


def _format_lines(format_func: callable, lines: list[str], **kwargs) -> list[str]:
    return list(map(partial(format_func, **kwargs), lines))


class ArithmeticTrainDataset(Dataset):
    """Concatenate lines in file and split into sequences of length seq_len"""

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
        # number of lines, not sequences (a seq contains many examples)
        self.n_examples = len(lines)
        # merge lines into one string
        text = "".join(lines)
        # keep seq_len * n_seq + 1 tokens (+1 to make target)
        tokens = self.tokenizer.encode(text)
        self.n_seq = (len(tokens) - 1) // seq_len
        self.tokens = torch.tensor(tokens[: self.n_seq * seq_len + 1], dtype=torch.long)

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        y = self.tokens[idx * self.seq_len + 1 : (idx + 1) * self.seq_len + 1]
        return x, y


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
            self.prompts.append(torch.tensor(self.tokenizer.encode(prompt + "=")))
            self.answers.append(torch.tensor(self.tokenizer.encode(ans)))

        assert len(self.prompts) == len(
            self.answers
        ), "prompts and answers length mismatch"

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tensor:
        return self.prompts[idx], self.answers[idx]


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
        self.test_ds_list = [test_ds] if isinstance(test_ds, Dataset) else test_ds
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
        for td in self.test_ds_list:
            dls.append(
                torch.utils.data.DataLoader(
                    td,
                    batch_size=None,  # disable automatic batching, return samples
                    shuffle=False,
                    pin_memory=True,
                    num_workers=self.num_workers,
                )
            )
        return dls
