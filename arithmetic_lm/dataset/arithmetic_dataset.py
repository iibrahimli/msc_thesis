"""
Defines the datasets. All examples are stored as lines in txt files without any
formatting (e.g. `32+11=43`). The Datasets are responsible for loading and
appropriate formatting of the examples. The datasets return tokenized sequences.

- ArithmeticLMDataset: for language modeling, concatenates lines and splits into
    sequences of length `seq_len` (e.g. 256). The targets are next tokens.
- ArithmeticExampleDataset: for src -> target training/testing. For training,
    each example is one sequence src: `32+11` tgt: `=43` and the sequences are
    padded to the longest, batching is enabled. For evaluation no batching is
    used, where each example is evaluated separately. The prompt is the left-hand
    side of the equation and the answer is the right-hand side e.g. "32+11=43"
    -> "32+11=" and "43".
"""

from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from arithmetic_lm.formatting import format_line
from arithmetic_lm.tokenizer import Tokenizer


def _format_lines(format_func: callable, lines: list[str], **kwargs) -> list[str]:
    return list(map(partial(format_func, **kwargs), lines))


class ArithmeticLMDataset(Dataset):
    """Concatenate lines in file and split into sequences of length seq_len"""

    def __init__(
        self,
        txtfile: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        pad: str,
        reverse_ans: bool,
        **kwargs,
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
        self.n_tokens = len(self.tokens)

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        y = self.tokens[idx * self.seq_len + 1 : (idx + 1) * self.seq_len + 1]
        return x, y


class ArithmeticExampleDataset(Dataset):
    """Dataset but instead of pure language modeling, we want keep examples separate"""

    def __init__(
        self,
        txtfile: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        pad: str,
        reverse_ans: bool,
        equal_in_prompt: bool = True,
    ):
        """
        Args:
            txtfile: path to txt file
            tokenizer: tokenizer
            seq_len: sequence length
            pad: padding token
            reverse_ans: whether to reverse the answer
            equal_in_prompt: whether to include the `=` in the prompt or answer
        """
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
            if equal_in_prompt:
                prompt += "="
            else:
                ans = "=" + ans
            self.prompts.append(torch.tensor(self.tokenizer.encode(prompt)))
            self.answers.append(torch.tensor(self.tokenizer.encode(ans)))
        self.n_tokens = sum(len(p) + len(a) for p, a in zip(self.prompts, self.answers))

        assert len(self.prompts) == len(
            self.answers
        ), "prompts and answers length mismatch"

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tensor:
        return self.prompts[idx], self.answers[idx]

    def collate_fn(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """
        Pads sequences to longest in batch. Not used for evaluation, only for
        training encoder-decoder models.
        """
        pad_val = self.tokenizer.pad_token_id
        prompts, answers = zip(*batch)
        prompts = pad_sequence(prompts, batch_first=True, padding_value=pad_val)
        answers = pad_sequence(answers, batch_first=True, padding_value=pad_val)
        return prompts, answers


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
        self.train_val_collate_fn = (
            train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None
        )
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
        assert self.train_val_collate_fn is not None, "collate_fn required for training"
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.train_val_collate_fn,
        )

    def val_dataloader(self):
        dls = [
            torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=self.train_val_collate_fn,
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
