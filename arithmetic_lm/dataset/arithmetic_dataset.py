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

import random
from functools import partial
from pathlib import Path

import lightning as L
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from arithmetic_lm.constants import TASK_PREFIX_LEN
from arithmetic_lm.formatting import format_line
from arithmetic_lm.tokenizer import Tokenizer


class DatasetBase(Dataset):
    """Child classes need to define _process_lines method"""

    def __init__(
        self,
        txtfile: str | Path,
        tokenizer: Tokenizer,
        seq_len: int,
        pad: str,
        reverse_ops: bool,
        reverse_ans: bool,
        pad_ops_zero: int | None = None,
        pad_ans_zero: int | None = None,
        filler_tokens_prompt: int | None = None,
        filler_tokens_ans: int | None = None,
        limit_examples: int | None = None,
        equal_in_prompt: bool = False,
        scratchpad: bool = False,
        operand_random_spaces_amount: int | float = 0,
        answer_random_spaces_amount: int | float = 0,
        index_hints: bool = False,
        use_task_prefix: bool = False,
    ):
        self.txtfile = txtfile
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.fmt_kwargs = dict(
            pad=pad,
            reverse_ops=reverse_ops,
            reverse_ans=reverse_ans,
            pad_ops_zero=pad_ops_zero,
            pad_ans_zero=pad_ans_zero,
            filler_tokens_prompt=filler_tokens_prompt,
            filler_tokens_ans=filler_tokens_ans,
            scratchpad=scratchpad,
            operand_random_spaces_amount=operand_random_spaces_amount,
            answer_random_spaces_amount=answer_random_spaces_amount,
            index_hints=index_hints,
            use_task_prefix=use_task_prefix,
        )
        self.limit_examples = limit_examples
        self.equal_in_prompt = equal_in_prompt
        self.multitask = use_task_prefix

        # load lines and process
        lines = self._get_lines()

        # check if task prefixes are there
        if use_task_prefix:
            assert all([l[:TASK_PREFIX_LEN].isalpha() for l in lines[:10]]), (
                "use_task_prefix=True, but not all of first 10 lines contain"
                f" task prefix: {lines[:10]}"
            )

        self._process_lines(lines)

    def _get_lines(self) -> list[str]:
        # read lines
        with open(self.txtfile, "r") as f:
            lines = f.readlines()
        if self.limit_examples is not None:
            lines = random.sample(lines, min(self.limit_examples, len(lines)))
        lines = self._format_lines(format_line, lines)
        # number of lines, not sequences (a seq contains many examples)
        self.n_examples = len(lines)
        return lines

    def _process_lines(self, lines: list[str]):
        "Do whatever with lines, assign fields etc."
        pass

    def _format_lines(self, format_func: callable, lines: list[str]) -> list[str]:
        # HACK decide if non-numeric task
        generic = False
        # not generic just because task prefixes are .isalpha
        if not self.multitask:
            for line in random.sample(lines, min(10, len(lines))):
                if any(c.isalpha() for c in line):
                    generic = True
                    break
        return list(
            map(partial(format_func, generic=generic, **self.fmt_kwargs), lines)
        )

    def reset(self):
        # NOTE Don't reset seed, this is actually for randomizing e.g. random spaces
        lines = self._get_lines()
        self._process_lines(lines)


class ArithmeticLMDataset(DatasetBase):
    """
    NOT USED ANYMORE
    Concatenate lines in file and split into sequences of length seq_len
    """

    def _process_lines(self, lines: list[str]):
        # merge lines into one string
        text = "".join(lines)
        # keep seq_len * n_seq + 1 tokens (+1 to make target)
        tokens = self.tokenizer.encode(text)
        self.n_seq = (len(tokens) - 1) // self.seq_len
        self.tokens = torch.tensor(
            tokens[: self.n_seq * self.seq_len + 1], dtype=torch.long
        )
        self.n_tokens = len(self.tokens)

    def __len__(self) -> int:
        return self.n_seq

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self.tokens[idx * self.seq_len : (idx + 1) * self.seq_len]
        y = self.tokens[idx * self.seq_len + 1 : (idx + 1) * self.seq_len + 1]
        return x, y


class ArithmeticLMSequenceDataset(DatasetBase):
    """LM dataset but with one example (prompt and answer together) per sequence"""

    def _process_lines(self, lines: list[str]):
        self.examples = [torch.tensor(self.tokenizer.encode(e)) for e in lines]
        self.n_tokens = sum(len(e) for e in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tensor:
        return self.examples[idx][:-1], self.examples[idx][1:]

    def collate_fn(self, batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        """
        Pads sequences to longest in batch. Used only for training.
        """
        pad_val = self.tokenizer.pad_token_id
        xs, ys = zip(*batch)
        xs = pad_sequence(xs, batch_first=True, padding_value=pad_val)
        ys = pad_sequence(ys, batch_first=True, padding_value=pad_val)
        return xs, ys


class ArithmeticExampleDataset(DatasetBase):
    """Dataset but instead of pure language modeling, we want keep prompts and answers separate"""

    def _process_lines(self, lines: list[str]):
        self.prompts = []
        self.answers = []
        for line in lines:
            prompt, ans = line.split("=", 1)
            if self.equal_in_prompt:
                prompt += "="
            else:
                ans = "=" + ans
            # HACK move filler tokens to prompt from ans (assume .)
            filler_tokens_ans = self.fmt_kwargs.get("filler_tokens_ans", 0)
            if filler_tokens_ans:
                ans = ans.replace(".", "")
                prompt += "." * filler_tokens_ans
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

    # HACK won't work if dataset does not have .reset() method
    # need a better solution for randomized processing steps
    def reset_datasets(self):
        for ds in [self.train_ds, self.val_ds] + self.test_ds_list:
            if hasattr(ds, "reset"):
                ds.reset()

    def train_dataloader(self):
        # reset datasets in dataloader methods, and pass
        # reload_dataloaders_every_n_epochs to the trainer in train.py
        self.reset_datasets()
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_val_collate_fn,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        dls = [
            torch.utils.data.DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.train_val_collate_fn,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=True,
            ),
        ]
        for td in self.test_ds_list:
            dls.append(
                torch.utils.data.DataLoader(
                    td,
                    batch_size=None,  # disable automatic batching, return samples
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=True if self.num_workers > 0 else False,
                    pin_memory=True,
                )
            )
        return dls
