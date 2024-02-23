import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from arithmetic_lm.tokenizer import Tokenizer


def eval_sample(prompt: str, pred_answer: str, answer: str = None) -> bool:
    """Evaluate a single example, true if correct"""

    if answer is None:
        prompt = prompt.strip()
        if "=" in prompt:
            prompt = prompt.split("=")[0]
        answer = eval(prompt)
    # return str(answer) == pred_answer.strip()
    return answer.strip().startswith(pred_answer)


def eval_on_batch(model, tokenizer, batch, **gen_kwargs) -> tuple[int, int]:
    """Returns correct, total"""
    correct = 0
    total = 0

    for prompt, answer in batch:
        if prompt.ndim == 1:
            prompt = prompt.unsqueeze(0).to(model.device)
        pred_answer = model.generate(prompt, **gen_kwargs)
        correct += eval_sample(
            tokenizer.decode(prompt.squeeze().tolist()),
            tokenizer.decode(pred_answer.squeeze().tolist()),
            tokenizer.decode(answer.squeeze().tolist()),
        )
        total += 1

    return {"accuracy": correct / total}
