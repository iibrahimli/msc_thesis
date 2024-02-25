"""Evaluation utility functions"""

import torch

from arithmetic_lm.tokenizer import Tokenizer


def eval_sample(pred_answer: str, answer: str = None) -> bool:
    """Evaluate a single example, true if correct"""

    return pred_answer.strip() == answer.strip()


def eval_on_batch(
    model, tokenizer: Tokenizer, batch: tuple[torch.Tensor, torch.Tensor], **gen_kwargs
) -> dict:
    """Returns dict"""
    correct = 0
    total = 0

    prompt, answer = batch
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    pred_answer = model.generate(prompt, max_new_tokens=answer.numel(), **gen_kwargs)
    correct += eval_sample(
        tokenizer.decode(pred_answer.squeeze().tolist()),
        tokenizer.decode(answer.squeeze().tolist()),
    )
    total += 1

    return {"accuracy": correct / total, "correct": correct, "total": total}
