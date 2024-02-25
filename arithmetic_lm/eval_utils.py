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
    prompt, answer = batch
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    pred_answer = model.generate(prompt, max_new_tokens=answer.numel(), **gen_kwargs)
    correct = int(
        eval_sample(
            tokenizer.decode(pred_answer.squeeze().tolist()),
            tokenizer.decode(answer.squeeze().tolist()),
        )
    )

    return {"accuracy": correct}
