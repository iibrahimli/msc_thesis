"""Evaluation utility functions"""

import torch

from arithmetic_lm.tokenizer import Tokenizer


def eval_sample(pred_answer: str, answer: str, strict: bool = False) -> bool:
    """Evaluate a single example, true if correct, strict = whether to only compare digit chars"""

    if strict:
        return pred_answer.strip() == answer.strip()
    else:
        # HACK: if CoT, only look at part after last `=`
        if "," in answer and "=" in answer:
            answer = answer.split("=")[-1]
            pred_answer = pred_answer.split("=")[-1]
        # only compare digits
        pred_answer_digits = "".join(filter(str.isdigit, pred_answer))
        answer_digits = "".join(filter(str.isdigit, answer))
        return pred_answer_digits == answer_digits


def eval_on_batch(
    model, tokenizer: Tokenizer, batch: tuple[torch.Tensor, torch.Tensor], **gen_kwargs
) -> dict:
    """Returns dict"""
    prompt, answer = batch
    answer = answer.squeeze()
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    pred_answer = model.generate(prompt, max_new_tokens=answer.numel(), **gen_kwargs)
    if model.enc_dec:
        # enc-dec are prompted with `=` and don't return it, so remove it from `answer`
        answer = answer[0:]
    correct = int(
        eval_sample(
            tokenizer.decode(pred_answer.squeeze().tolist()),
            tokenizer.decode(answer.tolist()),
        )
    )

    return {"accuracy": correct}
