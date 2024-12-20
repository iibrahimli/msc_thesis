"""Evaluation utility functions"""

from functools import partial

import torch
from Levenshtein import distance

from arithmetic_lm.tokenizer import Tokenizer


def eval_sample_numeric(pred_answer: str, answer: str, strict: bool = False) -> bool:
    """Evaluate a single example, true if correct, strict = whether to only compare digit chars"""

    if strict:
        return pred_answer.strip() == answer.strip()
    else:
        # HACK: if CoT, only look at part after last `=`
        if "," in answer and ";" in answer and "|" in answer and "=" in answer:
            answer = answer.split("=")[-1]
            pred_answer = pred_answer.split("=")[-1]
        # only compare digits
        pred_answer_digits = "".join(filter(str.isdigit, pred_answer))
        answer_digits = "".join(filter(str.isdigit, answer))
        return pred_answer_digits == answer_digits


def eval_sample_string_match(
    pred_answer: str, answer: str, strict: bool = False
) -> bool | float:
    """Evaluate by full string match"""
    if strict:
        return pred_answer.strip() == answer.strip()
    else:
        # check how many chars match, return float
        n_correct = sum(1 for a, b in zip(pred_answer, answer) if a == b)
        return n_correct / len(answer)


def edit_distance_ratio(pred_answer: str, answer: str) -> float:
    """Return 1 - edit distance / max len, so 1 if equal, 0 if completely different"""

    return 1 - distance(pred_answer, answer) / max(len(pred_answer), len(answer))


def compute_loss_on_prompt_and_answer(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    answer: torch.Tensor,
) -> float:
    """Compute loss on prompt and answer"""
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    if answer.ndim == 1:
        answer = answer.unsqueeze(0)
    assert prompt.size(0) == answer.size(0) == 1, "Only supports batch size 1"
    seq = torch.cat([prompt, answer], dim=1)
    logits = model(seq)
    # take only the logits for the answer
    logits = logits[:, prompt.numel() - 1 : -1]
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), answer.reshape(-1)
    )
    return loss


def eval_on_batch(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch: tuple[torch.Tensor, torch.Tensor],
    eval_func: callable = eval_sample_numeric,
    **gen_kwargs
) -> dict:
    """Returns dict"""
    # compute accuracy
    prompt, answer = batch
    answer = answer.squeeze()
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    pred_answer = model.generate(prompt, max_new_tokens=answer.numel(), **gen_kwargs)
    if model.enc_dec:
        # enc-dec are prompted with `=` and don't return it, so remove it from `answer`
        answer = answer[0:]
        tokenizer.decode(pred_answer.squeeze().tolist()),
    correct = eval_func(
        tokenizer.decode(pred_answer.squeeze().tolist()),
        tokenizer.decode(answer.tolist()),
    )
    if isinstance(correct, bool):
        correct = int(correct)

    # compute loss
    with torch.inference_mode():
        loss = compute_loss_on_prompt_and_answer(model, prompt, answer)

    return {"accuracy": correct, "loss": loss.item()}


EVAL_FUNCS = {
    "numeric": eval_sample_numeric,
    "string_match": eval_sample_string_match,
    "string_match_exact": partial(eval_sample_string_match, strict=True),
    "edit_distance_ratio": edit_distance_ratio,
}
