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


def eval_on_dataloader(
    model: nn.Module,
    test_dataloader: DataLoader,
    tokenizer: Tokenizer,
    limit: int,
    **gen_kwargs
) -> dict:
    """Evaluate on a dataloader and return results"""

    correct = 0
    total = 0
    for i, batch in enumerate(test_dataloader):
        if i == limit:
            break
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
    return {"correct": correct, "total": total, "accuracy": correct / total}


class ArithmeticTestCallback(L.Callback):
    def __init__(self, limit: int = 10, **kwargs):
        self.limit = limit
        self.gen_kwargs = kwargs

    @torch.inference_mode
    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        results = eval_on_dataloader(
            pl_module,
            trainer.datamodule.test_dataloader(),
            trainer.datamodule.tokenizer,
            self.limit,
            **self.gen_kwargs
        )
        accuracy = results["accuracy"]
        for logger in trainer.loggers:
            logger.log_metrics({"test_acc": accuracy})
