import lightning as L
from torch import nn
from torch.utils.data import DataLoader


def eval_sample(prompt: str, pred_answer: str, answer: str = None) -> bool:
    """Evaluate a single example, true if correct"""

    prompt = prompt.strip()
    if "=" in prompt:
        prompt = prompt.split("=")[0]
    if answer is None:
        answer = eval(prompt)
    # return str(answer) == pred_answer.strip()
    return answer.strip().startswith(pred_answer)


def eval_on_dataloader(
    model: nn.Module,
    test_dataloader: DataLoader,
    temperature: float,
    top_k: int,
    stop_token: int = None,
    max_new_tokens: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate on a dataloader and return results"""

    correct = 0
    total = 0
    for batch in test_dataloader:
        for prompt, answer in zip(*batch):
            if prompt.ndim == 1:
                prompt = prompt.unsqueeze(0)
            pred_answer = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token=stop_token,
                seed=seed,
            )
            correct += eval_sample(prompt, pred_answer)
            total += 1
    return {"correct": correct, "total": total, "accuracy": correct / total}


class ArithmeticTestCallback(L.Callback):
    def __init__(self, **kwargs):
        self.eval_params = kwargs

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        assert (
            len(trainer.test_dataloaders) == 1
        ), "Multiple test dataloaders unsupported for now"
        for test_dataloader in trainer.test_dataloaders:
            results = eval_on_dataloader(pl_module, test_dataloader, **self.eval_params)
            accuracy = results["accuracy"]
        for logger in trainer.loggers:
            logger.log_metrics({"test_acc": accuracy})
