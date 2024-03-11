import lightning as L
import torch
from torch import Tensor, nn

from arithmetic_lm.eval_utils import eval_on_batch
from arithmetic_lm.tokenizer import Tokenizer
from arithmetic_lm.train_utils import lr_cosine_annealing_with_warmup

from .utils import generate


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        test_dataloader_names: list,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        warmup_iters: int = 100,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.tokenizer = tokenizer
        self.test_dataloader_names = test_dataloader_names
        # whether is encoder-decoder model
        self.enc_dec = model.enc_dec if hasattr(model, "enc_dec") else False
        self.save_hyperparameters(
            ignore=[
                "model",
                "lr",
                "betas",
                "weight_decay",
                "warmup_iters",
                "tokenizer",
                "test_dataloader",
                "enc_dec",
            ]
        )

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        return self.model(*x) if self.enc_dec else self.model(x[0])

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        # forward pass
        logits = self.forward(batch)
        # calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tensor | list, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Dataloader 0 is the val split, others are from test data (see self.test_dataloader_names)"""
        if dataloader_idx == 0:
            # evaluate language modeling loss on sequence
            x, y = batch
            logits = self.forward(batch)

            # shift target to the left and add padding if encoder-decoder model
            # so e.g. decoder gets '=123$' and target is '123$'
            if self.enc_dec:
                y[:, :-1] = y[:, 1:]
                y[:, -1] = self.tokenizer.pad_token_id

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.reshape(-1)
            )
            self.log("val_loss", loss, prog_bar=True, add_dataloader_idx=False)
            return loss
        else:
            # evaluate accuracy on TEST set (other val dataloaders than 0)
            res = eval_on_batch(
                self, self.tokenizer, batch, stop_token=self.tokenizer.encode("\n")[0]
            )
            # index - 1 coz 0 is val
            test_dl_name = self.test_dataloader_names[dataloader_idx - 1]
            self.log_dict(
                {
                    f"test_acc_{test_dl_name}": res["accuracy"],
                },
                batch_size=1,  # since ArithmeticEvalDataset returns 1 example
                add_dataloader_idx=False,
                prog_bar=True,
            )

    def configure_optimizers(self):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        use_fused = torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=1, betas=self.betas, **extra_args
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda i: lr_cosine_annealing_with_warmup(
                i,
                self.lr,
                warmup_iters=self.warmup_iters,
                lr_decay_iters=self.trainer.max_steps,
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def param_count(self) -> int:
        return self.model.param_count()

    @torch.inference_mode()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        stop_token: int = None,
        seed: int = 42,
    ) -> Tensor:
        encoder_source = None
        if self.enc_dec:
            encoder_source = idx
            idx = torch.tensor(self.tokenizer.encode("="), device=idx.device)
        return generate(
            model=self.model,
            idx=idx,
            max_new_tokens=max_new_tokens,
            encoder_source=encoder_source,
            temperature=temperature,
            top_k=top_k,
            stop_token=stop_token,
            seed=seed,
        )
