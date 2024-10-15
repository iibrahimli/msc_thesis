import lightning as L
import torch
from torch import Tensor, nn

from arithmetic_lm.eval_utils import eval_on_batch, eval_sample_numeric
from arithmetic_lm.model import generate
from arithmetic_lm.model.utils import answer_mask, get_linear_lsd_weight
from arithmetic_lm.tokenizer import Tokenizer
from arithmetic_lm.train_utils import lr_cosine_annealing_with_warmup


class LightningModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        test_dataloader_names: list,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.1,
        warmup_iters: int = 100,
        model_hparams: dict = None,
        extra_hparams: dict = None,
        eval_func: callable = eval_sample_numeric,
        pause_token: str | None = None,
        only_answer_loss: bool = False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.tokenizer = tokenizer
        self.test_dataloader_names = test_dataloader_names
        self.eval_func = eval_func
        self.only_answer_loss = only_answer_loss
        self.equal_token_id = self.tokenizer.encode("=")[0]

        # pause token to be ignored
        self.pause_token_id: int | None = (
            self.tokenizer.encode(pause_token)[0] if pause_token else None
        )
        if pause_token:
            # compute class weights to ignore pause token in loss
            class_weights = torch.ones(self.tokenizer.vocab_size, device=self.device)
            class_weights[self.pause_token_id] = 0
            self.register_buffer("class_weights", class_weights)

        # whether is encoder-decoder model
        self.enc_dec = model.enc_dec if hasattr(model, "enc_dec") else False

        # save model class and hparams for instantiation as well
        self.model_class = model.__class__.__name__
        self.model_hparams = model_hparams
        self.model_hparams["vocab_size"] = tokenizer.vocab_size

        # save extra hparams
        self.extra_hparams = extra_hparams

        self.save_hyperparameters(
            ignore=[
                "model",
                "tokenizer",
                "test_dataloader_names",
                "lr",
                "betas",
                "weight_decay",
                "warmup_iters",
                "enc_dec",
                "only_answer_loss",
                "equal_token_id",
            ]
        )

    def forward(self, x: Tensor | tuple[Tensor, Tensor]) -> Tensor:
        if isinstance(x, Tensor):
            src = x
        else:
            src, tgt = x
        return self.model(src, tgt) if self.enc_dec else self.model(src)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        # forward pass
        logits = self.forward(batch)

        # shift target to the left and add padding if encoder-decoder model
        # so e.g. decoder gets '=123$' and target is '123$'
        if self.enc_dec:
            tgt = y.clone()
            tgt[:, :-1] = y[:, 1:]
            tgt[:, -1] = self.tokenizer.pad_token_id
        else:
            tgt = y

        if self.only_answer_loss:
            # mask out all tokens except answer tokens
            tgt = answer_mask(
                tgt,
                pad_token_id=self.tokenizer.pad_token_id,
                equal_token_id=self.equal_token_id,
            )

        # calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=self.tokenizer.pad_token_id,
            weight=self.class_weights if self.pause_token_id else None,
            reduction="mean",
        )
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(
        self, batch: Tensor | list, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """
        Dataloader 0 is the val split, others are from "test" data
        (see self.test_dataloader_names). I mistakenly refer to validation
        datasets as test. The true test dataset that is run to get the final
        results is not included here.
        """
        if dataloader_idx == 0:
            # evaluate language modeling loss on sequence
            x, y = batch
            logits = self.forward(batch)

            # shift target to the left and add padding if encoder-decoder model
            # so e.g. decoder gets '=123$' and target is '123$'
            if self.enc_dec:
                tgt = y.clone()
                tgt[:, :-1] = y[:, 1:]
                tgt[:, -1] = self.tokenizer.pad_token_id
            else:
                tgt = y

            if self.only_answer_loss:
                # mask out all tokens except answer tokens
                tgt = answer_mask(
                    tgt,
                    pad_token_id=self.tokenizer.pad_token_id,
                    equal_token_id=self.equal_token_id,
                )

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                tgt.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
                weight=self.class_weights if self.pause_token_id else None,
            )
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )
            return loss
        else:
            # evaluate accuracy and loss on TEST set (other val dataloaders than 0)
            res = eval_on_batch(
                self,
                self.tokenizer,
                batch,
                stop_token=self.tokenizer.encode("$")[0],
                eval_func=self.eval_func,
            )
            # index - 1 coz 0 is val
            test_dl_name = self.test_dataloader_names[dataloader_idx - 1]
            self.log_dict(
                {
                    f"val_acc/{test_dl_name}": res["accuracy"],
                    f"val_loss/{test_dl_name}": res["loss"],
                },
                batch_size=1,  # since ArithmeticEvalDataset returns 1 example
                add_dataloader_idx=False,
                prog_bar=True,
                sync_dist=True,
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

        use_scheduler = True

        use_fused = torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=1,
            betas=self.betas,
            **extra_args,
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
            "lr_scheduler": {
                "scheduler": (
                    lr_scheduler
                    if use_scheduler
                    else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: self.lr)
                ),
                "interval": "step",
            },
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
        n_beams: int = 0,
        return_logits: bool = False,
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
            n_beams=n_beams,
            return_logits=return_logits,
        )
