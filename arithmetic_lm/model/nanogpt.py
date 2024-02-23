import math

import lightning as L
import torch
from torch import Tensor, nn

from arithmetic_lm.eval_utils import eval_on_batch
from arithmetic_lm.tokenizer import Tokenizer
from arithmetic_lm.train_utils import lr_cosine_annealing_with_warmup


# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class NanoGPT(nn.Module):
    """Simple small decoder-only transformer model using nn.TransformerDecoder."""

    def __init__(
        self,
        context_len: int,
        n_embd: int,
        n_head: int,
        n_layers: int,
        vocab_size: int,
        ff_factor: int = 4,
        dropout: float = 0.1,
    ):
        """
        Arguments:
            context_len: context length, i.e. the number of expected features in the input
            n_embd: dimensionality of model embeddings
            n_head: number of heads in the multi-head attention
            n_layers: number of nn.TransformerDecoderLayer layers
            vocab_size: size of the vocabulary
            ff_factor: factor by which to scale the hidden layer dimensionality in the feedforward layer
            dropout: dropout probability
        """

        super().__init__()
        self.context_len = context_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.ff_factor = ff_factor
        self.dropout = dropout

        # embedding
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_encoder = PositionalEncoding(
            n_embd, max_len=context_len, dropout=dropout
        )

        # same as decoder layer essentially, but without cross attention
        self.layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * ff_factor,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(n_embd),
        )

        # output to vocab dim
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embedding.weight

        # TODO init weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len]``

        Returns:
            logits: Tensor, shape ``[batch_size, seq_len, vocab_size]``
        """
        x = self.embedding(x)
        x = self.pos_encoder(x)

        x = self.transformer_encoder(
            x,
            is_causal=True,
            mask=nn.Transformer.generate_square_subsequent_mask(
                self.context_len, device=x.device
            ),
        )
        x = self.lm_head(x)
        return x

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.inference_mode()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 1,
        stop_token: int = None,
        seed: int = 42,
    ) -> Tensor:
        """
        Take a conditioning sequence of indices idx (tensor of shape [batch, seq_len]) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        # TODO implement seed w/ device support

        assert isinstance(idx, torch.Tensor), "idx must be a torch.Tensor"
        assert idx.dim() == 2, "idx must be a 2D tensor of shape [batch, seq_len]"
        assert idx.size(1) <= self.context_len, "sequence length must be <= context_len"
        assert idx.size(0) == 1, "only batch size = 1 supported"

        # keep track of where generated part starts to only return it
        gen_start_idx = idx.size(-1)

        for _ in range(max_new_tokens):
            # crop to context_len if necessary
            if idx.size(1) > self.context_len:
                idx_cond = idx[:, -self.context_len :]
                # can only move by 1, since 1 token is generated
                gen_start_idx -= max(0, gen_start_idx - 1)
            else:
                idx_cond = idx

            # logits shape: [batch, seq_len, vocab_size]
            logits = self.forward(idx_cond)

            # get logits at final step and apply temperature
            logits = logits[:, -1, :] / temperature

            # optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            # apply softmax
            probs = nn.functional.softmax(logits, dim=-1)

            # sample from the distribution
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            # append to the sequence
            idx = torch.cat([idx, next_token], dim=1)

            # stop if stop_token is generated
            if stop_token is not None and next_token.item() == stop_token:
                break

        return idx[:, gen_start_idx:]


class LightningNanoGPT(L.LightningModule):
    def __init__(
        self,
        tokenizer: Tokenizer,
        context_len: int,
        n_embd: int,
        n_head: int,
        n_layers: int,
        vocab_size: int,
        ff_factor: int = 4,
        dropout: float = 0.1,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.1,
        warmup_iters: int = 100,
    ):
        super().__init__()
        self.model = NanoGPT(
            context_len=context_len,
            n_embd=n_embd,
            n_head=n_head,
            n_layers=n_layers,
            vocab_size=vocab_size,
            ff_factor=ff_factor,
            dropout=dropout,
        )
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.tokenizer = tokenizer
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # batch: (batch_size, seq_len)
        # split into input and target (shifted by 1)
        x, y = batch[:, :-1], batch[:, 1:]
        # forward pass
        logits = self.model(x)
        # calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1)
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        # evaluate language modeling loss on sequence
        x, y = batch[:, :-1], batch[:, 1:]
        logits = self.model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1)
        )
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: list, batch_idx: int) -> Tensor:
        res = eval_on_batch(
            self, self.tokenizer, batch, stop_token=self.tokenizer.encode("\n")
        )
        self.log("test_acc", res["accuracy"], batch_size=len(batch))
        return {"loss": 0, "test_acc": res["accuracy"]}

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, torch.nn.MultiheadAttention
                ):
                    # special case for multihead attention
                    decay.add(fpn)

        # subtle: 'embedding.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'embedding.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via embedding.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        # TODO: again, somehow get device from trainer
        use_fused = torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, betas=self.betas, **extra_args
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
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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
        return self.model.generate(
            idx, max_new_tokens, temperature, top_k, stop_token, seed
        )
