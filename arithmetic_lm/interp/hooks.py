import torch

from ..model import generate


def get_attention_map(name: str, cache: dict):
    def hook(module, inputs, output):
        # nn.MultiheadAttention outputs 2 tensors by default:
        # - the output of the last linear transformation with shape [bs, tgt_len, embed_dim]
        # - the attention map (weights) with shape [bs, n_heads, tgt_len, src_len]
        # keeps only last output, which is fine for our purposes
        cache[name] = output[1].detach()

    return hook


def set_attn_kwargs_prehook(module, args, kwargs):
    """
    make sure self.attn module is called with need_weights=True and
    average_attn_weights=False so that we get per-head attention weights
    """
    kwargs["need_weights"] = True
    kwargs["average_attn_weights"] = False
    return args, kwargs


def generate_hooked(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    stop_token: int,
    hook_config: dict[str, dict[str, callable]],
    decoder_prompt: torch.Tensor = None,
) -> str:
    model.eval()

    handles = []

    for module_name, hook_dict in hook_config.items():
        module = eval(f"model.{module_name}", {"model": model})

        if pre_hook := hook_dict.get("pre_hook"):
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))

        if hook := hook_dict.get("hook"):
            handles.append(module.register_forward_hook(hook))

    # HACK: encode, since just calling generate does not call
    # forward hook in the encoder for some weird reason (decoder hooks work fine)
    if model.enc_dec:
        model.encode(prompt)

    pred_tensor = generate(
        model,
        idx=decoder_prompt if model.enc_dec else prompt,
        encoder_source=prompt if model.enc_dec else None,
        max_new_tokens=100,
        stop_token=stop_token,
    )

    # remove hooks
    for handle in handles:
        handle.remove()

    return pred_tensor
