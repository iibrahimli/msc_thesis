import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..tokenizer import Tokenizer
from .hooks import generate_hooked, get_attention_map, set_attn_kwargs_prehook


def plot_head(
    ax: plt.Axes,
    map: torch.Tensor,
    title: str,
    cmap: str = "binary",
    xticks: list = None,
    yticks: list = None,
    colorbar: bool = False,
    alpha: float = 1.0,
):
    ax.imshow(map, cmap=cmap, interpolation="none", alpha=alpha)
    if yticks:
        ax.set_yticks(np.arange(len(yticks)) - 0.5)
        ax.set_yticklabels(yticks, va="top")
    if xticks:
        ax.set_xticks(np.arange(len(xticks)) - 0.5)
        ax.set_xticklabels(xticks, ha="left")
    ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(ax.images[0], ax=ax, shrink=0.3)
    # grid
    ax.grid(which="both", color="k", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("source")
    ax.set_ylabel("target")


def plot_module(
    fig: plt.Figure,
    module_name: str,
    attn_map: torch.Tensor,
    ticks: list[str],
    plot_combined: bool = False,
):
    n_heads = attn_map.shape[1]
    axs = fig.subplots(1, n_heads + 1 if plot_combined else n_heads)
    fig.suptitle(module_name)

    # choose cmaps for combined attn map
    cmaps = ["Reds", "Blues", "Purples", "Greens", "Oranges"]

    for i in range(n_heads):
        plot_head(
            axs[i],
            attn_map[0, i],
            title=f"head {i}",
            xticks=ticks,
            yticks=ticks,
        )
        # combined attn map
        if plot_combined:
            plot_head(
                axs[-1],
                attn_map[0, i],
                title="combined",
                cmap=cmaps[i % len(cmaps)],
                alpha=0.5,
                xticks=ticks,
                yticks=ticks,
                colorbar=False,
            )
    # rotate yticks
    for ax in axs:
        ax.tick_params(axis="y", rotation=90)


def plot_attn_maps(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    a: int,
    b: int,
    module_names: list[str],
    savepath: str,
    pad_zeros: int = 0,
    filler_tokens_prompt: int = 0,
    save: bool = False,
    fig_scale: float = 1,
    reverse_ops: bool = False,
    reverse_ans: bool = False,
    figtitle_prefix: str = "",
) -> dict[str, torch.Tensor]:
    astr = str(a)
    bstr = str(b)

    if reverse_ops:
        astr = astr[::-1]
        bstr = bstr[::-1]

    prompt_str = (
        f"${'.' * filler_tokens_prompt}{astr.zfill(pad_zeros)}+{bstr.zfill(pad_zeros)}="
    )
    # prompt_str = "\n" + prompt_str
    print("prompt:", repr(prompt_str), f"{len(astr)}+{len(bstr)}")
    true_ans = str(a + b)
    if reverse_ans:
        true_ans = true_ans[::-1]
    print("true_ans:", true_ans)

    prompt = torch.tensor([tokenizer.encode(prompt_str)])
    stop_token_id = tokenizer.encode("$")[0]

    attn_maps = {}

    # generate answer
    pred_tensor = generate_hooked(
        model,
        prompt=prompt,
        stop_token=stop_token_id,
        hook_config={
            mn: {
                "hook": get_attention_map(mn, attn_maps),
                "pre_hook": set_attn_kwargs_prehook,
            }
            for mn in module_names
        },
    )

    pred_answer_str = tokenizer.decode(pred_tensor[0].tolist())
    pred_answer_num = "".join(c for c in pred_answer_str if c.isdigit())
    print("pred_answer:", pred_answer_str)

    for mn, matts in attn_maps.items():
        print(mn, matts.shape)

    # tokens for easier visualization
    ticks = list(prompt_str + pred_answer_str)
    ticks[0] = "\\n" if ticks[0] == "\n" else ticks[0]

    # for each module, in a subfigure plot heads as subplots
    n_heads = attn_maps[module_names[0]].shape[1]
    figsize = (n_heads * fig_scale, len(attn_maps) * fig_scale)
    fig = plt.figure(layout="constrained", figsize=figsize)
    fig.suptitle(
        f"{figtitle_prefix} Attention maps for prompt: {repr(prompt_str).replace('$', '\$')}, [{len(astr)}+{len(bstr)}]"
        f"\n predicted answer: {repr(pred_answer_str).replace('$', '\$')} ({'correct' if pred_answer_num == true_ans else 'incorrect, true: ' + true_ans})",
    )

    subfigs = fig.subfigures(len(attn_maps), 1, hspace=0, wspace=0)
    if len(attn_maps) == 1:
        subfigs = [subfigs]
    for i, (module_name, attn_map) in enumerate(attn_maps.items()):
        plot_module(subfigs[i], module_name, attn_map, ticks)

    if save:
        plt.savefig(savepath, dpi=90)
    plt.show()

    return attn_maps


def get_attn_maps_fig_for_model(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: torch.Tensor,
    fig_scale: float = 1,
):
    """
    TODO: refactor, mostly copy-paste from plot_attn_maps
    """
    prompt_str = repr(tokenizer.decode(prompt.squeeze().tolist()))
    stop_token_id = tokenizer.encode("$")[0]

    # all attn modules
    module_names = [mn for mn, _ in model.named_modules() if mn.endswith("self_attn")]
    # HACK change model.transformer_encoder.layers.0.self_attn to model.transformer_encoder.layers[0].self_attn
    module_names = [re.sub(r"\.(\d+)\.", r"[\1].", mn) for mn in module_names]

    attn_maps = {}

    # generate answer
    pred_tensor = generate_hooked(
        model,
        prompt=prompt,
        stop_token=stop_token_id,
        hook_config={
            mn: {
                "hook": get_attention_map(mn, attn_maps),
                "pre_hook": set_attn_kwargs_prehook,
            }
            for mn in module_names
        },
    )

    n_heads = attn_maps[module_names[0]].shape[1]
    ticks = list(prompt_str + repr(tokenizer.decode(pred_tensor.squeeze().tolist())))

    # adjust figsize based on n_heads (width) and n_modules (height)
    figsize = (n_heads * fig_scale, len(module_names) * fig_scale)
    fig = plt.figure(layout="constrained", figsize=figsize)
    fig.suptitle(f"Attention maps for prompt: {prompt_str}")

    subfigs = fig.subfigures(len(attn_maps), 1, hspace=0, wspace=0)
    if len(attn_maps) == 1:
        subfigs = [subfigs]
    for i, (module_name, attn_map) in enumerate(attn_maps.items()):
        plot_module(
            subfigs[i], module_name, attn_map.cpu().detach(), ticks, plot_combined=False
        )

    return fig
