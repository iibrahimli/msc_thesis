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
):
    n_heads = attn_map.shape[1]
    axs = fig.subplots(1, n_heads + 1)  # +1 for combined attn map
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
    figsize: tuple[int, int] = (8, 8),
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
    fig = plt.figure(layout="constrained", figsize=figsize)
    fig.suptitle(
        f"{figtitle_prefix} Attention maps for prompt: {repr(prompt_str).replace('$', '\$')}, [{len(astr)}+{len(bstr)}]"
        f"\n predicted answer: {repr(pred_answer_str).replace('$', '\$')} ({'correct' if pred_answer_num == true_ans else 'incorrect, true: ' + true_ans})",
    )

    subfigs = fig.subfigures(len(attn_maps), 1, hspace=0, wspace=0)
    for i, (module_name, attn_map) in enumerate(attn_maps.items()):
        plot_module(subfigs[i], module_name, attn_map, ticks)

    if save:
        plt.savefig(savepath, dpi=90)
    plt.show()

    return attn_maps


def merge_heads_across_layers(attn_maps: list[torch.Tensor]) -> torch.Tensor:
    """
    Merge attention maps, such that non-zero weights in result indicate a
    possible indirect connection between the corresponding tokens in the
    input and output sequences. This is just a matrix multiply with second
    attention map transposed, over last 2 dims.

    Args:
        attn_maps: List of attention maps for each layer in order, each map has shape
            [batch_size, n_heads, tgt_seq_len, src_seq_len].

    Returns:
        A tensor of merged attention maps, of shape [batch_size, n_heads, tgt_seq_len, src_seq_len]
    """

    assert len(attn_maps) > 1, "Need at least 2 attention maps to merge"

    merged = attn_maps[0]
    # TODO
