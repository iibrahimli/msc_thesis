import torch
from torch import Tensor, nn


def beam_search_generate(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    encoder_source: Tensor | None,
    temperature: float,
    top_k: int,
    stop_token: int,
    n_beams: int,
) -> Tensor:
    gen_start_idx = idx.size(-1)

    if encoder_source is not None:
        if encoder_source.ndim == 1:
            encoder_source = encoder_source.unsqueeze(0)
        memory = model.encode(encoder_source)
        memory = memory.repeat(n_beams, 1, 1)

    # Initialize beam
    beams = [idx.clone() for _ in range(n_beams)]
    beam_scores = torch.zeros(n_beams, device=idx.device)
    finished_beams = []
    finished_scores = []

    for _ in range(max_new_tokens):
        all_candidates = []
        all_scores = []

        for beam_idx, beam in enumerate(beams):
            if beam.size(1) > model.context_len:
                beam_cond = beam[:, -model.context_len :]
            else:
                beam_cond = beam

            if encoder_source is not None:
                logits = model.decode(beam_cond, memory[beam_idx : beam_idx + 1])
            else:
                logits = model(beam_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = nn.functional.log_softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(n_beams)

            for prob, token in zip(top_probs[0], top_indices[0]):
                candidate = torch.cat([beam, token.unsqueeze(0).unsqueeze(0)], dim=1)
                score = beam_scores[beam_idx] + prob.item()
                all_candidates.append(candidate)
                all_scores.append(score)

        ordered = sorted(zip(all_scores, all_candidates), key=lambda x: -x[0])
        beams = []
        beam_scores = torch.zeros(n_beams, device=idx.device)

        for i, (score, candidate) in enumerate(ordered):
            if i >= n_beams:
                break
            if stop_token is not None and candidate[0, -1].item() == stop_token:
                finished_beams.append(candidate)
                finished_scores.append(score)
            else:
                beams.append(candidate)
                beam_scores[len(beams) - 1] = score

        if len(beams) == 0:
            break

        # Check for repetition in last 20 tokens
        if all(torch.all(beam[:, -20:] == beam[:, -1]) for beam in beams):
            break

    if finished_beams:
        best_beam = max(zip(finished_scores, finished_beams), key=lambda x: x[0])[1]
    else:
        best_beam = max(zip(beam_scores, beams), key=lambda x: x[0])[1]

    return best_beam[:, gen_start_idx:]


@torch.inference_mode()
def generate(
    model: nn.Module,
    idx: Tensor,
    max_new_tokens: int,
    encoder_source: Tensor | None = None,
    temperature: float = 1.0,
    top_k: int = 1,
    stop_token: int = None,
    n_beams: int = 0,
    return_logits: bool = False,
) -> Tensor:
    """
    Take a conditioning sequence of indices idx (tensor of shape [batch, seq_len])
    and complete the sequence max_new_tokens times, feeding the predictions back
    into the model each time. Most likely you'll want to make sure to be in
    model.eval() mode of operation for this.

    encoder_source hints that the model is an encoder-decoder model and the
    encoder_source will be encoded and used as memory for the decoder.
    """

    # unsqueeze
    if idx.ndim == 1:
        idx = idx.unsqueeze(0)

    if isinstance(stop_token, list) and len(stop_token) == 1:
        stop_token = stop_token[0]

    assert isinstance(idx, torch.Tensor), "idx must be a torch.Tensor"
    assert idx.dim() == 2, "idx must be a 2D tensor of shape [batch, seq_len]"
    assert idx.size(1) <= model.context_len, "sequence length must be <= context_len"
    assert idx.size(0) == 1, "only batch size = 1 supported"

    if n_beams > 0:
        return beam_search_generate(
            model,
            idx,
            max_new_tokens,
            encoder_source,
            temperature,
            top_k,
            stop_token,
            n_beams,
        )
    # else, do top-k sampling

    # keep track of where generated part starts to only return it
    gen_start_idx = idx.size(-1)

    if return_logits:
        # seq_len, vocab_size
        token_logits = []

    # get hidden state from encoder
    if encoder_source is not None:
        # don't care about masks for now since only batch size = 1
        if encoder_source.ndim == 1:
            encoder_source = encoder_source.unsqueeze(0)
        memory = model.encode(encoder_source)

    for i in range(max_new_tokens):

        # crop to context_len if necessary
        if idx.size(1) > model.context_len:
            idx_cond = idx[:, -model.context_len :]
            # can only move by 1, since 1 token is generated
            gen_start_idx = max(0, gen_start_idx - 1)
        else:
            idx_cond = idx

        # logits shape: [batch, seq_len, vocab_size]
        if encoder_source is not None:
            # enc-dec model
            logits = model.decode(idx_cond, memory)
        else:
            logits = model(idx_cond)

        if return_logits:
            token_logits.append(logits[0, -1, :].detach().cpu().numpy())

        # get logits at final step and apply temperature
        logits = logits[:, -1, :] / temperature

        # optionally apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # apply softmax
        probs = nn.functional.softmax(logits, dim=-1)

        # sample from the distribution, next_tokens [B, 1]
        next_tokens = torch.multinomial(probs, num_samples=1)

        # append to the sequence
        idx = torch.cat([idx, next_tokens], dim=1)

        # stop if stop_token is generated
        if stop_token is not None and next_tokens.item() == stop_token:
            break

        # HACK: stop generating if last 20 tokens are the same
        same_tok_tol = 20
        if i > same_tok_tol and (idx[:, -same_tok_tol:] == idx[:, -1]).all():
            break

    if return_logits:
        return idx[:, gen_start_idx:], token_logits

    return idx[:, gen_start_idx:]
