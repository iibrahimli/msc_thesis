import random

import torch
from torch import Tensor, nn


class AbacusEncoding(nn.Module):
    """
    Abacus Embeddings, learned emebddings resued for each digit.
    Integers must be reversed for this to work correctly.
    Transformers Can Do Arithmetic with the Right Embeddings, McLeish et al. (2024)
    Taken from: https://github.com/mcleish7/arithmetic/blob/86022a57d38c0fde46444d62e8dcbebcc0af614c/abacus.py
    """

    def __init__(
        self,
        # digit_tokens: list[int],
        embedding_dim: int,
        max_seq_length: int = 256,
        max_k: int = 50,
    ):
        """
        digit_tokens (list): list of the tokens for each of the 10 digits, in pseudocode:
            `digit_tokens = tokenizer.convert_tokens_to_ids(['0','1','2','3','4','5','6','7','8','9'])`
        embedding_dim (int): dimension to embed into
        max_seq_length (int): maximum number of embeddings that can be trained
        max_k (int): maximum k value which we randomly shift by during training
        """
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding = torch.nn.Embedding(max_seq_length, embedding_dim)

        # TODO: hardcoded, CharTokenizer maps 0..9 -> 0..9
        digit_tokens = list(range(10))
        self.register_buffer("digits", torch.tensor(digit_tokens), persistent=False)

        self.max_k = max_k

    def helper(self, mask: Tensor, device: torch.device):
        """
        Converts a binary mask of digit locations into spans of consecutive digits
        """
        mask_shape = mask.shape

        # Create a shifted version of the mask to detect changes from 0 to 1
        shifted_mask = torch.cat(
            [
                torch.zeros((mask_shape[0], 1), device=device, dtype=mask.dtype),
                mask[:, :-1],
            ],
            dim=1,
        )
        starts = (shifted_mask != mask) & mask

        # Generate IDs for each segment of 1s, processing row-wise
        segment_ids = torch.cumsum(starts, dim=1)

        # Generate an index array row-wise
        index = torch.arange(mask.size(1)).repeat(mask.size(0), 1).to(device)

        # Reset index at the start of each segment
        reset_index = torch.zeros_like(mask).long()
        second_term = index * starts.long()
        reset_index = reset_index.scatter_add(1, segment_ids, second_term)

        # Calculate positions in segment
        positions = index - reset_index.gather(1, segment_ids) + 1

        # Ensure only values within 1-segments are non-zero
        result = positions * mask

        return result

    def forward(self, input_ids: Tensor):
        """
        input_ids (tensor): a batch of inputs, each row is a sample
        NOTE: returns pos embeddings to be added to input embeddings,
            does not add here unlike abs/learned pos encoding classes
        """
        mask = torch.isin(input_ids, self.digits)
        output = self.helper(mask, input_ids.device)

        k = 0
        if self.training:
            k = random.randint(0, self.max_k)
            # as we already have ones in the tensor, the tensor values will be k+1
            output[output > 0] += k
            output[output > 0] = torch.clamp(
                output[output > 0], max=self.max_seq_length - 1
            )

        return self.embedding(output)
