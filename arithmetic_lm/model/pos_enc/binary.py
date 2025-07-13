import math

import torch
import torch.nn as nn


class BinaryPositionalEncoding(nn.Module):
    """
    Binary positional encoding that encodes absolute position in binary representation.

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        d_bits: Dimension of binary embedding (should be << d_model, typically <= 16)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int,
        dropout: float,
        d_bits: int = 16,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.d_bits = d_bits

        # Calculate number of bits needed for max_len
        self.num_bits = math.ceil(math.log2(max_len)) if max_len > 1 else 1

        # Learnable embedding matrix for binary features
        # Use modest initialization (std=0.02)
        self.w_bits = nn.Parameter(torch.randn(self.num_bits, d_bits) * 0.02)

        # Register binary encodings as buffer (not learned)
        self.register_buffer("binary_encodings", self._create_binary_encodings())

    def _create_binary_encodings(self):
        """Create binary encodings for all positions up to max_len"""
        encodings = torch.zeros(self.max_len, self.num_bits)

        for i in range(self.max_len):
            # Convert position to binary and pad to num_bits
            binary_str = format(i, f"0{self.num_bits}b")
            for j, bit in enumerate(binary_str):
                encodings[i, j] = int(bit)

        return encodings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        
        Returns:
            Tensor, shape ``[batch_size, seq_len, d_model]`` where last d_bits dimensions
            are replaced with binary positional encodings.
        """
        # Get binary representations for positions 0 to seq_len-1
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len {self.max_len}")
        binary_positions = self.binary_encodings[:seq_len]

        # Expand to match batch size
        binary_positions = binary_positions.unsqueeze(0).expand(x.size(0), -1, -1)
        # Replace last d_bits dimensions with binary encodings
        x[:, :, -self.d_bits:] = torch.matmul(binary_positions, self.w_bits)

        return x