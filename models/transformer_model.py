"""
Transformer-based language model for causal language modeling.
Based on "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len, seq_len) or (seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape to (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with self-attention and feed-forward.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Causal mask (seq_len, seq_len)

        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Pre-LayerNorm: normalize BEFORE attention (more stable training)
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout1(attn_output)

        # Pre-LayerNorm: normalize BEFORE feed-forward
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout2(ff_output)

        return x


class TransformerLanguageModel(nn.Module):
    """
    Transformer-based language model for next token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        """
        Initialize Transformer language model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding dimension)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Dimension of feed-forward network
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            pad_token_id: ID of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization (required for Pre-LN architecture)
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights with improved strategy for better training.
        Based on GPT-2 initialization.
        """
        # Initialize embedding layers
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Initialize all linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Special scaled init for residual projections (GPT-2 style)
        # Scale down the output projection in each layer
        for layer_idx, layer in enumerate(self.layers):
            # Scale attention output projection
            nn.init.normal_(
                layer.self_attn.W_o.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.num_layers)
            )
            # Scale feed-forward output projection
            nn.init.normal_(
                layer.feed_forward.linear2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.num_layers)
            )

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generate causal mask to prevent attending to future tokens.

        Args:
            sz: Sequence length

        Returns:
            Causal mask of shape (sz, sz)
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return ~mask  # Invert: 1 means can attend, 0 means cannot

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)

        Returns:
            logits: Predictions for next token (batch_size, seq_len, vocab_size)
        """
        seq_len = input_ids.size(1)
        device = input_ids.device

        # Create causal mask
        mask = self._generate_square_subsequent_mask(seq_len).to(device)

        # Embed tokens and add positional encoding
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer normalization (Pre-LN pattern)
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Initial token IDs (batch_size, initial_seq_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: If set, sample from top-k tokens only
            eos_token_id: End-of-sequence token ID

        Returns:
            generated_ids: Generated token IDs
        """
        self.eval()
        device = input_ids.device

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :]

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

                # Stop if EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        return generated

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Transformer model
    vocab_size = 10000
    batch_size = 4
    seq_len = 32

    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        dropout=0.1,
    )

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    print(f"Model: Transformer Language Model")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Parameters: {model.get_num_parameters():,}")

    # Test generation
    initial_ids = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(initial_ids, max_length=20, temperature=1.0)
    print(f"\nGeneration test:")
    print(f"Initial length: {initial_ids.size(1)}")
    print(f"Generated length: {generated.size(1)}")
