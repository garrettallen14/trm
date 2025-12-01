"""
Tiny Recursive Model (TRM) for ARC-AGI.

A 2-layer looped transformer with:
- Cell-based tokenization (each grid cell is a token)
- 2D Rotary Position Embeddings (RoPE)
- Deep supervision at each recursion step
- Input injection for stable gradients
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for grid-based inputs."""
    
    def __init__(self, dim: int, max_size: int = 32):
        super().__init__()
        self.dim = dim
        self.max_size = max_size
        
        # Separate frequencies for row and column
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // 2, 2).float() / (dim // 2)))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute sin/cos for all positions
        self._build_cache(max_size)
    
    def _build_cache(self, max_size: int):
        """Precompute sin/cos embeddings for efficiency."""
        positions = torch.arange(max_size)
        
        # Row and column frequencies
        row_freqs = torch.outer(positions, self.inv_freq)  # [max_size, dim//4]
        col_freqs = torch.outer(positions, self.inv_freq)  # [max_size, dim//4]
        
        # Create meshgrid for 2D positions
        row_pos, col_pos = torch.meshgrid(positions, positions, indexing='ij')
        
        # Combine row and column embeddings
        # Each position (r, c) gets [sin(r*f), cos(r*f), sin(c*f), cos(c*f)]
        row_emb = torch.stack([
            torch.sin(row_pos.unsqueeze(-1) * self.inv_freq),
            torch.cos(row_pos.unsqueeze(-1) * self.inv_freq)
        ], dim=-1).flatten(-2)  # [max_size, max_size, dim//2]
        
        col_emb = torch.stack([
            torch.sin(col_pos.unsqueeze(-1) * self.inv_freq),
            torch.cos(col_pos.unsqueeze(-1) * self.inv_freq)
        ], dim=-1).flatten(-2)  # [max_size, max_size, dim//2]
        
        # Concatenate row and column embeddings
        self.register_buffer("sin_cos_cache", torch.cat([row_emb, col_emb], dim=-1))
    
    def apply_rotary(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to input tensor.
        
        Args:
            x: [batch, seq_len, heads, dim]
            positions: [batch, seq_len, 2] (row, col pairs)
        """
        batch, seq_len, heads, dim = x.shape
        
        # Get sin/cos for each position
        row_pos = positions[:, :, 0].clamp(0, self.max_size - 1)
        col_pos = positions[:, :, 1].clamp(0, self.max_size - 1)
        
        # Look up precomputed embeddings
        sin_cos = self.sin_cos_cache[row_pos, col_pos]  # [batch, seq_len, dim]
        sin_cos = sin_cos.unsqueeze(2)  # [batch, seq_len, 1, dim]
        
        # Split into sin and cos components
        half_dim = dim // 2
        sin_part = sin_cos[..., :half_dim]
        cos_part = sin_cos[..., half_dim:]
        
        # Apply rotation
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        rotated = torch.cat([
            x1 * cos_part - x2 * sin_part,
            x1 * sin_part + x2 * cos_part
        ], dim=-1)
        
        return rotated


class MultiHeadAttention(nn.Module):
    """Multi-head attention with 2D RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.rope = RoPE2D(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Spectral normalization for stability
        self._apply_spectral_norm()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization to projections for training stability."""
        self.q_proj = nn.utils.parametrizations.spectral_norm(self.q_proj)
        self.k_proj = nn.utils.parametrizations.spectral_norm(self.k_proj)
        self.v_proj = nn.utils.parametrizations.spectral_norm(self.v_proj)
        self.out_proj = nn.utils.parametrizations.spectral_norm(self.out_proj)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            positions: [batch, seq_len, 2] (row, col)
            mask: [batch, seq_len] or None
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        
        # Apply 2D RoPE to Q and K
        q = self.rope.apply_rotary(q, positions)
        k = self.rope.apply_rotary(k, positions)
        
        # Reshape for attention: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)  # [batch, heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        return self.out_proj(out), attn_probs


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Spectral normalization
        self.w1 = nn.utils.parametrizations.spectral_norm(self.w1)
        self.w2 = nn.utils.parametrizations.spectral_norm(self.w2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN attention
        normed = self.ln1(x)
        attn_out, attn_probs = self.attn(normed, positions, mask)
        x = x + self.dropout(attn_out)
        
        # Pre-LN feedforward
        x = x + self.dropout(self.ff(self.ln2(x)))
        
        return x, attn_probs


class LoopedTransformer(nn.Module):
    """Looped transformer with shared weights across iterations."""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Shared transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_ln = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Single pass through all layers."""
        attn_probs_list = []
        
        for block in self.blocks:
            x, attn_probs = block(x, positions, mask)
            attn_probs_list.append(attn_probs)
        
        return self.final_ln(x), attn_probs_list


class TinyRecursiveModel(nn.Module):
    """
    Tiny Recursive Model for ARC-AGI.
    
    Architecture:
    - Input: Flattened grid cells + demo grids
    - Encoder: Embed cells, add 2D positions
    - Recursive refinement: Loop through shared transformer
    - Output: Per-cell color predictions
    
    The model alternates between:
    - "Think" step: Process with transformer
    - "Act" step: Update output prediction
    
    Deep supervision is applied at each recursion step.
    """
    
    def __init__(
        self,
        n_colors: int = 11,  # 0-9 colors + padding
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        n_recursions: int = 16,
        max_grid_size: int = 32,
        dropout: float = 0.0
    ):
        super().__init__()
        self.n_colors = n_colors
        self.d_model = d_model
        self.n_recursions = n_recursions
        self.max_grid_size = max_grid_size
        
        # Input embeddings
        self.cell_embed = nn.Embedding(n_colors, d_model)
        
        # Special tokens
        self.input_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.output_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Looped transformer backbone
        self.transformer = LoopedTransformer(d_model, n_heads, n_layers, dropout)
        
        # Output head: predict cell colors
        self.output_head = nn.Linear(d_model, n_colors)
        
        # Latent state for recursion
        self.latent_proj = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def encode_grid(
        self,
        grid: torch.Tensor,
        is_output: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a grid into embeddings with positions.
        
        Args:
            grid: [batch, height, width] cell values
            is_output: Whether this is an output grid
        
        Returns:
            embeddings: [batch, height*width, d_model]
            positions: [batch, height*width, 2]
        """
        batch, height, width = grid.shape
        
        # Flatten grid
        flat_grid = grid.view(batch, -1)  # [batch, height*width]
        
        # Embed cells
        embeddings = self.cell_embed(flat_grid)  # [batch, h*w, d_model]
        
        # Add special token
        token = self.output_token if is_output else self.input_token
        embeddings = embeddings + token
        
        # Create position indices
        row_pos = torch.arange(height, device=grid.device).view(-1, 1).expand(height, width)
        col_pos = torch.arange(width, device=grid.device).view(1, -1).expand(height, width)
        positions = torch.stack([row_pos.flatten(), col_pos.flatten()], dim=-1)
        positions = positions.unsqueeze(0).expand(batch, -1, -1)  # [batch, h*w, 2]
        
        return embeddings, positions
    
    def forward(
        self,
        demo_inputs: list[torch.Tensor],   # List of [batch, h, w] input grids
        demo_outputs: list[torch.Tensor],  # List of [batch, h, w] output grids
        test_input: torch.Tensor,          # [batch, h, w] test input
        n_recursions: int | None = None,
        return_intermediates: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            demo_inputs: Demonstration input grids
            demo_outputs: Demonstration output grids
            test_input: Test input grid to predict output for
            n_recursions: Number of recursive steps (default: self.n_recursions)
            return_intermediates: Return predictions at each step
        
        Returns:
            Dictionary with:
            - logits: [batch, h*w, n_colors] final predictions
            - intermediates: List of predictions at each step (if requested)
            - attn_entropy: Attention entropy for monitoring
        """
        n_recursions = n_recursions or self.n_recursions
        batch = test_input.shape[0]
        test_h, test_w = test_input.shape[1], test_input.shape[2]
        
        # Encode all demos
        all_embeddings = []
        all_positions = []
        
        for inp, out in zip(demo_inputs, demo_outputs):
            inp_emb, inp_pos = self.encode_grid(inp, is_output=False)
            out_emb, out_pos = self.encode_grid(out, is_output=True)
            
            # Add separator between input and output
            sep = self.sep_token.expand(batch, 1, -1)
            sep_pos = torch.zeros(batch, 1, 2, device=inp.device, dtype=torch.long)
            
            all_embeddings.extend([inp_emb, sep, out_emb, sep])
            all_positions.extend([inp_pos, sep_pos, out_pos, sep_pos])
        
        # Encode test input
        test_emb, test_pos = self.encode_grid(test_input, is_output=False)
        sep = self.sep_token.expand(batch, 1, -1)
        sep_pos = torch.zeros(batch, 1, 2, device=test_input.device, dtype=torch.long)
        
        all_embeddings.extend([test_emb, sep])
        all_positions.extend([test_pos, sep_pos])
        
        # Concatenate all embeddings
        x = torch.cat(all_embeddings, dim=1)
        positions = torch.cat(all_positions, dim=1)
        
        # Track where test output should go
        test_output_start = x.shape[1]
        
        # Initialize output predictions (start with zeros = unknown)
        output_emb = self.cell_embed(torch.zeros(batch, test_h * test_w, device=x.device, dtype=torch.long))
        output_emb = output_emb + self.output_token
        
        # Append output embeddings
        x = torch.cat([x, output_emb], dim=1)
        positions = torch.cat([positions, test_pos], dim=1)
        
        # Recursive refinement
        intermediates = []
        total_attn_entropy = 0.0
        
        for step in range(n_recursions):
            # Pass through transformer
            x, attn_probs_list = self.transformer(x, positions)
            
            # Track attention entropy for monitoring
            for attn_probs in attn_probs_list:
                entropy = -(attn_probs * (attn_probs + 1e-10).log()).sum(-1).mean()
                total_attn_entropy += entropy
            
            # Get output predictions
            output_hidden = x[:, test_output_start:, :]  # [batch, h*w, d_model]
            logits = self.output_head(output_hidden)     # [batch, h*w, n_colors]
            
            if return_intermediates:
                intermediates.append(logits)
            
            # Update output embeddings with predictions (input injection)
            if step < n_recursions - 1:
                pred_colors = logits.argmax(dim=-1)  # [batch, h*w]
                new_output_emb = self.cell_embed(pred_colors) + self.output_token
                
                # Blend with latent state for stability
                latent = self.latent_proj(output_hidden)
                new_output_emb = new_output_emb + 0.5 * latent
                
                x = torch.cat([x[:, :test_output_start, :], new_output_emb], dim=1)
        
        avg_attn_entropy = total_attn_entropy / (n_recursions * len(attn_probs_list))
        
        result = {
            "logits": logits,
            "attn_entropy": avg_attn_entropy
        }
        
        if return_intermediates:
            result["intermediates"] = intermediates
        
        return result
    
    def compute_loss(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        n_recursions: int | None = None,
        supervision_weights: str = "uniform"  # "uniform" or "linear"
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss with deep supervision.
        
        Args:
            demo_inputs: Demonstration input grids
            demo_outputs: Demonstration output grids
            test_input: Test input grid
            test_output: Ground truth test output
            n_recursions: Number of recursive steps
            supervision_weights: How to weight intermediate losses
        
        Returns:
            Dictionary with total_loss and per-step losses
        """
        n_recursions = n_recursions or self.n_recursions
        
        # Forward with intermediates
        result = self.forward(
            demo_inputs, demo_outputs, test_input,
            n_recursions=n_recursions,
            return_intermediates=True
        )
        
        intermediates = result["intermediates"]
        target = test_output.view(test_output.shape[0], -1)  # [batch, h*w]
        
        # Compute loss at each step
        step_losses = []
        for i, logits in enumerate(intermediates):
            step_loss = F.cross_entropy(
                logits.view(-1, self.n_colors),
                target.view(-1),
                ignore_index=-1  # Ignore padding
            )
            step_losses.append(step_loss)
        
        # Weight the losses
        if supervision_weights == "uniform":
            weights = [1.0 / len(step_losses)] * len(step_losses)
        elif supervision_weights == "linear":
            # Linear ramp: later steps weighted more
            total = sum(range(1, len(step_losses) + 1))
            weights = [(i + 1) / total for i in range(len(step_losses))]
        else:
            raise ValueError(f"Unknown supervision_weights: {supervision_weights}")
        
        total_loss = sum(w * l for w, l in zip(weights, step_losses))
        
        # Add entropy regularization (prevent attention collapse)
        entropy_loss = -result["attn_entropy"] * 0.01
        total_loss = total_loss + entropy_loss
        
        return {
            "total_loss": total_loss,
            "step_losses": step_losses,
            "final_loss": step_losses[-1],
            "attn_entropy": result["attn_entropy"]
        }
    
    @torch.no_grad()
    def predict(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        n_recursions: int | None = None
    ) -> torch.Tensor:
        """
        Predict output grid.
        
        Returns:
            predictions: [batch, height, width] predicted colors
        """
        self.eval()
        result = self.forward(demo_inputs, demo_outputs, test_input, n_recursions)
        logits = result["logits"]  # [batch, h*w, n_colors]
        
        # Get predictions
        preds = logits.argmax(dim=-1)  # [batch, h*w]
        
        # Reshape to grid
        test_h, test_w = test_input.shape[1], test_input.shape[2]
        return preds.view(-1, test_h, test_w)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown by component
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params
    
    return {
        "total": total,
        "trainable": trainable,
        "breakdown": breakdown
    }


if __name__ == "__main__":
    # Quick test
    model = TinyRecursiveModel(
        d_model=512,
        n_heads=4,
        n_layers=2,
        n_recursions=8
    )
    
    print("Model parameter count:")
    counts = count_parameters(model)
    print(f"  Total: {counts['total']:,}")
    print(f"  Trainable: {counts['trainable']:,}")
    print("  Breakdown:")
    for name, count in counts['breakdown'].items():
        print(f"    {name}: {count:,}")
    
    # Test forward pass
    batch = 2
    demo_inputs = [torch.randint(0, 10, (batch, 5, 5)) for _ in range(2)]
    demo_outputs = [torch.randint(0, 10, (batch, 5, 5)) for _ in range(2)]
    test_input = torch.randint(0, 10, (batch, 5, 5))
    test_output = torch.randint(0, 10, (batch, 5, 5))
    
    # Forward
    result = model(demo_inputs, demo_outputs, test_input, return_intermediates=True)
    print(f"\nOutput shape: {result['logits'].shape}")
    print(f"Attention entropy: {result['attn_entropy']:.4f}")
    print(f"Intermediate steps: {len(result['intermediates'])}")
    
    # Loss
    loss_result = model.compute_loss(
        demo_inputs, demo_outputs, test_input, test_output,
        supervision_weights="linear"
    )
    print(f"\nTotal loss: {loss_result['total_loss']:.4f}")
    print(f"Final loss: {loss_result['final_loss']:.4f}")
    print(f"Step losses: {[f'{l:.4f}' for l in loss_result['step_losses']]}")
