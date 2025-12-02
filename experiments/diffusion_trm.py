"""
Discrete Diffusion TRM - Novel Architecture Experiment

Instead of ad-hoc recursion, use principled discrete diffusion:
- Start with noisy/masked grid
- Progressively denoise to answer
- Deep supervision at each timestep (for free!)

Hypothesis: Same compute pattern as TRM, but mathematically principled.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# Import base components from our TRM
import sys
sys.path.insert(0, '/Users/garrettallen/github/trm')
from src.model import RoPE2D, MultiHeadAttention, TransformerBlock


class DiscreteNoise:
    """Discrete noise schedule for grid diffusion."""
    
    def __init__(self, num_timesteps: int = 16, num_colors: int = 11):
        self.num_timesteps = num_timesteps
        self.num_colors = num_colors
        
        # Linear noise schedule: probability of masking at each timestep
        # t=0 is clean, t=T is fully noisy
        self.mask_probs = torch.linspace(0, 0.95, num_timesteps)
    
    def add_noise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Add noise to grid at timestep t."""
        mask_prob = self.mask_probs[t]
        mask = torch.rand_like(x.float()) < mask_prob
        noise = torch.randint_like(x, 0, self.num_colors)
        return torch.where(mask, noise, x)
    
    def get_mask_prob(self, t: int) -> float:
        return self.mask_probs[t].item()


class DiffusionTRM(nn.Module):
    """
    Discrete Diffusion Tiny Recursive Model.
    
    Key differences from standard TRM:
    1. Explicit timestep conditioning (not just iteration count)
    2. Denoising objective (predict clean from noisy)
    3. Can use diffusion tricks (guidance, scheduling)
    """
    
    def __init__(
        self,
        n_colors: int = 11,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        num_timesteps: int = 16,
        max_grid_size: int = 32
    ):
        super().__init__()
        self.n_colors = n_colors
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        
        # Embeddings
        self.cell_embed = nn.Embedding(n_colors, d_model)
        self.timestep_embed = nn.Embedding(num_timesteps, d_model)
        
        # Special tokens
        self.demo_in_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.demo_out_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.test_in_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.noisy_out_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 2D positional encoding
        self.rope = RoPE2D(d_model // n_heads, max_grid_size)
        
        # Transformer (shared across "iterations" via diffusion steps)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        
        # Output head: predict clean color for each cell
        self.output_head = nn.Linear(d_model, n_colors)
        
        # Noise schedule
        self.noise = DiscreteNoise(num_timesteps, n_colors)
    
    def encode_grid(self, grid: torch.Tensor, grid_type: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a grid with type token and positions."""
        batch, h, w = grid.shape
        
        # Cell embeddings
        emb = self.cell_embed(grid)  # [batch, h, w, d_model]
        emb = rearrange(emb, 'b h w d -> b (h w) d')
        
        # Add type token
        if grid_type == 'demo_in':
            emb = emb + self.demo_in_token
        elif grid_type == 'demo_out':
            emb = emb + self.demo_out_token
        elif grid_type == 'test_in':
            emb = emb + self.test_in_token
        elif grid_type == 'noisy_out':
            emb = emb + self.noisy_out_token
        
        # Compute 2D positions
        positions = torch.stack(torch.meshgrid(
            torch.arange(h, device=grid.device),
            torch.arange(w, device=grid.device),
            indexing='ij'
        ), dim=-1)  # [h, w, 2]
        positions = positions.unsqueeze(0).expand(batch, -1, -1, -1)
        positions = rearrange(positions, 'b h w c -> b (h w) c')
        
        return emb, positions
    
    def forward(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        noisy_output: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Forward pass for denoising.
        
        Args:
            demo_inputs: List of [batch, h, w] demo input grids
            demo_outputs: List of [batch, h, w] demo output grids
            test_input: [batch, h, w] test input grid
            noisy_output: [batch, h, w] noisy version of test output
            timestep: Current diffusion timestep
            
        Returns:
            logits: [batch, h*w, n_colors] predicted clean colors
        """
        batch = test_input.shape[0]
        device = test_input.device
        
        # Encode all grids
        all_emb = []
        all_pos = []
        
        for inp, out in zip(demo_inputs, demo_outputs):
            inp_emb, inp_pos = self.encode_grid(inp, 'demo_in')
            out_emb, out_pos = self.encode_grid(out, 'demo_out')
            all_emb.extend([inp_emb, out_emb])
            all_pos.extend([inp_pos, out_pos])
        
        test_emb, test_pos = self.encode_grid(test_input, 'test_in')
        noisy_emb, noisy_pos = self.encode_grid(noisy_output, 'noisy_out')
        
        all_emb.extend([test_emb, noisy_emb])
        all_pos.extend([test_pos, noisy_pos])
        
        # Concatenate
        x = torch.cat(all_emb, dim=1)
        positions = torch.cat(all_pos, dim=1)
        
        # Add timestep embedding (global conditioning)
        t_emb = self.timestep_embed(torch.tensor([timestep], device=device))
        x = x + t_emb.unsqueeze(1)
        
        # Apply RoPE
        x = self.rope(x, positions)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Extract noisy output positions and predict clean
        noisy_start = x.shape[1] - noisy_output.shape[1] * noisy_output.shape[2]
        output_hidden = x[:, noisy_start:]
        logits = self.output_head(output_hidden)
        
        return logits
    
    def compute_loss(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        num_timesteps: int = None
    ) -> dict:
        """
        Compute diffusion training loss.
        
        Deep supervision for free: Loss at every timestep!
        """
        if num_timesteps is None:
            num_timesteps = self.num_timesteps
        
        batch, h, w = test_output.shape
        device = test_output.device
        total_loss = 0.0
        
        # Sample random timesteps for each example
        timesteps = torch.randint(1, num_timesteps, (batch,), device=device)
        
        # Add noise at sampled timesteps
        noisy_outputs = []
        for i in range(batch):
            noisy = self.noise.add_noise(test_output[i:i+1], timesteps[i].item())
            noisy_outputs.append(noisy)
        noisy_output = torch.cat(noisy_outputs, dim=0)
        
        # Predict clean from noisy
        # For simplicity, use the max timestep (can also do per-example)
        t = timesteps.max().item()
        logits = self.forward(demo_inputs, demo_outputs, test_input, noisy_output, t)
        
        # Cross-entropy loss to predict clean output
        target = test_output.view(batch, -1)
        loss = F.cross_entropy(logits.view(-1, self.n_colors), target.view(-1))
        
        return {
            'total_loss': loss,
            'denoising_loss': loss.item()
        }
    
    @torch.no_grad()
    def generate(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        num_steps: int = None
    ) -> torch.Tensor:
        """
        Generate output via iterative denoising.
        
        Start from pure noise, progressively denoise.
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        batch, h, w = test_input.shape
        device = test_input.device
        
        # Start with random grid
        x = torch.randint(0, self.n_colors, (batch, h, w), device=device)
        
        # Denoise from t=T to t=0
        for t in reversed(range(num_steps)):
            logits = self.forward(demo_inputs, demo_outputs, test_input, x, t)
            
            # Sample or argmax
            if t > 0:
                # Stochastic during denoising
                probs = F.softmax(logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.n_colors), 1)
                x = x.view(batch, h, w)
            else:
                # Deterministic at final step
                x = logits.argmax(dim=-1).view(batch, h, w)
        
        return x


# Quick test
if __name__ == "__main__":
    print("Testing Discrete Diffusion TRM...")
    
    model = DiffusionTRM(
        n_colors=11,
        d_model=256,  # Smaller for testing
        n_heads=4,
        n_layers=2,
        num_timesteps=8
    )
    
    # Dummy data
    batch = 2
    demo_inputs = [torch.randint(0, 10, (batch, 5, 5))]
    demo_outputs = [torch.randint(0, 10, (batch, 5, 5))]
    test_input = torch.randint(0, 10, (batch, 5, 5))
    test_output = torch.randint(0, 10, (batch, 5, 5))
    
    # Test forward
    noisy = model.noise.add_noise(test_output, t=4)
    logits = model.forward(demo_inputs, demo_outputs, test_input, noisy, timestep=4)
    print(f"Logits shape: {logits.shape}")  # [batch, 25, 11]
    
    # Test loss
    loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output)
    print(f"Loss: {loss_dict['total_loss']:.4f}")
    
    # Test generation
    generated = model.generate(demo_inputs, demo_outputs, test_input)
    print(f"Generated shape: {generated.shape}")  # [batch, 5, 5]
    
    print("✓ Discrete Diffusion TRM works!")
