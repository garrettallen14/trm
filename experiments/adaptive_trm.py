"""
Adaptive Halting TRM - Novel Architecture Experiment

Instead of fixed N iterations, learn when to stop:
- Easy tasks: 2-4 iterations
- Hard tasks: 16-32 iterations
- Halting head predicts confidence

Based on PonderNet / Adaptive Computation Time (ACT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

import sys
sys.path.insert(0, '/Users/garrettallen/github/trm')
from src.model import TinyRecursiveModel


class HaltingHead(nn.Module):
    """Predicts probability of halting at current step."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, d_model] hidden states
        Returns:
            halt_prob: [batch] probability of halting
        """
        # Pool over sequence (mean)
        pooled = x.mean(dim=1)  # [batch, d_model]
        return self.net(pooled).squeeze(-1)  # [batch]


class AdaptiveTRM(nn.Module):
    """
    TRM with learned adaptive halting.
    
    Key idea: Don't waste compute on easy tasks.
    - Train halting head to predict when output is "good enough"
    - Use geometric prior (PonderNet) for stable training
    """
    
    def __init__(
        self,
        n_colors: int = 11,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        max_recursions: int = 32,
        lambda_p: float = 0.01,  # Ponder cost weight
        max_grid_size: int = 32
    ):
        super().__init__()
        
        # Base TRM (without its own recursion loop)
        self.base_model = TinyRecursiveModel(
            n_colors=n_colors,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_recursions=1,  # We control recursion externally
            max_grid_size=max_grid_size
        )
        
        self.d_model = d_model
        self.max_recursions = max_recursions
        self.lambda_p = lambda_p
        
        # Halting head
        self.halt_head = HaltingHead(d_model)
        
        # Geometric prior for halting (PonderNet)
        # p(halt at step n) = (1-p)^(n-1) * p
        self.register_buffer('halt_prior', self._compute_halt_prior(0.2, max_recursions))
    
    def _compute_halt_prior(self, p: float, n_steps: int) -> torch.Tensor:
        """Geometric distribution prior for halting."""
        probs = torch.zeros(n_steps)
        for n in range(n_steps):
            probs[n] = (1 - p) ** n * p
        probs[-1] = 1 - probs[:-1].sum()  # Ensure sums to 1
        return probs
    
    def forward(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        return_ponder_stats: bool = False
    ) -> dict:
        """
        Forward with adaptive halting.
        
        Returns weighted sum of outputs at each step,
        where weights are halting probabilities.
        """
        batch = test_input.shape[0]
        device = test_input.device
        
        # Initialize
        h, w = test_input.shape[1], test_input.shape[2]
        output = torch.zeros(batch, h * w, self.base_model.n_colors, device=device)
        
        # Track halting
        halt_probs = []  # p(halt) at each step
        outputs = []     # Output at each step
        halted = torch.zeros(batch, dtype=torch.bool, device=device)
        remaining_prob = torch.ones(batch, device=device)
        
        # Get initial encoding (demos + test)
        # We'll do this inside the base model
        
        for step in range(self.max_recursions):
            # Run one iteration through base model
            result = self.base_model(
                demo_inputs, demo_outputs, test_input,
                n_recursions=step + 1,  # Cumulative
                return_intermediates=True
            )
            
            step_logits = result['intermediates'][-1] if result['intermediates'] else result['logits']
            step_hidden = result.get('hidden', step_logits)  # Use logits as proxy if no hidden
            
            # Predict halting probability
            halt_p = self.halt_head(step_hidden.view(batch, -1, self.d_model))
            halt_probs.append(halt_p)
            outputs.append(step_logits)
            
            # Update running probability
            # p(stop at step n) = p(not stopped before) * p(halt now)
            step_halt = remaining_prob * halt_p
            remaining_prob = remaining_prob * (1 - halt_p)
            
            # Early stopping during inference
            if not self.training:
                halted = halted | (halt_p > 0.95)
                if halted.all():
                    break
        
        # Compute weighted output
        halt_probs = torch.stack(halt_probs, dim=1)  # [batch, steps]
        outputs = torch.stack(outputs, dim=1)  # [batch, steps, h*w, n_colors]
        
        # Normalize halting probs
        halt_probs = halt_probs / halt_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Weighted sum
        final_output = torch.einsum('bs,bshc->bhc', halt_probs, outputs)
        
        result = {
            'logits': final_output,
            'halt_probs': halt_probs,
            'n_steps': (step + 1),
            'expected_steps': (halt_probs * torch.arange(1, halt_probs.shape[1] + 1, device=device)).sum(dim=1).mean()
        }
        
        if return_ponder_stats:
            result['outputs'] = outputs
        
        return result
    
    def compute_loss(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor
    ) -> dict:
        """
        Compute loss with ponder cost regularization.
        
        Loss = task_loss + lambda_p * KL(halt_dist || geometric_prior)
        """
        result = self.forward(demo_inputs, demo_outputs, test_input, return_ponder_stats=True)
        
        batch, h, w = test_output.shape
        target = test_output.view(batch, -1)
        
        # Task loss (cross-entropy on final weighted output)
        task_loss = F.cross_entropy(
            result['logits'].view(-1, self.base_model.n_colors),
            target.view(-1)
        )
        
        # Ponder cost: KL divergence from geometric prior
        # Encourages the model to not ponder too long
        halt_dist = result['halt_probs'].mean(dim=0)  # Average over batch
        n_steps = halt_dist.shape[0]
        prior = self.halt_prior[:n_steps]
        prior = prior / prior.sum()  # Renormalize
        
        kl_loss = F.kl_div(
            (halt_dist + 1e-8).log(),
            prior,
            reduction='batchmean'
        )
        
        total_loss = task_loss + self.lambda_p * kl_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss.item(),
            'ponder_loss': kl_loss.item(),
            'expected_steps': result['expected_steps'].item()
        }


class SimpleAdaptiveTRM(nn.Module):
    """
    Simpler version: Just learn when to stop, no weighted sum.
    
    More like ACT (Adaptive Computation Time).
    """
    
    def __init__(
        self,
        n_colors: int = 11,
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        max_recursions: int = 32,
        halt_threshold: float = 0.95
    ):
        super().__init__()
        
        self.base_model = TinyRecursiveModel(
            n_colors=n_colors,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            n_recursions=1,
            max_grid_size=32
        )
        
        self.d_model = d_model
        self.max_recursions = max_recursions
        self.halt_threshold = halt_threshold
        
        # Confidence head (predicts if output is "good enough")
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor
    ) -> dict:
        """Forward with early stopping based on confidence."""
        
        steps_taken = 0
        
        for step in range(1, self.max_recursions + 1):
            result = self.base_model(
                demo_inputs, demo_outputs, test_input,
                n_recursions=step,
                return_intermediates=True
            )
            
            steps_taken = step
            
            # Check confidence
            if not self.training:
                hidden = result['logits']  # [batch, h*w, n_colors]
                pooled = hidden.mean(dim=1)  # [batch, n_colors]
                
                # Simple confidence: entropy of predictions
                probs = F.softmax(hidden, dim=-1)
                entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
                
                # Low entropy = high confidence
                if entropy < 0.5:  # Threshold
                    break
        
        return {
            'logits': result['logits'],
            'n_steps': steps_taken,
            'intermediates': result.get('intermediates', [])
        }
    
    def compute_loss(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        n_recursions: int = 8
    ) -> dict:
        """Standard TRM loss (use base model's method)."""
        return self.base_model.compute_loss(
            demo_inputs, demo_outputs, test_input, test_output,
            n_recursions=n_recursions
        )


# Quick test
if __name__ == "__main__":
    print("Testing Adaptive TRM...")
    
    # Test SimpleAdaptiveTRM first (easier)
    model = SimpleAdaptiveTRM(
        n_colors=11,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_recursions=16
    )
    
    batch = 2
    demo_inputs = [torch.randint(0, 10, (batch, 5, 5))]
    demo_outputs = [torch.randint(0, 10, (batch, 5, 5))]
    test_input = torch.randint(0, 10, (batch, 5, 5))
    test_output = torch.randint(0, 10, (batch, 5, 5))
    
    # Test forward
    model.eval()
    with torch.no_grad():
        result = model(demo_inputs, demo_outputs, test_input)
        print(f"Output shape: {result['logits'].shape}")
        print(f"Steps taken: {result['n_steps']}")
    
    # Test training loss
    model.train()
    loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output)
    print(f"Loss: {loss_dict['total_loss']:.4f}")
    
    print("✓ Simple Adaptive TRM works!")
    
    # Test full AdaptiveTRM
    print("\nTesting Full Adaptive TRM...")
    
    full_model = AdaptiveTRM(
        n_colors=11,
        d_model=256,
        n_heads=4,
        n_layers=2,
        max_recursions=8,
        lambda_p=0.01
    )
    
    # This will be slower, just test it runs
    try:
        result = full_model(demo_inputs, demo_outputs, test_input)
        print(f"Output shape: {result['logits'].shape}")
        print(f"Expected steps: {result['expected_steps']:.2f}")
        print("✓ Full Adaptive TRM works!")
    except Exception as e:
        print(f"Full model needs more work: {e}")
