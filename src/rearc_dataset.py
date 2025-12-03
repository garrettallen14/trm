"""
RE-ARC Dataset - generates synthetic ARC tasks on-the-fly.

No pre-generation needed - creates fresh samples every epoch.
"""

import random
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset


class ReARCDataset(Dataset):
    """
    Dataset that generates synthetic ARC tasks using RE-ARC.
    
    Generates fresh samples on-the-fly, so every epoch sees new data.
    This is better for generalization than pre-generated data.
    """
    
    def __init__(
        self,
        size: int = 20000,  # Number of samples per epoch
        seed: Optional[int] = None,
        max_grid_size: int = 30
    ):
        self.size = size
        self.max_grid_size = max_grid_size
        self.seed = seed
        self._generators = None
        self._task_ids = None
        
        # Try to load RE-ARC
        self._load_rearc()
    
    def _load_rearc(self):
        """Load RE-ARC generators."""
        import sys
        
        rearc_path = Path("data/re-arc")
        if not rearc_path.exists():
            print("RE-ARC not found. Run: git clone https://github.com/michaelhodel/re-arc.git data/re-arc")
            self._generators = None
            self._task_ids = []
            return
        
        # Add RE-ARC to path (flat module structure, not a package)
        rearc_abs = str(rearc_path.resolve())
        if rearc_abs not in sys.path:
            sys.path.insert(0, rearc_abs)
        
        try:
            # RE-ARC has flat structure: generators.py contains the generators dict
            import generators as rearc_generators
            
            # generators.py should have a dict mapping task_id -> generator function
            if hasattr(rearc_generators, 'generators'):
                self._generators = rearc_generators.generators
                self._task_ids = list(self._generators.keys())
                print(f"RE-ARC loaded: {len(self._task_ids)} task generators")
            else:
                # Maybe it's the module itself with functions
                self._task_ids = [name for name in dir(rearc_generators) 
                                  if not name.startswith('_') and callable(getattr(rearc_generators, name))]
                self._generators = {tid: getattr(rearc_generators, tid) for tid in self._task_ids}
                print(f"RE-ARC loaded: {len(self._task_ids)} task generators (function mode)")
                
        except Exception as e:
            print(f"RE-ARC import failed: {e}")
            import traceback
            traceback.print_exc()
            self._generators = None
            self._task_ids = []
    
    def __len__(self):
        return self.size if self._generators else 0
    
    def __getitem__(self, idx):
        if not self._generators:
            raise RuntimeError("RE-ARC not loaded")
        
        # Pick a random task generator
        task_id = random.choice(self._task_ids)
        generator = self._generators[task_id]
        
        try:
            # Generate a random example
            example = generator()
            
            # RE-ARC returns dict with 'train' and 'test' lists
            train_pairs = example.get('train', [])
            test_pairs = example.get('test', [])
            
            if not train_pairs or not test_pairs:
                # Fallback to another task if this one failed
                return self.__getitem__((idx + 1) % self.size)
            
            # Convert to tensors
            demo_inputs = []
            demo_outputs = []
            
            for pair in train_pairs:
                inp = torch.tensor(pair['input'], dtype=torch.long)
                out = torch.tensor(pair['output'], dtype=torch.long)
                demo_inputs.append(inp)
                demo_outputs.append(out)
            
            # Use first test pair
            test_pair = test_pairs[0]
            test_input = torch.tensor(test_pair['input'], dtype=torch.long)
            test_output = torch.tensor(test_pair['output'], dtype=torch.long)
            
            return {
                'demo_inputs': demo_inputs,
                'demo_outputs': demo_outputs,
                'test_input': test_input,
                'test_output': test_output,
                'task_id': f"rearc_{task_id}"
            }
            
        except Exception as e:
            # If generation fails, try another task
            return self.__getitem__((idx + 1) % self.size)


def get_rearc_dataset(size: int = 20000) -> Optional[ReARCDataset]:
    """Get RE-ARC dataset if available."""
    dataset = ReARCDataset(size=size)
    if dataset._generators:
        return dataset
    return None
