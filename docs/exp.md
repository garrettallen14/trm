root@f792b46500db:/workspace/trm# uv run python experiments/train_diffusion.py \
  --d_model 256 \
  --batch_size 64 \
  --grad_accum 2 \
  --augment_factor 100 \
  --num_timesteps 16 \
  --max_demos 1 \
  --epochs 100 \
  --amp \
  --dashboard
warning: The `tool.uv.dev-dependencies` field (used in `pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
Device: cuda
GPU: NVIDIA A40
VRAM: 47.6 GB
GPU optimizations: TF32 + cuDNN benchmark enabled
Dashboard: http://localhost:3000 (in-process)

============================================================
DISCRETE DIFFUSION TRM TRAINING
============================================================
  d_model: 256
  n_heads: 4
  n_layers: 2
  n_colors: 11
  max_grid_size: 32
  max_demos: 1
  num_timesteps: 16
  noise_schedule: cosine
  self_conditioning: False
  lr: 0.0001
  lr_embed_mult: 100.0
  weight_decay: 0.01
  warmup_epochs: 5
  epochs: 100
  batch_size: 64
  grad_accum: 2
  clip_grad: 1.0
  augment_factor: 100
  use_amp: True
  compile_model: False
  gradient_checkpointing: False

Parameters: 1,718,795
Embed params: 135,424
Other params: 1,583,371
Using mixed precision (AMP)
Loaded 400 tasks from data/arc-agi-1/data/training
Training samples: 40,000

Saving to: experiments/runs/diffusion_20251202_235806

============================================================
TRAINING
============================================================
Epoch 1/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:03<00:00,  5.04it/s, loss=0.3363, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 1: loss=0.7237, task_acc=0.0%, cell_acc=24.7%, lr=4.00e-05, time=124s, oom=0
Epoch 2/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.15it/s, loss=0.4648, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 2: loss=0.3718, task_acc=0.0%, cell_acc=43.6%, lr=6.00e-05, time=121s, oom=0
Epoch 3/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.15it/s, loss=0.2887, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 3: loss=0.3041, task_acc=0.0%, cell_acc=65.4%, lr=8.00e-05, time=121s, oom=0
Epoch 4/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.16it/s, loss=0.2258, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 4: loss=0.2769, task_acc=0.0%, cell_acc=65.2%, lr=1.00e-04, time=121s, oom=0
Epoch 5/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.14it/s, loss=0.2172, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
  Evaluating on ARC-AGI-2...
Loaded 120 tasks from data/arc-agi-2/data/evaluation
  AGI-2: task_acc=0.0%, cell_acc=59.3%
Epoch 5: loss=0.2614, task_acc=0.0%, cell_acc=69.5%, lr=1.00e-04, time=122s, oom=0
Epoch 6/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.15it/s, loss=0.1957, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 6: loss=0.2563, task_acc=0.5%, cell_acc=57.0%, lr=1.00e-04, time=121s, oom=0
Epoch 7/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:00<00:00,  5.18it/s, loss=0.3114, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 7: loss=0.2498, task_acc=0.0%, cell_acc=57.3%, lr=9.99e-05, time=121s, oom=0
Epoch 8/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.15it/s, loss=0.1993, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 8: loss=0.2434, task_acc=0.0%, cell_acc=51.6%, lr=9.98e-05, time=121s, oom=0
Epoch 9/100: 100%|████████████████████████████████████████████████████████████████████████████████| 625/625 [02:00<00:00,  5.17it/s, loss=0.3129, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 9: loss=0.2399, task_acc=0.0%, cell_acc=71.0%, lr=9.96e-05, time=121s, oom=0
Epoch 10/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.15it/s, loss=0.2913, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
  Evaluating on ARC-AGI-2...
Loaded 120 tasks from data/arc-agi-2/data/evaluation
  AGI-2: task_acc=0.0%, cell_acc=39.2%
Epoch 10: loss=0.2355, task_acc=0.0%, cell_acc=49.8%, lr=9.93e-05, time=121s, oom=0
Epoch 11/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:00<00:00,  5.18it/s, loss=0.2666, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 11: loss=0.2387, task_acc=0.0%, cell_acc=65.0%, lr=9.90e-05, time=121s, oom=0
Epoch 12/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.14it/s, loss=0.2582, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 12: loss=0.2306, task_acc=0.0%, cell_acc=60.1%, lr=9.87e-05, time=122s, oom=0
Epoch 13/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.16it/s, loss=0.2203, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 13: loss=0.2324, task_acc=0.5%, cell_acc=64.7%, lr=9.83e-05, time=121s, oom=0
Epoch 14/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:00<00:00,  5.18it/s, loss=0.2231, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 14: loss=0.2309, task_acc=0.5%, cell_acc=58.3%, lr=9.78e-05, time=121s, oom=0
Epoch 15/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:01<00:00,  5.16it/s, loss=0.1938, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
  Evaluating on ARC-AGI-2...
Loaded 120 tasks from data/arc-agi-2/data/evaluation
  AGI-2: task_acc=0.0%, cell_acc=30.1%
Epoch 15: loss=0.2247, task_acc=0.0%, cell_acc=54.9%, lr=9.73e-05, time=121s, oom=0
Epoch 16/100: 100%|███████████████████████████████████████████████████████████████████████████████| 625/625 [02:00<00:00,  5.17it/s, loss=0.2515, oom=0]
Loaded 400 tasks from data/arc-agi-1/data/evaluation
Epoch 16: loss=0.2261, task_acc=0.0%, cell_acc=64.0%, lr=9.67e-05, time=121s, oom=0
Epoch 17/100:  47%|█████████████████████████████████████▍                                         | 296/625 [00:57<01:03,  5.17it/s, loss=0.2454, oom=0]