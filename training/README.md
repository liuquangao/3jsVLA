# 3jsVLA-Tiny

This directory implements a small but structurally complete VLA in plain PyTorch. The code is
split by concept so a reader can follow every tensor from LeRobot v3 to the predicted action
chunk without entering a framework abstraction.

```text
top RGB --------> TinyVisionEncoder ---- visual tokens --+
gripper RGB ----> TinyVisionEncoder ---- visual tokens --+
language -------> TinyLanguageEncoder -- language tokens -+--> multimodal Transformer
joint state ----> Linear ---------------- state token -----+              |
                                                                          v
noise + time + action tokens --> self/cross-attention action expert --> action chunk
```

The two cameras share one vision encoder but receive separate modality embeddings. The action
expert learns a continuous flow from Gaussian noise to a 16-step joint-target chunk. During
inference, `sample_actions` integrates that flow in eight explicit Euler steps.

## Files

```text
tokenizer.py       string -> word IDs
vision.py          RGB -> spatial visual tokens
language.py        word IDs -> contextual language tokens
attention.py       explicit Q/K/V self-attention and cross-attention
model.py           multimodal context and action expert
flow_matching.py   training objective and Euler sampler
dataset.py         LeRobot v3 adapter and normalization
train.py           ordinary PyTorch training loop
```

## Data contract

Newly collected and converted runs must contain:

```text
observation.images.top
observation.images.gripper
observation.state
action
task
```

Runs collected before the dual-camera episode format do not contain the missing camera pixels
and cannot train this model without being recollected.

## Train

```powershell
.venv\Scripts\python -m training.train `
  --dataset-root data\<run-id>\lerobot_v3 `
  --repo-id local/3jsvla-a1z `
  --epochs 50 `
  --batch-size 16
```

The first milestone is deliberate overfitting on a small run. Generalization requires a larger
dual-camera dataset with varied cube layouts, instructions, lighting, and backgrounds.
