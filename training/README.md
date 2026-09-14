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

## Action space

`TinyVLAConfig.action_space` selects what the chunk means. The default, `delta`, predicts joint
movement relative to the observed pose, so the model answers "which way from here" instead of
having to infer absolute workspace coordinates from pixels. `absolute` predicts the joint angles
themselves.

Deltas need their own normalization: steps between poses are an order of magnitude smaller than
the poses, so reusing the dataset's `action` statistics would leave every target near zero and
the loss dominated by noise. `TinyVLADataset._delta_statistics` measures them over every frame
and chunk offset instead.

Drift is the usual objection to deltas, and closed-loop inference answers it: the collector
executes eight steps of each chunk and then observes again, so each chunk is anchored to a freshly
measured pose rather than to the model's own accumulated predictions. The checkpoint records which
space it was trained in, and `tools/policy_server.py` undoes exactly what training applied.

Measured, deltas are not yet an improvement. Two otherwise identical five-epoch runs over the
same 20-episode dataset, scored as mean absolute error against ground-truth joint targets on 24
held-out frames:

```text
             J1     J2     J3     J4     J5     J6   GRIP    MEAN     MAX
absolute   4.05   9.73   9.41   7.86   0.00   9.45   3.76    6.32   46.03
delta      3.97  12.28  14.44   8.89   0.00   4.92   3.04    6.79   91.24
```

Five epochs is far from convergence and one dataset of twenty episodes is thin, so this settles
very little. It does say the action space was not what made the early policies weak.

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
  --batch-size 16 `
  --action-space delta
```

The first milestone is deliberate overfitting on a small run. Generalization requires a larger
dual-camera dataset with varied cube layouts, instructions, lighting, and backgrounds.
