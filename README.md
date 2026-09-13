# 3jsVLA

![3jsVLA browser-based robot data collector](docs/collector-ui.png)

3jsVLA is a compact, browser-based robot manipulation environment for learning the complete Vision-Language-Action data loop. It renders an A1Z arm with a G1Z parallel gripper in Three.js, simulates contacts and rigid-body dynamics with Rapier, and records camera observations, robot state, language instructions, and actions as training episodes.

The current task is simple and varied: **move the requested coloured cube into the yellow target area**. Cube positions, target position, requested colour, cube rotations, and HDR environment are randomised for every episode.

## Task Demo

The scripted controller approaches the requested cube, establishes a physical bilateral grasp, moves it to the target, releases it, and returns the arm to its neutral pose.

![A1Z robot moving a requested cube into the target area](docs/3jsvla-demo.gif)

## Features

- Three.js scene with the official A1Z/G1Z URDF and mesh assets
- Rapier rigid-body physics for the arm, gripper, cubes, and desk
- Constrained numerical IK with collision-aware vertical approaches
- Friction-based grasping using the physical finger geometry
- Gripper-mounted and central overview RGB cameras
- Manual control, recording, replay, and automatic demonstration generation
- Per-episode scene and HDR randomisation
- Direct output to the project `data/` directory through the local Vite server
- Conversion to LeRobotDataset v3 with `tools/to_lerobot.py`

## Quick Start

Requirements: Node.js 20 or newer and a modern desktop browser.

```bash
npm install
npm run dev
```

Open the URL printed by Vite, normally [http://127.0.0.1:5173](http://127.0.0.1:5173).

The default robot is A1Z + G1Z. The legacy SO-101 scene remains available at:

```text
http://127.0.0.1:5173/?robot=so101
```

## Generate Demonstrations

1. Enter the number of episodes in **Auto-generate**.
2. Keep **Output** set to **LeRobot v3 + raw**.
3. Press **Generate** and keep the page in the foreground.
4. Completed episodes are written to a timestamped directory under `data/`, then successful episodes are converted to LeRobotDataset v3 automatically.

Every episode starts and ends at the neutral command pose:

```text
J1-J6: 0 degrees
Gripper: 100% open
```

An episode is successful when the requested cube is released, resting on the table, and fully inside the yellow circle. Candidate scenes are checked against the complete IK pick-and-place path before an attempt begins, so unreachable layouts are resampled and never consume the attempt budget. Executed attempts that collide, lose the grasp, miss the target, or time out are discarded and retried. A batch stops only after collecting the requested number of successful episodes or reaching ten times that number of executed attempts. Write failures still stop immediately so data cannot be reported as saved when storage is unavailable.

## Dataset Layout

```text
data/<run-id>/
|-- meta.json
|-- episodes/                    # recoverable raw capture
    `-- episode_00000/
        |-- episode.json
        `-- frames/
            |-- top/000000.jpg
            `-- gripper/000000.jpg
`-- lerobot_v3/                  # training-ready Parquet, MP4 and metadata
```

Each frame contains:

```text
timestamp
observation.image_paths.top
observation.image_paths.gripper
observation.joint_positions
observation.object_poses
observation.grasped
action.joint_targets
```

`joint_positions` records the measured physical state. `joint_targets` records the command a policy should produce. Object poses and grasp state are ground truth for debugging and evaluation; they are not required as policy inputs.

## LeRobot v3 Output

Automatic conversion uses the official `LeRobotDataset` writer. Install its Python environment once before generating:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r tools\requirements.txt
```

On Linux or macOS, replace `.venv\Scripts\python` with `.venv/bin/python`. If the dependency is unavailable, collection still preserves the raw run and the page reports the setup command. Existing runs can always be converted manually:

```powershell
.venv\Scripts\python tools\to_lerobot.py data\<run-id> --repo-id you/3jsvla-a1z --root data\<run-id>\lerobot_v3 --success-only
```

Without `--root` the dataset is written to the Hugging Face cache instead of the run directory, so pass it explicitly to keep the layout above.

Reading the resulting MP4s needs FFmpeg's shared libraries on `PATH`. Without them `torchcodec` cannot load and LeRobot falls back to a slower `pyav` decoder after printing a long traceback, which is noisy but harmless. On Windows:

```powershell
winget install --id BtbN.FFmpeg.GPL.Shared.7.1
```

Pick a *shared* build. The static builds ship only `ffmpeg.exe`, not the `avutil`/`avcodec` libraries `torchcodec` loads.

## Architecture

```text
Instruction + RGB cameras + joint state
                    |
                    v
               VLA policy
                    | action
                    v
Three.js renderer - Robot controller - Rapier physics
        ^                                  |
        `----------- observation ----------'
```

- **Three.js** renders the scene, URDF meshes, cameras, lighting, and interface.
- **Rapier** computes arm, gripper, object, contact, and desk physics.
- **Constrained IK/QP** plans reachable task-space waypoints.
- **The recorder** samples RGB, state, and next-step actions at 10 Hz.
- **Python tooling** converts browser output into a training-ready LeRobot dataset.

## Train 3jsVLA-Tiny

The educational TinyVLA keeps every important part visible in plain PyTorch: two-camera visual
tokens, language tokens, a state token, multimodal self-attention, and a flow-matching action
expert. It does not use Hugging Face Transformers or a hidden trainer.

```powershell
.venv\Scripts\python -m training.train `
  --dataset-root data\<run-id>\lerobot_v3 `
  --repo-id local/3jsvla-a1z
```

The model is about 9M parameters and trains on whatever device is available, falling back to CPU
when no CUDA device is present. CPU training is viable for the overfitting milestone but slow: on
eight cores, one epoch over a 20-episode run (3,946 samples, batch 16) takes roughly seven minutes,
and most of that is the vision encoder rather than video decoding. Lower the image resolution in
`TinyVLAConfig` when iterating locally.

See `training/README.md` for the architecture and training controls.

## Run a Policy in the Browser

Checkpoints written to `checkpoints/` appear in the collector's **Policy inference** panel. Pick
one and press **Run policy**: the page sends both camera images, the measured joint state and the
current instruction to the dev server, which keeps a `tools/policy_server.py` child alive holding
the loaded model, and writes the returned joint targets straight into the same `targetValues` the
sliders drive.

The policy predicts a chunk of sixteen absolute joint targets. Only the first eight are executed
before the scene is observed again, so tracking error and physics disturbances are corrected
rather than accumulated. On CPU a chunk takes about a second, which is slower than the eight
steps it covers, so the arm pauses briefly between chunks.

Inference reuses the loaded process across steps because loading a checkpoint costs seconds while
a chunk costs a fraction of one. **Stop** ends the run and releases the model.

## Project Structure

```text
3jsVLA/
|-- src/
|   |-- main.ts             # scene, physics, task, controller, recorder
|   |-- constrained-ik.ts   # numerical IK and constraints
|   |-- contact-state.ts    # physical contact and grasp checks
|   `-- style.css           # collector interface
|-- public/assets/
|   |-- robots/             # URDF and robot meshes
|   |-- models/             # desk model
|   `-- hdri/               # randomised environments
|-- tools/
|   |-- to_lerobot.py
|   |-- policy_server.py    # loaded checkpoint, JSON lines over stdio
|   |-- policy-server.mjs   # dev-server routes for the inference panel
|   `-- requirements.txt
|-- training/                 # readable TinyVLA and flow-matching trainer
|-- docs/
|-- data/                   # generated runs
|-- index.html
`-- package.json
```

## Current Scope

3jsVLA is an educational simulator and data-generation project, not a validated digital twin. The scripted controller is useful for producing demonstrations, but difficult layouts can still fail because of IK reach, tracking error, collision, or unstable frictional grasping.

The TinyVLA trainer exists and the loop from collection through training to browser inference runs end to end, but no policy has been trained to the point of being useful yet. A five-epoch checkpoint moves the arm in plausible directions without completing the task. Planned next steps are training to convergence on a larger dataset and adding repeatable evaluation metrics, so policy quality is measured rather than watched.

## License

A project license has not been selected yet.

