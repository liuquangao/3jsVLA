# 3jsVLA

3jsVLA is a minimal Vision-Language-Action (VLA) project for learning how a VLA system works from end to end.

The goal is to let users build each part step by step:

1. Create a small robot environment with Three.js.
2. Control the robot and collect demonstrations in the browser.
3. Save images, language instructions, robot states, and actions.
4. Train a small policy with behavior cloning.
5. Run the policy in the environment and evaluate the result.

The first working component is a browser-based data collector built with Three.js and Rapier. It
loads the official Galaxea (星海图) A1Z URDF with the G1Z parallel gripper, using the vendor's own
STL meshes and joint names (`arm_joint1` … `arm_joint6`), and puts it in front of three coloured
cubes and two target zones that it can actually pick up and put down.

## Run the Collector

```bash
npm install
npm run dev
```

Open the local URL shown by Vite. Set a count, press **Generate**, and it collects that many
episodes on its own, then hands you the dataset as JSON. The joint sliders are still there to
record an episode by hand, but they are for seeing how the loop works, not for bulk collection.

The robot is chosen by a query parameter, so the earlier SO-101 arm is still one URL away:

```text
http://localhost:5173/            # Galaxea A1Z + G1Z (default)
http://localhost:5173/?robot=so101   # SO-101
```

Both arms are entries in the `ROBOTS` table at the top of `src/main.ts`. An entry declares its
URDF, where the base is mounted, the two camera framings, and one slider per control — so adding a
third arm means adding a `RobotSpec`, not editing the collector.

## The Task

Move a named cube to a named zone. Three cubes (red, green, blue) and two zones (a circle and a
square) give six distinct tasks that all share the same scene, and every episode re-scatters the
cubes and zones and re-rolls the instruction:

```text
Instruction: "Move the green cube to the square."
Observation: RGB image + robot state
Action:      Six joint targets + gripper target
Success:     named cube at rest inside the named zone, no longer held
```

The point of three cubes and two zones is that **the instruction has to be read**. With a single
cube and a single target, a policy scores perfectly while ignoring the language entirely, and
nothing in the training curves tells you the language channel is dead. Here the same image maps
to different actions depending on the sentence.

## Generating Data

Demonstrations are scripted, not teleoperated. Dragging seven sliders produces trajectories that
move one joint at a time, and a policy trained on those learns slider-wiggling rather than
reaching — so the collector drives itself instead.

Each episode: reach over the named cube, drop onto it, close, lift, carry to the named zone,
lower, let go. Every episode re-rolls the layout, the instruction, the approach pitch, the hover
and drop heights, a few millimetres of aim error on the grasp, up to 4 cm on where in the zone it
aims, and the angle the cube is set down at — so a batch is a spread of demonstrations rather than
one motion repeated.

On this machine it runs at roughly **one episode per second of wall time**, 100% success, giving
4–7 second episodes of 21–37 frames each.

Pressing Generate asks for a folder and streams episodes into it as they finish, so memory stays
flat however many you ask for. Browsers without the File System Access API (Firefox, Safari) fall
back to collecting in memory and handing the lot over as one JSON download.

Generation is driven from the animation loop in ~8 ms slices, so the page keeps rendering and you
can watch it collect. That also means it needs a **foreground tab** — a backgrounded tab stops
getting animation frames and generation stalls until you come back.

Three knobs worth knowing, all constants in `src/main.ts`:

| Knob | Does what |
| --- | --- |
| `ARM_SPEED` | Demonstration speed. At 60 deg/s and 5 Hz capture there are ~12° between consecutive actions. |
| `CAPTURE_HZ` | Raise it if a policy needs finer action steps than that. |
| `jitter(...)` in `buildScript` | Aim error. Widen it to breed failures into the dataset; right now essentially every episode succeeds. |

## Getting the Data Out

The collector writes a deliberately dumb tree — one directory per episode, frames as plain JPEGs:

```text
<folder you picked>/
├── meta.json                       # robot, capture_hz, image size, joint names, counts
└── episodes/episode_00000/
    ├── episode.json                # instruction, task, success, per-frame state/action
    └── frames/000000.jpg …
```

`tools/to_lerobot.py` turns that into a **LeRobotDataset v3.0**:

```bash
python3 -m venv .venv
.venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/to_lerobot.py <folder> --repo-id you/nanovla-a1z --root ./lerobot_out
```

It drives `LeRobotDataset` itself rather than writing Parquet and MP4 by hand, so the on-disk
format stays correct without the script having to know what it looks like — parquet shards, AV1
video, episode offsets and normalisation stats are all lerobot's job. The result loads straight
back:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("you/nanovla-a1z", root="./lerobot_out")
ds[20]["observation.images.front"]  # torch.float32 [3, 192, 256], decoded from the mp4
ds[20]["task"]                      # "Move the red cube to the circle."
```

Two things worth knowing before you install: lerobot pins `torch<2.12` and installing it plainly
drags in ~3 GB of CUDA libraries this converter never uses — `tools/requirements.txt` documents
the CPU-only route, which lands at 1.7 GB instead of 5 GB. And v3.0 needs `lerobot >= 0.4.0` and
Python >= 3.12.

Episode ground truth (`object_poses`, `grasped`) is kept in the dump and ignored by the
converter; it is there for debugging and for scoring an evaluation run, not for training.

## Inverse Kinematics

Scripting the arm needs IK, and the A1Z's geometry hands you a closed form. J1 takes the azimuth
and J2/J3/J4 all turn about Y, so once the tool pitch is fixed what remains is a planar two-link
problem. Working in the (radial, height) plane as complex numbers — where a turn of q about Y is
a multiply by e^-iq — it solves exactly, with no iteration. `solveA1Z` in `src/main.ts` is about
forty lines and lands the jaw on target to within a rounding error.

Height costs pitch, and it costs a lot at the edge of the workspace:

| Radius | Max jaw height, tool straight down | at 80° | at 70° |
| --- | --- | --- | --- |
| 0.30 m | 0.108 m | 0.157 m | 0.207 m |
| 0.36 m | 0.094 m | 0.153 m | 0.212 m |
| 0.43 m | 0.057 m | 0.131 m | 0.199 m |

J4 tops out at ±75° and the wrist has no offset to make up the difference, so a far-out top-down
grasp cannot lift its approach very high. The script therefore prefers a vertical grasp and tilts
only as far as the reach demands, sweeping the pitch down from a steep start until every waypoint
solves.

## Physics

Rapier simulates the props only. The cubes are dynamic rigid bodies with box colliders, the table
is a static box, and objects fall, topple, slide, collide with each other and roll out of a zone
if you drop them badly — all of which lands in the dataset as failure modes a policy can learn to
recover from.

The arm stays kinematic: it is driven straight from the joint controls and carries no colliders.
That keeps the recorded actions clean and avoids the soft, drooping serial chain that an impulse
solver gives you for a force-driven arm. Grasping is a kinematic attach for the same reason — the
cube snaps to the jaw when the gripper closes on it, which never jitters and makes the pickup
moment obvious. Releasing hands the cube back to the physics world carrying the jaw's velocity.

The trade-off is that the arm is infinitely stiff and passes through anything it is not gripping.
Giving the links colliders is the natural next step if that starts to matter.

The first version will use:

- One robot arm
- One RGB camera
- A small tabletop scene
- A few movable objects
- Behavior cloning
- Task success rate for evaluation

## System Overview

```text
Language instruction ─┐
Camera image ─────────┼─> VLA policy ─> Robot action ─> Three.js environment
Robot state ──────────┘                                      │
        ^____________________________________________________|
```

Three.js provides the scene, camera, rendering, and browser interface. Simple kinematics and collision handling provide robot interaction. Python handles dataset loading, training, and model inference.

## Project Structure

```text
NanoVLA/
├── tools/
│   ├── to_lerobot.py            # dump -> LeRobotDataset v3.0
│   └── requirements.txt
├── public/assets/robots/
│   ├── a1z/                     # Galaxea A1Z + G1Z URDF and STL meshes
│   └── so101/                   # SO-101 URDF and STL meshes
├── src/
│   ├── main.ts     # Robot table, scene, physics, task sampling, controls, recorder
│   └── style.css   # Collector interface
├── index.html
└── package.json
```

Each downloaded episode is a JSON file containing:

```text
instruction: "Move the green cube to the square."
robot: "a1z"
task: { object: "green", target: "square" }
capture_hz: 5
success: true
frames[]
├── timestamp
├── observation.image             # 256 x 192 JPEG data URL
├── observation.joint_positions  # where the arm is
├── observation.object_poses     # ground truth, for debugging and scoring
├── observation.grasped          # which cube is in the gripper, or null
└── action.joint_targets         # where the arm was told to go
```

`joint_positions` is the state and `joint_targets` is the action. In a generated episode the
action is the commanded pose one capture interval ahead — what a policy would have to emit at
that frame to produce the motion that follows.

Generated episodes go to a folder as the tree above, or — where the browser cannot write to one —
come back as a single file of the same episodes under `format: "nanovla.dataset.v1"`.

Generated episodes run on a fixed simulated clock rather than wall time, so the dataset comes out
the same however fast the machine is, and a stalled tab cannot stretch a trajectory.

## Roadmap

- [x] Create the Three.js tabletop environment
- [x] Load and control the official A1Z + G1Z URDF model
- [x] Record and replay joint-control demonstrations
- [x] Simulate the props with Rapier and grasp them
- [x] Randomise the scene and make the instruction disambiguating
- [x] Closed-form IK and a scripted policy that generates episodes on its own
- [x] Stream episodes to disk and convert them to LeRobotDataset v3.0
- [ ] Train a small behavior-cloning policy
- [ ] Connect the policy to the browser environment
- [ ] Evaluate the complete closed-loop system

## Design Goals

- Keep every component small and understandable.
- Make intermediate data easy to inspect.
- Finish one complete task before adding complexity.
- Allow the environment, robot, and model to be replaced independently.
- Provide one tutorial for each stage of the system.

## License

A license has not been selected yet.
