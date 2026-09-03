# NanoVLA

NanoVLA is a minimal Vision-Language-Action (VLA) project for learning how a VLA system works from end to end.

The goal is to let users build each part step by step:

1. Create a small robot environment with Three.js.
2. Control the robot and collect demonstrations in the browser.
3. Save images, language instructions, robot states, and actions.
4. Train a small policy with behavior cloning.
5. Run the policy in the environment and evaluate the result.

The first working component is a browser-based SO-101 data collector built with Three.js. It loads the official SO-101 URDF and STL meshes from TheRobotStudio.

## Run the Collector

```bash
npm install
npm run dev
```

Open the local URL shown by Vite. Adjust the six SO-101 controls, record an episode, replay it, and download the result as JSON.

## Minimal Task

The first task will be simple: move an object to a target location based on a language instruction.

```text
Instruction: "Move the red cube to the blue area."
Observation: RGB image + robot state
Action: Five joint targets + gripper target
```

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
├── public/assets/robots/so101/  # Official URDF and STL meshes
├── src/
│   ├── main.ts     # Three.js scene, SO-101 controls, and recorder
│   └── style.css   # Collector interface
├── index.html
└── package.json
```

Each downloaded episode is a JSON file containing:

```text
instruction
robot: "so101"
capture_hz: 5
frames[]
├── timestamp
├── observation.image
├── observation.joint_positions
└── action.joint_targets
```

## Roadmap

- [x] Create the Three.js tabletop environment
- [x] Load and control the official SO-101 URDF model
- [x] Record and replay joint-control demonstrations
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
