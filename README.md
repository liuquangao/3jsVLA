# 3jsVLA

3jsVLA is a compact, browser-based robot manipulation sandbox. It renders an A1Z arm with a G1Z
parallel gripper in Three.js, simulates contacts and rigid-body dynamics with Rapier, and lets you
drive the arm either joint by joint or by pointing at where the gripper should go.

The scene is randomised on demand: three coloured cubes, a yellow target circle, and an HDR
environment, all resampled whenever you ask for a new one.

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

## Features

- Three orthographic and perspective views, with no un-measurable scene view competing for room
- Three.js scene with the official A1Z/G1Z URDF and mesh assets
- Rapier rigid-body physics for the arm, gripper, cubes, and desk
- Constrained numerical IK with collision-aware vertical approaches
- Friction-based grasping using the physical finger geometry
- Gripper-mounted and central overview RGB cameras
- Point-and-click positioning of the gripper midpoint, with jaw and scene controls
- Scene and HDR randomisation on demand

## Pick a Point

The interface is built to be read and operated from pixels alone, by a person or by an agent
driving the mouse: everything needed for a decision is on screen at once, and every action is a
click. Two orthographic views carry the geometry, one from above and one from the side. Click a
point in each and the gripper midpoint is sent where the two rays cross.

There is deliberately no perspective view of the whole scene. Nothing can be measured from one —
the same pixel offset is millimetres near the camera and centimetres far from it — so it would
occupy the screen without answering any question the operator can act on. Dropping it leaves the
two measurable views half again as large, which is what a click's precision is made of, and the
page down to a single WebGL context.

```text
top view   ->  X and Z      (the ray is vertical)
side view  ->  Y            (the ray is horizontal)
```

Orthographic projection is what makes two clicks enough. Every ray through a pixel is parallel to
the view axis, so the coordinates read across the image do not depend on how far away the surface
under the cursor is. The same two clicks on a perspective camera would need a depth guess.

Both views are framed on the arm's reachable workspace and share one horizontal scale, so X lines
up between them. The top view's up vector is -Z, which keeps its X unmirrored and puts "away from
the side view's eye" at the top of the picture.

A third view, the centre camera, is reference only and cannot be clicked. The orthographic views
are the right instrument for *where* but a poor one for *which*: flattened and untextured, they
make a red cube and a green one hard to tell apart. The perspective camera answers that, and
leaves the measurable views to do the measuring.

Three things are drawn over the views, each answering a question that otherwise costs a whole
look-decide-click round trip:

- **The reachable sector**, outlined on the top view. A point outside it is refused by the IK, and
  without the outline the only way to discover that is to click and be told no.
- **`NOW` and `GOTO`**, the gripper's current position and the pending target, marked in all three
  views. A correction like "3 cm further left" needs to know where the gripper actually is.
- **Ticks every 5 cm** along both edges. Orthographic projection is linear, so a distance on
  screen is a fixed number of millimetres anywhere in the picture; printing the scale means the
  move can be worked out before clicking rather than clicked at and corrected.

The simulation runs on a timer rather than on `requestAnimationFrame`. rAF is tied to the
compositor: it stops when the window is unfocused or minimised, and in some embedded browsers it
never fires at all. A page meant to be driven by whoever is looking at it, including an agent
reading screenshots of a window it never focuses, cannot use the clock that stops when nobody is
watching.

After a move the panel reports the measured distance from the jaws to the point asked for, and
says when the arm has stopped moving. It deliberately does not report the IK residual, which this
once showed: that number answers "did the solver converge", which it always did, while looking
like it answers "will the jaws be where you asked", which it often was not. The solver knows
nothing about the table, so a point low enough for the fingers to bottom out solves perfectly and
is then missed by centimetres — and the old readout called that a 0.1 mm hit. Only the measured
pose can tell the difference, and when the shortfall is mostly vertical the panel says outright
that the jaws are resting on the table.

A straight-down approach is tried first, because that is how the jaws reach a table. Shallower
pitches down to 30° are fallbacks for points the wrist cannot stand over, and the angle that
worked is reported along with the residual between the solved pose and the point you asked for. A
point outside the workspace is reported as out of reach rather than silently approximated.

## Architecture

```text
      sliders ----\
                   >-- targetValues -- motors -- Rapier physics -- Three.js scene
   picked point --/        (joint commands)          (what actually happens)
```

- **Three.js** renders the scene, URDF meshes, cameras, lighting, and interface.
- **Rapier** computes arm, gripper, object, contact, and desk physics.
- **Constrained IK/QP** turns a task-space point into reachable joint angles.

Every input writes the same `targetValues`. Nothing downstream knows where a command came from,
so the arm behaves identically whatever drove it: motors chase the command and the lag comes from
the arm's own inertia rather than from any smoothing in the interface.

## Project Structure

```text
3jsVLA/
|-- src/
|   |-- main.ts             # scene, physics, task, picker
|   |-- constrained-ik.ts   # numerical IK and constraints
|   |-- contact-state.ts    # physical contact and grasp checks
|   `-- style.css           # interface
|-- public/assets/
|   |-- robots/             # URDF and robot meshes
|   |-- models/             # desk model
|   `-- hdri/               # randomised environments
|-- training/               # readable TinyVLA and flow-matching trainer
|-- docs/
|-- index.html
`-- package.json
```

## Current Scope

3jsVLA is an educational simulator, not a validated digital twin.

Earlier revisions recorded demonstration episodes, converted them to LeRobotDataset v3, and ran a
small VLA policy in the browser loop. That machinery was removed; `training/` is what remains of
it, and it reads datasets this repository no longer produces. The history has the collector, the
converter, and the policy runner if any of them is wanted back.

## License

A project license has not been selected yet.
