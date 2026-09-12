import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import URDFLoader, { type URDFLink, type URDFRobot } from "urdf-loader";
import "./style.css";

type JointValues = Record<string, number>;

type EpisodeFrame = {
  timestamp: number;
  observation: {
    image: string;
    joint_positions: JointValues;
    /** Ground truth, for debugging and for scoring an evaluation run. */
    object_poses: Array<{
      id: PropId;
      position: [number, number, number];
      quaternion: [number, number, number, number];
    }>;
    grasped: PropId | null;
  };
  action: {
    joint_targets: JointValues;
  };
};

const round4 = (value: number) => Number(value.toFixed(4));

/** One slider. `urdfJoints` are the URDF joints it drives, `toURDF` maps slider units to them. */
type JointSpec = {
  name: string;
  label: string;
  min: number;
  max: number;
  initial: number;
  unit: "deg" | "percent";
  urdfJoints: string[];
  /** Slider units to URDF units, and back again for reading the arm's measured pose. */
  toURDF: (value: number) => number;
  fromURDF: (value: number) => number;
};

type Vec3 = [number, number, number];

/** Per-link surface override, for URDFs that ship one flat colour for the whole robot. */
type LinkFinish = { color: number; roughness: number; metalness: number };

type RobotSpec = {
  id: string;
  label: string;
  urdf: string;
  packages: Record<string, string>;
  /** Where the URDF base_link sits on the table, in scene coordinates. */
  mount: Vec3;
  view: {
    camera: Vec3;
    target: Vec3;
    distance: [number, number];
    obsCamera: Vec3;
    obsTarget: Vec3;
  };
  /** Keyed by URDF link name. Links absent here keep the colour the URDF declares. */
  finish: Record<string, LinkFinish>;
  /** Where this arm holds an object, and how closed the gripper has to be to hold it. */
  grasp: {
    /** URDF link the jaw anchor hangs off. */
    link: string;
    /** Jaw centre in that link's frame. */
    anchor: Vec3;
    /** Half-extents of the graspable pocket around the anchor, in that link's axes. */
    region: Vec3;
    /**
     * Jaw opening in metres at control 100, when that is a linear function of the control. Given
     * it, the closed position is derived from the cube instead of guessed — guessing is exactly
     * how the fingers ended up 9 mm inside it.
     */
    fullOpening?: number;
    /** Fallback closed position for grippers whose opening is not linear in the control. */
    closedAt?: number;
  };
  /**
   * Props are scattered into an annulus in front of the base. `min`/`max` are radii in metres
   * and `yaw` a half-angle in radians; the band has to stay inside the arm's top-down grasp
   * envelope, and be roomy enough for three cubes to sit apart without overlapping.
   */
  reach: { min: number; max: number; yaw: number };
  /** Cube edge length, which must stay under the gripper's jaw opening. */
  props: { cube: number };
  /**
   * Closed-form IK, when this arm has one. Without it the scripted generator is unavailable
   * and the arm is joint-slider only. `pitch` is how far the tool points below horizontal and
   * `jawAzimuth` which way the jaw opening faces, both in radians.
   */
  solve?: (mount: Vec3, target: THREE.Vector3, pitch: number, jawAzimuth: number) => JointValues | null;
  joints: JointSpec[];
};

const revolute = (value: number) => THREE.MathUtils.degToRad(value);
const measured = (value: number) => THREE.MathUtils.radToDeg(value);

/**
 * Closed-form IK for the A1Z. J1 takes the azimuth and J2/J3/J4 all turn about Y, so once the
 * tool pitch is fixed what is left is a planar two-link problem with an exact solution. Working
 * in the (radial, height) plane as complex numbers, a turn of q about Y is a multiply by e^-iq.
 */
const A1Z_LINKS = {
  shoulder: new THREE.Vector2(0.02, 0.118), // joint 2, from the URDF base
  upper: new THREE.Vector2(-0.264, 0), // joint 2 -> joint 3
  fore: new THREE.Vector2(0.245, 0.06), // joint 3 -> joint 4
  wrist: new THREE.Vector2(0.074 + 0.0235 + 0.17, 0), // joint 4 -> jaw anchor, with J5 at zero
};

/** Wraps an angle into the joint's range using the cube's four-fold symmetry. */
function wrapIntoLimit(angle: number, limit: number, step: number) {
  let candidate = THREE.MathUtils.euclideanModulo(angle + Math.PI, Math.PI * 2) - Math.PI;
  while (candidate > limit) candidate -= step;
  while (candidate < -limit) candidate += step;
  return Math.abs(candidate) <= limit ? candidate : null;
}

function solveA1Z(mount: Vec3, target: THREE.Vector3, pitch: number, jawAzimuth: number) {
  const dx = target.x - mount[0];
  const dz = target.z - mount[2];
  const azimuth = Math.atan2(-dz, dx);
  const radial = Math.hypot(dx, dz);

  const { shoulder, upper, fore, wrist } = A1Z_LINKS;
  // Peel the wrist off the target, leaving the two-link problem for J2 and J3.
  const rest = new THREE.Vector2(radial, target.y - mount[1])
    .sub(shoulder)
    .sub(new THREE.Vector2(wrist.x * Math.cos(pitch), -wrist.x * Math.sin(pitch)));

  const scale = upper.length() * fore.length();
  const cosine = (rest.lengthSq() - upper.lengthSq() - fore.lengthSq()) / (2 * scale);
  if (Math.abs(cosine) > 1) return null; // out of reach

  // phase of conj(upper) * fore, the built-in bend between the two link vectors
  const phase = Math.atan2(upper.x * fore.y - upper.y * fore.x, upper.x * fore.x + upper.y * fore.y);

  for (const elbow of [1, -1]) {
    const q3 = phase - elbow * Math.acos(cosine);
    // upper + e^-iq3 * fore
    const arm = new THREE.Vector2(
      upper.x + fore.x * Math.cos(q3) + fore.y * Math.sin(q3),
      upper.y - fore.x * Math.sin(q3) + fore.y * Math.cos(q3),
    );
    if (arm.lengthSq() < 1e-12) continue;
    const q2 = -(Math.atan2(rest.y, rest.x) - Math.atan2(arm.y, arm.x));
    const q4 = pitch - q2 - q3;
    // The jaw opening ends up facing pi/2 - J6 + J1, so invert that for the requested azimuth.
    const q6 = wrapIntoLimit(Math.PI / 2 + azimuth - jawAzimuth, revolute(115), Math.PI / 2);
    if (q6 === null) continue;

    const solution: JointValues = {
      arm_joint1: THREE.MathUtils.radToDeg(azimuth),
      arm_joint2: THREE.MathUtils.radToDeg(q2),
      arm_joint3: THREE.MathUtils.radToDeg(q3),
      arm_joint4: THREE.MathUtils.radToDeg(q4),
      arm_joint5: 0,
      arm_joint6: THREE.MathUtils.radToDeg(q6),
    };
    const withinLimits = A1Z.joints.every((joint) => {
      const value = solution[joint.name];
      return value === undefined || (value >= joint.min && value <= joint.max);
    });
    if (withinLimits) return solution;
  }
  return null;
}

/**
 * Galaxea (星海图) A1Z with the G1Z parallel gripper. Control names are the vendor SDK's
 * own joint names, so a recorded episode maps straight onto the real arm.
 * The 30 mm per-finger stroke is our own figure — see the asset's SOURCE.md.
 */
const A1Z_FINGER_STROKE = 0.03;

const A1Z: RobotSpec = {
  id: "a1z",
  label: "A1Z + G1Z",
  urdf: "/assets/robots/a1z/A1Z_G1Z.urdf",
  packages: { A1Z_G1Z: "/assets/robots/a1z" },
  mount: [-0.36, 0.001, 0],
  // Framed low enough that the backdrop reads as a room rather than just its floor, while the
  // observation still fills with the reachable patch of desk.
  view: {
    camera: [1.28, 0.82, 1.28],
    target: [-0.14, 0.05, 0],
    distance: [0.9, 5],
    obsCamera: [0.98, 0.62, 0.98],
    obsTarget: [-0.16, 0.05, 0],
  },
  // The vendor URDF paints all nine links the same SolidWorks default grey, so the arm
  // arrives colourless. These are read off Galaxea's own product render: black anodised
  // base and wrist, charcoal shoulder and upper arm, brushed-aluminium forearm.
  finish: {
    base_link: { color: 0x15161a, roughness: 0.42, metalness: 0.36 },
    arm_link1: { color: 0x3c4147, roughness: 0.45, metalness: 0.4 },
    arm_link2: { color: 0x4b5158, roughness: 0.47, metalness: 0.34 },
    arm_link3: { color: 0xc4c9cf, roughness: 0.31, metalness: 0.64 },
    arm_link4: { color: 0x42474e, roughness: 0.45, metalness: 0.4 },
    arm_link5: { color: 0x1b1e22, roughness: 0.4, metalness: 0.42 },
    arm_link6: { color: 0x1b1e22, roughness: 0.4, metalness: 0.42 },
    gripper_finger_left_link: { color: 0x272b2f, roughness: 0.5, metalness: 0.3 },
    gripper_finger_rIght_link: { color: 0x272b2f, roughness: 0.5, metalness: 0.3 },
  },
  // The G1Z jaw tips meet at x = 0.183 in the arm_link6 frame and open to 60 mm at control 100.
  grasp: {
    link: "arm_link6",
    anchor: [0.17, 0, 0],
    region: [0.042, 0.032, 0.032],
    fullOpening: A1Z_FINGER_STROKE * 2,
  },
  // A top-down grasp reaches out to 0.47 m at cube height before the ±75° wrist pitch runs out.
  reach: { min: 0.27, max: 0.43, yaw: THREE.MathUtils.degToRad(55) },
  props: { cube: 0.045 },
  solve: solveA1Z,
  joints: [
    { name: "arm_joint1", label: "J1 base yaw", min: -120, max: 120, initial: 0, unit: "deg", urdfJoints: ["arm_joint1"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint2", label: "J2 shoulder", min: 0, max: 180, initial: 100, unit: "deg", urdfJoints: ["arm_joint2"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint3", label: "J3 elbow", min: -180, max: 0, initial: -100, unit: "deg", urdfJoints: ["arm_joint3"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint4", label: "J4 wrist pitch", min: -75, max: 75, initial: 55, unit: "deg", urdfJoints: ["arm_joint4"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint5", label: "J5 wrist yaw", min: -85, max: 85, initial: 0, unit: "deg", urdfJoints: ["arm_joint5"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint6", label: "J6 wrist roll", min: -115, max: 115, initial: 0, unit: "deg", urdfJoints: ["arm_joint6"], toURDF: revolute, fromURDF: measured },
    {
      name: "gripper",
      label: "Gripper",
      min: 0,
      max: 100,
      initial: 100,
      unit: "percent",
      urdfJoints: ["gripper_finger_left_joint", "gripper_finger_rIght_joint"],
      toURDF: (value) => (value / 100) * A1Z_FINGER_STROKE,
      fromURDF: (value) => (value / A1Z_FINGER_STROKE) * 100,
    },
  ],
};

/** The original SO-101 collector, kept reachable with ?robot=so101. */
const SO101: RobotSpec = {
  id: "so101",
  label: "SO-101",
  urdf: "/assets/robots/so101/so101_new_calib.urdf",
  packages: {},
  mount: [-0.22, 0.001, 0],
  view: {
    camera: [0.82, 0.58, 0.84],
    target: [0, 0.18, 0],
    distance: [0.7, 3],
    obsCamera: [0.68, 0.5, 0.68],
    obsTarget: [0, 0.16, 0],
  },
  finish: {}, // the SO-101 URDF already declares per-link colours
  // The SO-101 jaw swings on a hinge, so its opening is not linear in the control and the closed
  // position is measured rather than derived.
  grasp: {
    link: "gripper_frame_link",
    anchor: [0, 0, 0],
    region: [0.03, 0.03, 0.03],
    closedAt: 20,
  },
  reach: { min: 0.13, max: 0.26, yaw: THREE.MathUtils.degToRad(55) },
  props: { cube: 0.025 },
  joints: [
    { name: "shoulder_pan", label: "Shoulder pan", min: -110, max: 110, initial: 0, unit: "deg", urdfJoints: ["shoulder_pan"], toURDF: revolute, fromURDF: measured },
    { name: "shoulder_lift", label: "Shoulder lift", min: -100, max: 100, initial: -28, unit: "deg", urdfJoints: ["shoulder_lift"], toURDF: revolute, fromURDF: measured },
    { name: "elbow_flex", label: "Elbow flex", min: -97, max: 97, initial: 62, unit: "deg", urdfJoints: ["elbow_flex"], toURDF: revolute, fromURDF: measured },
    { name: "wrist_flex", label: "Wrist flex", min: -95, max: 95, initial: -32, unit: "deg", urdfJoints: ["wrist_flex"], toURDF: revolute, fromURDF: measured },
    { name: "wrist_roll", label: "Wrist roll", min: -157, max: 163, initial: 0, unit: "deg", urdfJoints: ["wrist_roll"], toURDF: revolute, fromURDF: measured },
    {
      name: "gripper",
      label: "Gripper",
      min: 0,
      max: 100,
      initial: 70,
      unit: "percent",
      urdfJoints: ["gripper"],
      toURDF: (value) => THREE.MathUtils.degToRad(THREE.MathUtils.lerp(-10, 100, value / 100)),
      fromURDF: (value) => THREE.MathUtils.inverseLerp(-10, 100, THREE.MathUtils.radToDeg(value)) * 100,
    },
  ],
};

const ROBOTS = [A1Z, SO101];
const requestedRobot = new URLSearchParams(window.location.search).get("robot");
const spec = ROBOTS.find((entry) => entry.id === requestedRobot) ?? A1Z;
const JOINTS = spec.joints;

/**
 * Where the jaws sit through a grasp. `GRIPPER_SHUT` is the control value at which the jaws just
 * meet the cube, minus a hair so the grip reads as firm rather than floating; closing past it
 * drives the fingers through the cube. Grab and release thresholds bracket it with hysteresis.
 */
const GRIPPER_OPEN = 100;
const GRIPPER_SHUT = spec.grasp.fullOpening
  ? THREE.MathUtils.clamp((spec.props.cube / spec.grasp.fullOpening) * 100 - 2, 5, 95)
  : (spec.grasp.closedAt ?? 45);
const GRIPPER_GRAB_BELOW = GRIPPER_SHUT + 5;
const GRIPPER_RELEASE_ABOVE = GRIPPER_SHUT + 15;

const CAPTURE_HZ = 5;
const SAMPLE_INTERVAL = 1000 / CAPTURE_HZ;

/**
 * The desk model is Y-up and stands on the floor with its work surface at y = 0.7875, so it is
 * dropped by exactly that to put the surface on our y = 0 work plane. Width and depth come from
 * the glTF bounding box; the collider is built from the same numbers so the visible top and the
 * one the cubes rest on cannot drift apart.
 */
const DESK = { top: 0.7875, width: 2.0, depth: 0.9472 };

/** Swapped per episode so the policy cannot use the backdrop as a position cue. */
const BACKDROPS = [
  "small_empty_room_1",
  "small_empty_room_2",
  "small_empty_room_3",
  "small_empty_room_4",
  "pine_attic",
  "reading_room",
  "wooden_lounge",
  "en_suite",
  "cabin",
  "comfy_cafe",
];
const PHYSICS_STEP = 1 / 480;
// Membership and filter bits, so a cube can stop colliding with the arm while it is being held.
const GROUP_ARM = 0x0001;
const GROUP_PROP = 0x0002;
const GROUP_WORLD = 0x0004;
const collisionGroups = (member: number, collidesWith: number) => (member << 16) | collidesWith;
const PROP_FREE_GROUPS = collisionGroups(GROUP_PROP, GROUP_ARM | GROUP_PROP | GROUP_WORLD);
const PROP_HELD_GROUPS = collisionGroups(GROUP_PROP, GROUP_PROP | GROUP_WORLD);

/** What the policy sees. The converter reads these back out of the dump's meta.json. */
const OBSERVATION_WIDTH = 256;
const OBSERVATION_HEIGHT = 192;

type PropId = "red" | "green" | "blue";
const PROP_SPECS: Array<{ id: PropId; label: string; color: number }> = [
  { id: "red", label: "red", color: 0xd0472c },
  { id: "green", label: "green", color: 0x4a9257 },
  { id: "blue", label: "blue", color: 0x3a6ba5 },
];

type Prop = {
  id: PropId;
  label: string;
  color: number;
  mesh: THREE.Mesh;
  body: RAPIER.RigidBody;
  collider: RAPIER.Collider;
};
/** Bottom, middle and top of the stack the instruction asks for. */
type Task = { order: [PropId, PropId, PropId]; instruction: string };

const sceneElement = document.querySelector<HTMLDivElement>("#scene")!;
const controlsElement = document.querySelector<HTMLDivElement>("#joint-controls")!;
const instructionElement = document.querySelector<HTMLTextAreaElement>("#instruction")!;
const recordButton = document.querySelector<HTMLButtonElement>("#record-button")!;
const stopButton = document.querySelector<HTMLButtonElement>("#stop-button")!;
const replayButton = document.querySelector<HTMLButtonElement>("#replay-button")!;
const downloadButton = document.querySelector<HTMLButtonElement>("#download-button")!;
const resetButton = document.querySelector<HTMLButtonElement>("#reset-button")!;
const recordingBadge = document.querySelector<HTMLDivElement>("#recording-badge")!;
const frameCountElement = document.querySelector<HTMLElement>("#frame-count")!;
const durationElement = document.querySelector<HTMLElement>("#duration")!;
const statusElement = document.querySelector<HTMLElement>("#status-message")!;
const previewElement = document.querySelector<HTMLImageElement>("#observation-preview")!;
const robotBadgeElement = document.querySelector<HTMLElement>("#robot-badge")!;
const newTaskButton = document.querySelector<HTMLButtonElement>("#new-task-button")!;
const taskStateElement = document.querySelector<HTMLElement>("#task-state")!;
const generateButton = document.querySelector<HTMLButtonElement>("#generate-button")!;
const datasetButton = document.querySelector<HTMLButtonElement>("#dataset-button")!;
const episodeCountElement = document.querySelector<HTMLInputElement>("#episode-count")!;

robotBadgeElement.textContent = `${spec.label} / SIM`;

const initialValues = Object.fromEntries(JOINTS.map((joint) => [joint.name, joint.initial])) as JointValues;
/** What the arm is told to do. With dynamics on, what it actually does is `measuredValues`. */
const currentValues = { ...initialValues };
const targetValues = { ...initialValues };
const measuredValues = { ...initialValues };
const sliders = new Map<string, HTMLInputElement>();
const valueLabels = new Map<string, HTMLElement>();

let task: Task = { order: ["red", "green", "blue"], instruction: "" };
let frames: EpisodeFrame[] = [];
let dataset: Array<ReturnType<typeof buildEpisode>> = [];
let script: {
  plan: Script;
  time: number;
  tick: number;
  /** Which leg of the stack is running, and how much episode time the earlier legs used. */
  stage: number;
  clock: number;
} | null = null;
let generation: {
  requested: number;
  completed: number;
  succeeded: number;
  /** Set while an episode is being flushed to disk; ticking resumes when it settles. */
  flushing: boolean;
} | null = null;
let recording = false;
let recordStart = 0;
let lastSample = 0;
let replaying = false;
let replayStart = 0;
let replayIndex = 0;

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
camera.position.set(...spec.view.camera);
camera.lookAt(...spec.view.target);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
sceneElement.appendChild(renderer.domElement);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.target.set(...spec.view.target);
orbitControls.enableDamping = true;
orbitControls.minDistance = spec.view.distance[0];
orbitControls.maxDistance = spec.view.distance[1];
orbitControls.maxPolarAngle = Math.PI * 0.49;

const observationCamera = new THREE.PerspectiveCamera(46, OBSERVATION_WIDTH / OBSERVATION_HEIGHT, 0.01, 10);
observationCamera.position.set(...spec.view.obsCamera);
observationCamera.lookAt(...spec.view.obsTarget);

const observationRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
observationRenderer.setSize(OBSERVATION_WIDTH, OBSERVATION_HEIGHT, false);
observationRenderer.outputColorSpace = THREE.SRGBColorSpace;
observationRenderer.toneMapping = THREE.ACESFilmicToneMapping;
observationRenderer.toneMappingExposure = 1.05;

// The environment map now carries the ambient, so the fill light is only a gentle lift and the
// key light is kept mainly for a crisp contact shadow.
const hemiLight = new THREE.HemisphereLight(0xfff8e7, 0x5d6259, 0.35);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xfff2d8, 1.9);
keyLight.position.set(1.2, 2, 0.8);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
keyLight.shadow.camera.left = -1.9;
keyLight.shadow.camera.right = 1.9;
keyLight.shadow.camera.top = 1.9;
keyLight.shadow.camera.bottom = -1.9;
scene.add(keyLight);

/**
 * Everything below is awaited before anything renders, for the same reason the URDF meshes are:
 * the first observation frame is captured immediately, and a missing desk or backdrop in it would
 * be a silent hole in the dataset.
 */
const gltf = await new GLTFLoader().loadAsync("/assets/models/desk/metal_office_desk.gltf");
const desk = gltf.scene;
desk.position.y = -DESK.top;
desk.traverse((object) => {
  if (object instanceof THREE.Mesh) {
    object.castShadow = true;
    object.receiveShadow = true;
  }
});
scene.add(desk);

// Catches the desk's shadow without painting over the backdrop behind it.
const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.32 }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -DESK.top;
floor.receiveShadow = true;
scene.add(floor);

/**
 * One equirectangular panorama per backdrop, used as both the visible background and the light
 * source. Loading all of them up front costs a slower start but keeps `newEpisode` synchronous,
 * which matters because generation runs on a fixed simulated clock and cannot await anything.
 */
const pmrem = new THREE.PMREMGenerator(renderer);
pmrem.compileEquirectangularShader();
const rgbeLoader = new RGBELoader();

const backdrops = await Promise.all(
  BACKDROPS.map(async (name) => {
    const texture = await rgbeLoader.loadAsync(`/assets/hdri/${name}.hdr`);
    texture.mapping = THREE.EquirectangularReflectionMapping;
    return { name, background: texture, environment: pmrem.fromEquirectangular(texture).texture };
  }),
);

function useBackdrop(index: number) {
  const backdrop = backdrops[index];
  scene.background = backdrop.background;
  scene.environment = backdrop.environment;
}

// Physics runs for the props only. The arm stays kinematic — it is driven straight from the
// joint controls and has no colliders — which keeps the recorded actions clean and dodges the
// soft, drooping serial chain that an impulse solver gives you for a force-driven arm. Grasping
// is a kinematic attach for the same reason: it never jitters and the pickup moment is legible.
await RAPIER.init();
const physics = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
physics.timestep = PHYSICS_STEP;
physics.numSolverIterations = 8;

// A slab whose top face sits exactly on y = 0. A box is the right collider for a flat desktop —
// the drawers and legs below it are never touched, so they need no geometry.
physics.createCollider(
  RAPIER.ColliderDesc.cuboid(DESK.width / 2, 0.05, DESK.depth / 2)
    .setTranslation(0, -0.05, 0)
    .setFriction(0.9),
);
// Floor, so a cube knocked off the desk lands somewhere instead of falling forever.
physics.createCollider(
  RAPIER.ColliderDesc.cuboid(6, 0.05, 6).setTranslation(0, -DESK.top - 0.05, 0).setFriction(0.9),
);

const cubeGeometry = new THREE.BoxGeometry(spec.props.cube, spec.props.cube, spec.props.cube);

const props: Prop[] = PROP_SPECS.map((propSpec) => {
  const mesh = new THREE.Mesh(
    cubeGeometry,
    new THREE.MeshStandardMaterial({ color: propSpec.color, roughness: 0.7 }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  const body = physics.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic().setLinearDamping(0.35).setAngularDamping(0.55).setCcdEnabled(true),
  );
  const collider = physics.createCollider(
    RAPIER.ColliderDesc.cuboid(spec.props.cube / 2, spec.props.cube / 2, spec.props.cube / 2)
      .setFriction(1.1)
      .setRestitution(0.05)
      .setDensity(700)
      .setCollisionGroups(PROP_FREE_GROUPS),
    body,
  );

  return { ...propSpec, mesh, body, collider };
});

// urdf-loader resolves loadAsync as soon as the XML is parsed, while the STL meshes are
// still in flight on the loading manager. Wait for the manager to drain, or the robot is
// still an empty skeleton when we recolour it, enable its shadows, and shoot the first
// observation frame.
const loadingManager = new THREE.LoadingManager();
const meshesLoaded = new Promise<void>((resolve) => {
  loadingManager.onLoad = resolve;
});
loadingManager.onError = (url) => {
  statusElement.textContent = `FAILED TO LOAD ${url}`;
};

const urdfLoader = new URDFLoader(loadingManager);
urdfLoader.parseCollision = false;
urdfLoader.packages = spec.packages;
statusElement.textContent = `LOADING ${spec.label} URDF`;
const robot = await urdfLoader.loadAsync(spec.urdf);
await meshesLoaded;
robot.rotation.x = -Math.PI / 2;
robot.position.set(...spec.mount);
/**
 * urdf-loader builds every visual as a MeshPhongMaterial, which reads flat and washed out
 * under this scene's ACES tone mapping. Rebuild each one as a standard material, taking the
 * colour from the robot spec where it overrides a link and from the URDF otherwise.
 */
function applyFinish(robotModel: URDFRobot, robotSpec: RobotSpec) {
  for (const [linkName, link] of Object.entries(robotModel.links)) {
    const override = robotSpec.finish[linkName];
    link.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      const source = object.material as THREE.MeshPhongMaterial;
      object.material = new THREE.MeshStandardMaterial({
        color: override ? new THREE.Color(override.color) : source.color.clone(),
        roughness: override?.roughness ?? 0.55,
        metalness: override?.metalness ?? 0.1,
        map: source.map ?? null,
      });
      source.dispose();
    });
  }
}

applyFinish(robot, spec);
robot.traverse((object) => {
  if (object instanceof THREE.Mesh) {
    object.castShadow = true;
    object.receiveShadow = true;
  }
});
scene.add(robot);

// An empty parented to the gripper link: reading its world transform gives us the jaw pose
// for free, including the robot root's Z-up-to-Y-up rotation.
const jawAnchor = new THREE.Object3D();
jawAnchor.position.set(...spec.grasp.anchor);
robot.links[spec.grasp.link].add(jawAnchor);

let heldProp: Prop | null = null;
let physicsAccumulator = 0;
const jawPosition = new THREE.Vector3();
const jawRotation = new THREE.Quaternion();
const previousJawPosition = new THREE.Vector3();
const jawInverse = new THREE.Quaternion();
const heldRotation = new THREE.Quaternion();
const scratch = new THREE.Vector3();

function setPropDynamic(prop: Prop) {
  prop.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
}

/** Snaps the prop into the jaw and hands its pose over to the gripper. */
function grabProp(prop: Prop) {
  heldProp = prop;
  heldRotation.copy(jawInverse).multiply(prop.mesh.quaternion);
  prop.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
  // The grasp snaps the cube into the jaw, which now has colliders of its own. Without this the
  // cube and the fingers would be interpenetrating and shove each other apart.
  prop.collider.setCollisionGroups(PROP_HELD_GROUPS);
  statusElement.textContent = `GRIPPED ${prop.label.toUpperCase()} CUBE`;
}

/** Hands the prop back to the physics world, carrying the jaw's velocity with it. */
function releaseProp(deltaSeconds: number) {
  const prop = heldProp;
  if (!prop) return;
  heldProp = null;
  setPropDynamic(prop);
  prop.collider.setCollisionGroups(PROP_FREE_GROUPS);
  scratch.copy(jawPosition).sub(previousJawPosition).divideScalar(Math.max(deltaSeconds, PHYSICS_STEP));
  prop.body.setLinvel(scratch, true);
  prop.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  statusElement.textContent = `RELEASED ${prop.label.toUpperCase()} CUBE`;
}

function findPropInJaw() {
  const [rx, ry, rz] = spec.grasp.region;
  return props.find((prop) => {
    scratch.copy(prop.mesh.position).sub(jawPosition).applyQuaternion(jawInverse);
    return Math.abs(scratch.x) <= rx && Math.abs(scratch.y) <= ry && Math.abs(scratch.z) <= rz;
  });
}

function updateGrasp(deltaSeconds: number) {
  jawAnchor.updateWorldMatrix(true, false);
  jawPosition.setFromMatrixPosition(jawAnchor.matrixWorld);
  jawAnchor.getWorldQuaternion(jawRotation);
  jawInverse.copy(jawRotation).invert();

  const opening = currentValues.gripper;
  if (heldProp) {
    if (opening > GRIPPER_RELEASE_ABOVE) {
      releaseProp(deltaSeconds);
    } else {
      heldProp.body.setNextKinematicTranslation(jawPosition);
      heldProp.body.setNextKinematicRotation(jawRotation.clone().multiply(heldRotation));
    }
  } else if (opening < GRIPPER_GRAB_BELOW) {
    const candidate = findPropInJaw();
    if (candidate) grabProp(candidate);
  }

  previousJawPosition.copy(jawPosition);
}

function syncPropMeshes() {
  for (const prop of props) {
    const { x, y, z } = prop.body.translation();
    const rotation = prop.body.rotation();
    prop.mesh.position.set(x, y, z);
    prop.mesh.quaternion.set(rotation.x, rotation.y, rotation.z, rotation.w);
  }
}

function stepPhysics(deltaSeconds: number) {
  physicsAccumulator = Math.min(physicsAccumulator + deltaSeconds, 0.1);
  while (physicsAccumulator >= PHYSICS_STEP) {
    physics.step();
    physicsAccumulator -= PHYSICS_STEP;
  }
  syncPropMeshes();
}

/**
 * Arm dynamics.
 *
 * Until now the arm was kinematic: joint angles were written straight into the URDF tree and the
 * links had no colliders, so the arm tracked commands perfectly and passed through everything it
 * was not holding. Here each link becomes a rigid body with the URDF's own mass, joined by
 * motorised joints, so the arm is driven by torques and has to fight gravity and its own inertia.
 *
 * These are *impulse* joints, not Rapier's reduced-coordinate multibody joints, because the JS
 * bindings expose motors only on the former. That is the known cost: a maximal-coordinate chain
 * is held together by constraints and can sag under load. `AccelerationBased` motors (whose gains
 * do not scale with the inertia they are pushing) and a raised solver iteration count are what
 * keep it tight.
 */
/** The link bolted to the desk. Both URDFs here name it the same thing. */
const ROOT_LINK = "base_link";
/**
 * Tuned by sweeping against the settled tracking error at the home pose. The interesting part is
 * that stiffness alone does almost nothing: at a 1/120 step the shoulder sat 26 degrees low no
 * matter the gain, because a stiff motor needs a step small enough to integrate it. At 1/480 the
 * same gain tracks to about a degree. That is why PHYSICS_STEP is 1/480 and not 1/120.
 */
let motorStiffness = 5_000_000;
let motorDamping = 2 * Math.sqrt(5_000_000);
/** Generous compared to the real arm's 3.5-25 Nm limits: the sim has no gravity-compensation
 *  controller, so the motors carry the whole load themselves. */
const MOTOR_MAX_FORCE = 1_000_000;
/** Cap on hull points per link: the STLs run to tens of thousands of vertices and the hull of a
 *  dense mesh is unchanged by sampling it. */
const HULL_POINT_BUDGET = 4000;


type ArmJoint = {
  joint: RAPIER.RevoluteImpulseJoint | RAPIER.PrismaticImpulseJoint;
  parent: RAPIER.RigidBody;
  child: RAPIER.RigidBody;
  axis: THREE.Vector3;
  anchor: THREE.Vector3;
  prismatic: boolean;
};

/**
 * Link-local vertices of what this link alone draws, thinned to the point budget.
 *
 * Note the hand-rolled walk: the URDF tree nests each link under the joint that drives it, so a
 * plain `traverse` would sweep up every link downstream and hand back a hull enclosing the whole
 * arm from here on. Descending only until the next joint is what keeps one hull to one link.
 */
function linkHullPoints(link: THREE.Object3D) {
  const toLocal = new THREE.Matrix4().copy(link.matrixWorld).invert();
  const meshes: THREE.Mesh[] = [];
  const collect = (object: THREE.Object3D) => {
    for (const child of object.children) {
      if ((child as { isURDFJoint?: boolean }).isURDFJoint) continue;
      if (child instanceof THREE.Mesh) meshes.push(child);
      collect(child);
    }
  };
  collect(link);
  const total = meshes.reduce((sum, mesh) => sum + mesh.geometry.attributes.position.count, 0);
  const stride = Math.max(1, Math.ceil(total / HULL_POINT_BUDGET));

  const points: number[] = [];
  const vertex = new THREE.Vector3();
  const toLink = new THREE.Matrix4();
  for (const mesh of meshes) {
    toLink.multiplyMatrices(toLocal, mesh.matrixWorld);
    const attribute = mesh.geometry.attributes.position;
    for (let index = 0; index < attribute.count; index += stride) {
      vertex.fromBufferAttribute(attribute, index).applyMatrix4(toLink);
      points.push(vertex.x, vertex.y, vertex.z);
    }
  }
  return new Float32Array(points);
}

/**
 * Builds the articulation, or returns null when this URDF is not one we can do faithfully.
 * A rotated joint origin (`rpy`) means the parent and child frames disagree about where the axis
 * points, and getting that wrong silently produces an arm that bends the wrong way — so rather
 * than half-support it, those robots keep the kinematic path.
 */
function buildArmDynamics() {
  robot.updateMatrixWorld(true);

  const rotated = Object.values(robot.joints).filter(
    (joint) => joint.jointType !== "fixed" && joint.quaternion.angleTo(IDENTITY) > 1e-6,
  );
  if (rotated.length > 0) {
    statusElement.textContent =
      `${spec.label} HAS ROTATED JOINT ORIGINS — RUNNING KINEMATIC, NOT DYNAMIC`;
    return null;
  }

  const bodies = new Map<string, RAPIER.RigidBody>();
  const translation = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scratchScale = new THREE.Vector3();

  for (const [name, link] of Object.entries(robot.links)) {
    link.matrixWorld.decompose(translation, rotation, scratchScale);
    // The base is bolted to the desk; everything above it is free to be pushed around.
    const isRoot = link === robot.links[ROOT_LINK];
    const body = physics.createRigidBody(
      (isRoot ? RAPIER.RigidBodyDesc.fixed() : RAPIER.RigidBodyDesc.dynamic())
        .setTranslation(translation.x, translation.y, translation.z)
        .setRotation(rotation)
        .setLinearDamping(0.4)
        .setAngularDamping(0.8)
        .setCanSleep(false),
    );

    const points = linkHullPoints(link);
    const desc = points.length >= 12 ? RAPIER.ColliderDesc.convexHull(points) : null;
    if (desc) {
      const collider = physics.createCollider(
        desc.setFriction(0.9).setRestitution(0).setCollisionGroups(collisionGroups(GROUP_ARM, GROUP_PROP | GROUP_WORLD)),
        body,
      );
      // Prefer the URDF's own mass over whatever a density would imply for a hull.
      const mass = link.inertial?.mass;
      if (mass && mass > 0) collider.setMass(mass);
    }
    bodies.set(name, body);
  }

  const joints = new Map<string, ArmJoint>();
  for (const [name, urdfJoint] of Object.entries(robot.joints)) {
    const prismatic = urdfJoint.jointType === "prismatic";
    if (!prismatic && urdfJoint.jointType !== "revolute" && urdfJoint.jointType !== "continuous") continue;

    const childLink = urdfJoint.children.find((child) => (child as URDFLink).isURDFLink) as URDFLink | undefined;
    const parentLink = urdfJoint.parent as URDFLink | null;
    if (!childLink || !parentLink) continue;
    const parent = bodies.get(parentLink.urdfName);
    const child = bodies.get(childLink.urdfName);
    if (!parent || !child) continue;

    const axis = urdfJoint.axis.clone().normalize();
    // urdf-loader nests parent link -> joint -> child link, so the joint's own local position is
    // the anchor in the parent's frame and the child sits at the joint's origin.
    const anchor = urdfJoint.position.clone();
    const data = prismatic
      ? RAPIER.JointData.prismatic(anchor, childLink.position, axis)
      : RAPIER.JointData.revolute(anchor, childLink.position, axis);
    data.limitsEnabled = true;
    data.limits = [urdfJoint.limit.lower, urdfJoint.limit.upper];

    const joint = physics.createImpulseJoint(data, parent, child, true) as
      | RAPIER.RevoluteImpulseJoint
      | RAPIER.PrismaticImpulseJoint;
    joint.configureMotorModel(RAPIER.MotorModel.AccelerationBased);
    joint.setMotorMaxForce(MOTOR_MAX_FORCE);
    joints.set(name, { joint, parent, child, axis, anchor, prismatic });
  }

  physics.numSolverIterations = 16;
  return joints;
}

const IDENTITY = new THREE.Quaternion();
const armJoints = buildArmDynamics();

const parentRotation = new THREE.Quaternion();
const childRotation = new THREE.Quaternion();
const relativeRotation = new THREE.Quaternion();
const relativeOffset = new THREE.Vector3();

/** Reads a joint's actual position back out of the solver, in URDF units. */
function measureJoint(entry: ArmJoint) {
  const parentAt = entry.parent.rotation();
  const childAt = entry.child.rotation();
  parentRotation.set(parentAt.x, parentAt.y, parentAt.z, parentAt.w);
  childRotation.set(childAt.x, childAt.y, childAt.z, childAt.w);

  if (entry.prismatic) {
    const from = entry.parent.translation();
    const to = entry.child.translation();
    relativeOffset
      .set(to.x - from.x, to.y - from.y, to.z - from.z)
      .applyQuaternion(parentRotation.clone().invert())
      .sub(entry.anchor);
    return relativeOffset.dot(entry.axis);
  }

  relativeRotation.copy(parentRotation).invert().multiply(childRotation);
  const along =
    relativeRotation.x * entry.axis.x +
    relativeRotation.y * entry.axis.y +
    relativeRotation.z * entry.axis.z;
  return 2 * Math.atan2(along, relativeRotation.w);
}

/** Points the motors at a commanded pose. What the arm actually does is up to the solver. */
function commandArm(values: JointValues) {
  if (!armJoints) return;
  for (const control of JOINTS) {
    const target = control.toURDF(values[control.name]);
    for (const name of control.urdfJoints) {
      armJoints.get(name)?.joint.configureMotorPosition(target, motorStiffness, motorDamping);
    }
  }
}

/** Copies the solved arm pose back into the URDF tree, and into the state we record. */
function syncArmFromPhysics() {
  if (!armJoints) return;
  for (const [name, entry] of armJoints) {
    robot.setJointValue(name, measureJoint(entry));
  }
  for (const control of JOINTS) {
    const entry = armJoints.get(control.urdfJoints[0]);
    if (entry) measuredValues[control.name] = control.fromURDF(measureJoint(entry));
  }
}

function setURDFJoint(robotModel: URDFRobot, joint: JointSpec, value: number) {
  const urdfValue = joint.toURDF(value);
  for (const urdfJoint of joint.urdfJoints) {
    robotModel.setJointValue(urdfJoint, urdfValue);
  }
}

function buildJointControls() {
  for (const joint of JOINTS) {
    const row = document.createElement("div");
    row.className = "joint-row";
    const label = document.createElement("label");
    label.htmlFor = `joint-${joint.name}`;
    label.textContent = joint.label;
    const input = document.createElement("input");
    input.id = `joint-${joint.name}`;
    input.type = "range";
    input.min = String(joint.min);
    input.max = String(joint.max);
    input.step = "1";
    input.value = String(joint.initial);
    const value = document.createElement("span");
    value.className = "joint-value";
    value.textContent = formatJointValue(joint.name, joint.initial);
    input.addEventListener("input", () => {
      targetValues[joint.name] = Number(input.value);
      value.textContent = formatJointValue(joint.name, targetValues[joint.name]);
    });
    row.append(label, input, value);
    controlsElement.append(row);
    sliders.set(joint.name, input);
    valueLabels.set(joint.name, value);
  }
}

function formatJointValue(name: string, value: number) {
  const joint = JOINTS.find((entry) => entry.name === name)!;
  return joint.unit === "percent" ? `${Math.round(value)}%` : `${Math.round(value)}°`;
}

/**
 * Drives the arm to a commanded pose. With dynamics the motors are aimed and the solver decides
 * what actually happens; without it the angles are written straight into the URDF tree.
 */
function setRobotPose(values: JointValues) {
  if (armJoints) {
    commandArm(values);
    return;
  }
  for (const joint of JOINTS) {
    setURDFJoint(robot, joint, values[joint.name]);
    measuredValues[joint.name] = values[joint.name];
  }
}

function updateJointUI(values: JointValues) {
  for (const joint of JOINTS) {
    sliders.get(joint.name)!.value = String(values[joint.name]);
    valueLabels.get(joint.name)!.textContent = formatJointValue(joint.name, values[joint.name]);
  }
}

function cloneJointValues(values: JointValues): JointValues {
  return { ...values };
}

function startRecording() {
  frames = [];
  recording = true;
  replaying = false;
  recordStart = performance.now();
  lastSample = recordStart - SAMPLE_INTERVAL;
  recordButton.classList.add("is-recording");
  recordButton.disabled = true;
  stopButton.disabled = false;
  replayButton.disabled = true;
  downloadButton.disabled = true;
  instructionElement.disabled = true;
  recordingBadge.hidden = false;
  statusElement.textContent = "RECORDING — move the joint controls";
  updateStats(0);
}

function stopRecording() {
  if (!recording) return;
  recording = false;
  recordButton.classList.remove("is-recording");
  recordButton.disabled = false;
  stopButton.disabled = true;
  replayButton.disabled = frames.length === 0;
  downloadButton.disabled = frames.length === 0;
  instructionElement.disabled = false;
  recordingBadge.hidden = true;
  statusElement.textContent = `EPISODE READY — ${frames.length} frames captured`;
}

/** Renders the observation camera and returns the frame as a JPEG data URL. */
function renderObservation() {
  observationRenderer.render(scene, observationCamera);
  return observationRenderer.domElement.toDataURL("image/jpeg", 0.72);
}

function captureFrame(seconds: number) {
  const image = renderObservation();
  frames.push({
    timestamp: Number(seconds.toFixed(3)),
    observation: {
      image,
      joint_positions: cloneJointValues(measuredValues),
      object_poses: props.map((prop) => ({
        id: prop.id,
        position: prop.mesh.position.toArray().map(round4) as [number, number, number],
        quaternion: prop.mesh.quaternion.toArray().map(round4) as [number, number, number, number],
      })),
      grasped: heldProp?.id ?? null,
    },
    action: {
      joint_targets: cloneJointValues(targetValues),
    },
  });
  previewElement.src = image;
  updateStats(seconds * 1000);
}

function updateStats(elapsedMs: number) {
  frameCountElement.textContent = String(frames.length).padStart(3, "0");
  durationElement.textContent = (elapsedMs / 1000).toFixed(1).padStart(4, "0");
}

function replayEpisode() {
  if (frames.length === 0) return;
  replaying = true;
  recording = false;
  replayStart = performance.now();
  replayIndex = 0;
  replayButton.disabled = true;
  recordButton.disabled = true;
  resetButton.disabled = true;
  statusElement.textContent = "REPLAYING CAPTURED EPISODE";
}

function updateReplay(now: number) {
  const elapsed = (now - replayStart) / 1000;
  while (replayIndex < frames.length && frames[replayIndex].timestamp <= elapsed) {
    Object.assign(currentValues, frames[replayIndex].observation.joint_positions);
    Object.assign(targetValues, currentValues);
    previewElement.src = frames[replayIndex].observation.image;
    updateJointUI(currentValues);
    replayIndex += 1;
  }
  if (replayIndex >= frames.length) {
    replaying = false;
    replayButton.disabled = false;
    recordButton.disabled = false;
    resetButton.disabled = false;
    statusElement.textContent = "REPLAY COMPLETE";
  }
}

function buildEpisode() {
  return {
    format: "3jsvla.episode.v2",
    robot: spec.id,
    instruction: instructionElement.value.trim(),
    task: { order: task.order },
    capture_hz: CAPTURE_HZ,
    success: taskSucceeded(),
    created_at: new Date().toISOString(),
    frames,
  };
}

function downloadJSON(payload: unknown, name: string) {
  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadEpisode() {
  downloadJSON(buildEpisode(), `${spec.id}_episode_${Date.now()}.json`);
  statusElement.textContent = "EPISODE DOWNLOADED";
}

function downloadDataset() {
  downloadJSON(
    {
      format: "3jsvla.dataset.v1",
      robot: spec.id,
      capture_hz: CAPTURE_HZ,
      created_at: new Date().toISOString(),
      episodes: dataset,
    },
    `${spec.id}_dataset_${dataset.length}ep_${Date.now()}.json`,
  );
  statusElement.textContent = `DATASET DOWNLOADED — ${dataset.length} EPISODES`;
}

/**
 * A random ordering of the three cubes into a stack. The instruction is the only thing that says
 * which cube goes where, and the order matters — build it wrong and the stack is wrong even
 * though every cube was moved. Six permutations share the same visual scene, so a policy that
 * ignores the language cannot do better than chance.
 */
function rollTask(): Task {
  const shuffled = [...PROP_SPECS];
  for (let i = shuffled.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const [base, middle, top] = shuffled;
  return {
    order: [base.id, middle.id, top.id],
    instruction:
      `Stack the ${middle.label} cube on the ${base.label} cube, ` +
      `then put the ${top.label} cube on top.`,
  };
}

/** A spot claimed on the table, with the radius it needs kept clear around it. */
type Placement = { at: THREE.Vector2; clearance: number };

/**
 * Samples a spot in the arm's reachable annulus that clears everything placed so far. Returns
 * null rather than falling back to a fixed spot, so a cramped layout is retried instead of
 * silently overlapping.
 */
function samplePlacement(taken: Placement[], clearance: number) {
  const { min, max, yaw } = spec.reach;
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const radius = THREE.MathUtils.lerp(min, max, Math.random());
    const angle = THREE.MathUtils.lerp(-yaw, yaw, Math.random());
    const at = new THREE.Vector2(
      spec.mount[0] + radius * Math.cos(angle),
      spec.mount[2] - radius * Math.sin(angle),
    );
    if (taken.every((other) => other.at.distanceTo(at) >= clearance + other.clearance)) {
      return { at, clearance };
    }
  }
  return null;
}

/**
 * One spot per cube, or null if this attempt boxed itself in. They are kept a good way apart:
 * the arm carries a cube kinematically, and a cube passing close over another one would shove
 * it, so the stack has room around it.
 */
function sampleLayout() {
  const clearance = spec.props.cube * 2.2;
  const taken: Placement[] = [];
  for (let index = 0; index < props.length; index += 1) {
    const placement = samplePlacement(taken, clearance);
    if (!placement) return null;
    taken.push(placement);
  }
  return taken;
}

function newEpisode() {
  if (heldProp) {
    const prop = heldProp;
    heldProp = null;
    setPropDynamic(prop);
  }

  let layout = sampleLayout();
  for (let retry = 0; !layout && retry < 20; retry += 1) layout = sampleLayout();
  if (!layout) {
    statusElement.textContent = "COULD NOT LAY OUT THE SCENE — WIDEN spec.reach";
    return;
  }

  const spots = layout.map((placement) => placement.at);
  props.forEach((prop, index) => {
    const spot = spots[index];
    prop.body.setTranslation({ x: spot.x, y: spec.props.cube / 2, z: spot.y }, true);
    prop.body.setRotation(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.random() * Math.PI * 2),
      true,
    );
    prop.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    prop.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  });

  syncPropMeshes();
  // A fixed backdrop is a shortcut: the same wall in the same place is a free position cue, and
  // a policy will happily use it instead of looking at the cubes. Re-rolling it per episode
  // forces the visual encoder onto the desk and the props.
  useBackdrop(Math.floor(Math.random() * backdrops.length));
  task = rollTask();
  instructionElement.value = task.instruction;
  frames = [];
  replayButton.disabled = true;
  downloadButton.disabled = true;
  updateStats(0);
  resetPose();
  // Refresh the preview so it shows the scene you are actually looking at. Skipped while
  // generating, where the episode's own first capture lands a frame a moment later anyway.
  if (!generation) previewElement.src = renderObservation();
  statusElement.textContent = "NEW TASK — adjust a joint, then start recording";
}

/**
 * Done when the three cubes are stacked in the order the instruction named, at rest and nothing
 * held. Physics decides this, not a region test: each cube has to actually be sitting at its
 * level and lined up over the base, so a knocked-over or badly aimed stack simply fails.
 */
function taskSucceeded() {
  if (heldProp) return false;
  const cube = spec.props.cube;
  const stack = task.order.map((id) => props.find((entry) => entry.id === id)!);
  const base = stack[0].mesh.position;

  return stack.every((prop, level) => {
    const at = prop.mesh.position;
    if (Math.abs(at.y - (level + 0.5) * cube) > cube * 0.35) return false;
    if (Math.hypot(at.x - base.x, at.z - base.z) > cube * 0.6) return false;
    const velocity = prop.body.linvel();
    return Math.hypot(velocity.x, velocity.y, velocity.z) <= 0.03;
  });
}

function updateTaskState() {
  const done = taskSucceeded();
  taskStateElement.textContent = done ? "Stacked" : "Not stacked";
  taskStateElement.classList.toggle("is-success", done);
}

/** One leg of the scripted trajectory: ease to `pose` over `duration` seconds. */
type ScriptStep = { pose: JointValues; duration: number };
type Script = { steps: ScriptStep[]; start: JointValues; total: number };

// Demonstration speed, not the arm's limit. At 5 Hz capture this puts roughly 12 degrees
// between consecutive actions; raise CAPTURE_HZ if a policy needs finer steps than that.
const ARM_SPEED = 28; // deg/s
const GRIPPER_SPEED = 90; // percent/s

function stepDuration(from: JointValues, to: JointValues) {
  let seconds = 0.2;
  for (const joint of JOINTS) {
    const delta = Math.abs(to[joint.name] - from[joint.name]);
    seconds = Math.max(seconds, delta / (joint.unit === "percent" ? GRIPPER_SPEED : ARM_SPEED));
  }
  return seconds;
}

/**
 * Waypoints for the current task: reach over the cube, drop onto it, close, lift, carry to the
 * zone, lower, let go. Every episode re-rolls the approach pitch, hover height, drop height, a
 * few millimetres of aim jitter and the angle the cube is set down at, so a batch is a spread of
 * demonstrations rather than one trajectory repeated. Returns null when the IK cannot reach.
 */
/** An episode is two of these: middle cube onto the base, then top cube onto the pair. */
const STACK_STAGES = 2;

/**
 * Waypoints for one leg: fetch `sourceId` and set it down centred on the base cube at `level`
 * (1 sits on the base, 2 sits on the pair). Positions are read live from the physics bodies, so
 * planning the second leg after the first has landed absorbs whatever the base drifted.
 *
 * Every leg re-rolls the approach pitch, hover height and a little aim error, so a batch is a
 * spread of demonstrations rather than one motion repeated. Placement jitter is deliberately far
 * tighter than the pick jitter: a few millimetres off and the stack topples, which is the whole
 * difference between this task and dropping a cube in a wide zone.
 */
function buildStage(sourceId: PropId, level: number): Script | null {
  if (!spec.solve) return null;
  const source = props.find((entry) => entry.id === sourceId)!;
  const base = props.find((entry) => entry.id === task.order[0])!;
  const cube = spec.props.cube;

  const jitter = (spread: number) => (Math.random() - 0.5) * spread;
  const sourceAt = source.body.translation();
  const baseAt = base.body.translation();
  const sourceYaw = new THREE.Euler().setFromQuaternion(source.mesh.quaternion, "YXZ").y;
  const baseYaw = new THREE.Euler().setFromQuaternion(base.mesh.quaternion, "YXZ").y;

  const grasp = new THREE.Vector3(sourceAt.x + jitter(0.008), cube / 2, sourceAt.z + jitter(0.008));
  const place = new THREE.Vector3(
    baseAt.x + jitter(0.005),
    (level + 0.5) * cube + THREE.MathUtils.lerp(0.003, 0.010, Math.random()),
    baseAt.z + jitter(0.005),
  );

  // Height costs pitch: straight down the jaw only clears 57 mm at the far edge of the workspace
  // but 199 mm at 70 degrees, and carrying over a growing stack needs real height. So prefer a
  // vertical grasp and give up pitch, then hover, only as far as the reach demands.
  const firstPitch = THREE.MathUtils.lerp(82, 90, Math.random());
  const firstHover = THREE.MathUtils.lerp(0.075, 0.12, Math.random());

  const plan = ((): Array<{ pose: JointValues; hold?: number }> | null => {
    for (let hover = firstHover; hover >= 0.05; hover -= 0.015) {
      // The carried cube is a kinematic body and will shove anything it clips, so the transit
      // height has to clear the stack that is already there, not just the cube being picked up.
      const transit = level * cube + hover;
      const overSource = grasp.clone().setY(Math.max(cube / 2 + hover, transit));
      const overPlace = place.clone().setY(transit + cube / 2);

      for (let pitchDeg = firstPitch; pitchDeg >= 58; pitchDeg -= 4) {
        const pitch = revolute(pitchDeg);
        const at = (point: THREE.Vector3, azimuth: number, gripper: number) => {
          const solution = spec.solve!(spec.mount, point, pitch, azimuth);
          return solution ? { ...solution, gripper } : null;
        };
        // hold: repeat a pose so the attach registers, and so a placed cube settles before the cut
        const attempt: Array<{ pose: JointValues | null; hold?: number }> = [
          { pose: at(overSource, sourceYaw, GRIPPER_OPEN) },
          { pose: at(grasp, sourceYaw, GRIPPER_OPEN) },
          { pose: at(grasp, sourceYaw, GRIPPER_SHUT) },
          { pose: at(grasp, sourceYaw, GRIPPER_SHUT), hold: 0.7 },
          { pose: at(overSource, sourceYaw, GRIPPER_SHUT) },
          { pose: at(overPlace, baseYaw, GRIPPER_SHUT) },
          { pose: at(place, baseYaw, GRIPPER_SHUT) },
          { pose: at(place, baseYaw, GRIPPER_SHUT), hold: 0.7 },
          { pose: at(place, baseYaw, GRIPPER_OPEN) },
          { pose: at(place, baseYaw, GRIPPER_OPEN), hold: 0.7 },
          { pose: at(overPlace, baseYaw, GRIPPER_OPEN) },
        ];
        if (attempt.every((entry) => entry.pose !== null)) {
          return attempt as Array<{ pose: JointValues; hold?: number }>;
        }
      }
    }
    return null;
  })();
  if (!plan) return null;

  const start = { ...currentValues };
  let previous = start;
  let total = 0;
  const steps = plan.map((entry) => {
    const pose = entry.pose;
    const duration = entry.hold ?? stepDuration(previous, pose);
    previous = pose;
    total += duration;
    return { pose, duration };
  });
  return { steps, start, total };
}

/** The commanded pose at a point in the script, eased between waypoints. */
function poseAt(script: Script, time: number): JointValues {
  let previous = script.start;
  let clock = 0;
  for (const step of script.steps) {
    if (time < clock + step.duration) {
      const alpha = THREE.MathUtils.smoothstep((time - clock) / step.duration, 0, 1);
      const pose: JointValues = {};
      for (const joint of JOINTS) {
        pose[joint.name] = THREE.MathUtils.lerp(previous[joint.name], step.pose[joint.name], alpha);
      }
      return pose;
    }
    clock += step.duration;
    previous = step.pose;
  }
  return { ...previous };
}

function resetPose() {
  Object.assign(currentValues, initialValues);
  Object.assign(targetValues, initialValues);
  updateJointUI(initialValues);
  setRobotPose(initialValues);
  statusElement.textContent = "POSE RESET";
}

/**
 * Streams episodes to a folder the user picks, one directory each, images as plain JPEG files.
 * The alternative — accumulating everything and handing over one JSON blob — holds the whole
 * dataset in memory and pays a third again in size for base64, which stops being reasonable at
 * a couple of hundred episodes. `tools/to_lerobot.py` turns this tree into a LeRobotDataset.
 */
declare global {
  // Not in TypeScript's DOM lib yet; everything else the writer touches already is.
  interface Window {
    showDirectoryPicker(options?: { mode?: "read" | "readwrite" }): Promise<FileSystemDirectoryHandle>;
  }
}

type DiskWriter = {
  root: FileSystemDirectoryHandle;
  episodes: FileSystemDirectoryHandle;
  written: number;
};

let writer: DiskWriter | null = null;

const canWriteToDisk = () => "showDirectoryPicker" in window;

/** JPEG data URL to bytes, so a frame can be written without a round trip through base64. */
function dataUrlToBlob(dataUrl: string) {
  const binary = atob(dataUrl.slice(dataUrl.indexOf(",") + 1));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return new Blob([bytes], { type: "image/jpeg" });
}

async function writeFile(directory: FileSystemDirectoryHandle, name: string, contents: Blob | string) {
  const handle = await directory.getFileHandle(name, { create: true });
  const stream = await handle.createWritable();
  await stream.write(contents);
  await stream.close();
}

/** Writes one episode as its own directory: the frames as JPEGs, everything else as JSON. */
async function writeEpisode(episode: ReturnType<typeof buildEpisode>, index: number) {
  const name = `episode_${String(index).padStart(5, "0")}`;
  const directory = await writer!.episodes.getDirectoryHandle(name, { create: true });
  const frameDirectory = await directory.getDirectoryHandle("frames", { create: true });

  const record = {
    ...episode,
    frames: await Promise.all(
      episode.frames.map(async (frame, frameIndex) => {
        const file = `${String(frameIndex).padStart(6, "0")}.jpg`;
        await writeFile(frameDirectory, file, dataUrlToBlob(frame.observation.image));
        const { image: _image, ...observation } = frame.observation;
        return { ...frame, observation: { ...observation, image_path: `frames/${file}` } };
      }),
    ),
  };
  await writeFile(directory, "episode.json", JSON.stringify(record));
  writer!.written += 1;
}

async function writeDatasetMeta(summary: { completed: number; succeeded: number }) {
  await writeFile(
    writer!.root,
    "meta.json",
    JSON.stringify(
      {
        format: "3jsvla.dump.v1",
        robot: spec.id,
        capture_hz: CAPTURE_HZ,
        image: { width: OBSERVATION_WIDTH, height: OBSERVATION_HEIGHT },
        joints: JOINTS.map((joint) => joint.name),
        episodes: summary.completed,
        successes: summary.succeeded,
        created_at: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
}

/**
 * Generated episodes run on a fixed simulated clock rather than wall time, so the dataset is the
 * same however fast the browser happens to be, and pausing the tab cannot stretch a trajectory.
 */
const GENERATION_DT = PHYSICS_STEP * 2;
const TICKS_PER_CAPTURE = Math.round(1 / (CAPTURE_HZ * GENERATION_DT));
/** Ticks are processed in slices, so the page keeps rendering and you can watch it collect. */
const GENERATION_BUDGET_MS = 8;

function startGeneratedEpisode() {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    newEpisode();
    const plan = buildStage(task.order[1], 1);
    // Check the second leg is reachable too before committing to the layout — the stack sits
    // where the base already is, so it can be planned now even though it is re-planned later.
    if (plan && buildStage(task.order[2], 2)) {
      script = { plan, time: 0, tick: 0, stage: 0, clock: 0 };
      return true;
    }
  }
  return false;
}

/** One deterministic tick. Returns false once the script has run out. */
function advanceScript() {
  if (!script) return false;
  const { plan } = script;

  Object.assign(currentValues, poseAt(plan, script.time));
  // The action is the command one capture interval ahead — what a policy would have to emit.
  Object.assign(targetValues, poseAt(plan, Math.min(script.time + 1 / CAPTURE_HZ, plan.total)));

  setRobotPose(currentValues);
  stepPhysics(GENERATION_DT);
  syncArmFromPhysics();
  updateGrasp(GENERATION_DT);

  if (script.tick % TICKS_PER_CAPTURE === 0) captureFrame(script.clock + script.time);
  script.tick += 1;
  script.time += GENERATION_DT;
  if (script.time < plan.total) return true;

  // Leg finished. Plan the next one from where the world actually ended up, so any drift in the
  // base cube while the first cube landed on it is taken into account rather than assumed away.
  const stage = script.stage + 1;
  if (stage >= STACK_STAGES) return false;
  const next = buildStage(task.order[stage + 1], stage + 1);
  if (!next) return false; // unreachable from here: end the episode, it will score as a failure
  script = { plan: next, time: 0, tick: script.tick, stage, clock: script.clock + plan.total };
  return true;
}

function setGenerationControls(running: boolean) {
  recordButton.disabled = running;
  newTaskButton.disabled = running;
  resetButton.disabled = running;
  generateButton.disabled = running;
  generateButton.textContent = running ? "Generating…" : "Generate";
}

function finishGeneration(problem?: string) {
  const summary = generation!;
  generation = null;
  script = null;
  setGenerationControls(false);
  datasetButton.disabled = dataset.length === 0;
  replayButton.disabled = frames.length === 0;
  downloadButton.disabled = frames.length === 0;
  const rate = summary.completed ? Math.round((summary.succeeded / summary.completed) * 100) : 0;
  const done = `${summary.completed} EPISODES — ${summary.succeeded} SUCCEEDED (${rate}%)`;
  const target = writer ? ` INTO ${writer.root.name}/` : "";
  const report = problem ? `STOPPED AFTER ${done} — ${problem}` : `GENERATED ${done}${target}`;

  if (writer) {
    const finished = writer;
    writeDatasetMeta(summary)
      .then(() => {
        statusElement.textContent = `${report} — RUN tools/to_lerobot.py ON ${finished.root.name}/`;
      })
      .catch((error) => {
        statusElement.textContent = `${report} — COULD NOT WRITE meta.json: ${error}`;
      });
  }
  statusElement.textContent = report;
}

async function startGeneration() {
  if (!spec.solve) {
    statusElement.textContent = `NO IK FOR ${spec.label} — SCRIPTED GENERATION UNAVAILABLE`;
    return;
  }
  const requested = Math.max(1, Math.min(2000, Number(episodeCountElement.value) || 1));
  episodeCountElement.value = String(requested);

  writer = null;
  if (canWriteToDisk()) {
    try {
      // Must be called straight off the click: the picker needs the user gesture.
      const root = await window.showDirectoryPicker({ mode: "readwrite" });
      writer = { root, episodes: await root.getDirectoryHandle("episodes", { create: true }), written: 0 };
    } catch {
      statusElement.textContent = "NO FOLDER CHOSEN — GENERATION CANCELLED";
      return;
    }
  }

  dataset = [];
  generation = { requested, completed: 0, succeeded: 0, flushing: false };
  setGenerationControls(true);
  datasetButton.disabled = true;
  statusElement.textContent = writer
    ? `GENERATING 0/${requested} — WRITING TO ${writer.root.name}/`
    : `GENERATING 0/${requested} — IN MEMORY (THIS BROWSER CANNOT WRITE TO A FOLDER)`;
}

function runGeneration() {
  const deadline = performance.now() + GENERATION_BUDGET_MS;
  while (generation && !generation.flushing && performance.now() < deadline) {
    if (!script) {
      if (generation.completed >= generation.requested) {
        finishGeneration();
        return;
      }
      if (!startGeneratedEpisode()) {
        finishGeneration("COULD NOT PLAN A REACHABLE EPISODE");
        return;
      }
    }
    if (!advanceScript()) {
      const episode = buildEpisode();
      const index = generation.completed;
      generation.completed += 1;
      if (episode.success) generation.succeeded += 1;
      script = null;
      statusElement.textContent =
        `GENERATING ${generation.completed}/${generation.requested} — ${generation.succeeded} SUCCEEDED`;

      if (writer) {
        // Hand it to disk and stop ticking until it lands, so memory stays flat.
        const pending = generation;
        pending.flushing = true;
        writeEpisode(episode, index)
          .then(() => {
            pending.flushing = false;
          })
          .catch((error) => {
            pending.flushing = false;
            finishGeneration(`WRITE FAILED: ${error}`);
          });
      } else {
        dataset.push(episode);
      }
    }
  }
}

function resizeRenderer() {
  const width = sceneElement.clientWidth;
  const height = sceneElement.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

let previousTime = performance.now();
function animate(now: number) {
  const deltaSeconds = Math.min((now - previousTime) / 1000, 0.1);
  previousTime = now;

  if (generation) {
    runGeneration();
    updateJointUI(currentValues);
  } else {
    if (replaying) {
      updateReplay(now);
    } else {
      // Sliders command the motors directly. The lag between command and pose used to be a
      // hand-rolled exponential blend; it now comes from the arm's own inertia.
      const blend = armJoints ? 1 : 1 - Math.exp(-9 * deltaSeconds);
      for (const joint of JOINTS) {
        currentValues[joint.name] = THREE.MathUtils.lerp(
          currentValues[joint.name],
          targetValues[joint.name],
          blend,
        );
      }
    }
    setRobotPose(currentValues);
    stepPhysics(deltaSeconds);
    syncArmFromPhysics();
    updateGrasp(deltaSeconds);
  }
  updateTaskState();
  orbitControls.update();
  renderer.render(scene, camera);

  if (recording && now - lastSample >= SAMPLE_INTERVAL) {
    captureFrame((now - recordStart) / 1000);
    lastSample = now;
  }
}

buildJointControls();
resizeRenderer();
useBackdrop(0);
newEpisode();
stepPhysics(PHYSICS_STEP);
previewElement.src = renderObservation();
renderer.setAnimationLoop(animate);

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
replayButton.addEventListener("click", replayEpisode);
downloadButton.addEventListener("click", downloadEpisode);
resetButton.addEventListener("click", resetPose);
newTaskButton.addEventListener("click", newEpisode);
generateButton.addEventListener("click", startGeneration);
datasetButton.addEventListener("click", downloadDataset);
window.addEventListener("resize", resizeRenderer);
