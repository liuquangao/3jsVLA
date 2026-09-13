import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import URDFLoader, { type URDFLink, type URDFRobot } from "urdf-loader";
import { ConstrainedIK } from "./constrained-ik";
import { bothFingersGrip, manifoldHasContact } from "./contact-state";
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
  /** World Y rotation applied to the robot and its reachable workspace. */
  heading?: number;
  view: {
    camera: Vec3;
    target: Vec3;
    distance: [number, number];
    obsCamera: Vec3;
    obsTarget: Vec3;
  };
  /** Keyed by URDF link name. Links absent here keep the colour the URDF declares. */
  finish: Record<string, LinkFinish>;
  /** How the gripper closes on an object. */
  grasp: {
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
  solve?: (mount: Vec3, target: THREE.Vector3, pitch: number, jawAzimuth: number, seed?: JointValues) => JointValues | null;
  joints: JointSpec[];
};

const revolute = (value: number) => THREE.MathUtils.degToRad(value);
const measured = (value: number) => THREE.MathUtils.radToDeg(value);

/**
 * Closed-form IK for the A1Z. J1 takes the azimuth and J2/J3/J4 all turn about Y, so once the
 * tool pitch is fixed what is left is a planar two-link problem with an exact solution. Working
 * in the (radial, height) plane as complex numbers, a turn of q about Y is a multiply by e^-iq.
 */
/** Jaw centre along x in the arm_link6 frame: the point the IK aims, between the closed tips. */
const A1Z_JAW_ANCHOR = 0.17;
const A1Z_LINKS = {
  shoulder: new THREE.Vector2(0.02, 0.118), // joint 2, from the URDF base
  upper: new THREE.Vector2(-0.264, 0), // joint 2 -> joint 3
  fore: new THREE.Vector2(0.245, 0.06), // joint 3 -> joint 4
  wrist: new THREE.Vector2(0.074 + 0.0235 + A1Z_JAW_ANCHOR, 0), // joint 4 -> jaw, with J5 at zero
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
  // Mounted along the long table edge, facing +Z into the desktop.
  // Base extends 37.5 mm towards the edge; leave its outer face 3 mm inside the desktop.
  mount: [0.20, 0.001, -0.4331],
  heading: -Math.PI / 2,
  // Framed low enough that the backdrop reads as a room rather than just its floor, while the
  // observation still fills with the reachable patch of desk.
  view: {
    camera: [1.15, 1.15, 1.55],
    target: [0.10, 0.12, 0],
    distance: [0.9, 5],
    obsCamera: [0.95, 0.72, 0.85],
    obsTarget: [0.10, 0.05, -0.05],
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
  // The G1Z jaws open to 60 mm at control 100.
  grasp: { fullOpening: A1Z_FINGER_STROKE * 2 },
  // A top-down grasp reaches out to 0.47 m at cube height before the ±75° wrist pitch runs out.
  reach: { min: 0.27, max: 0.43, yaw: THREE.MathUtils.degToRad(55) },
  props: { cube: 0.045 },
  solve: (mount, target, pitch, jawAzimuth, seed) => {
    const localTarget = target.clone().sub(new THREE.Vector3(...mount))
      .applyAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    const analyticSeed = solveA1Z([0, 0, 0], localTarget, pitch, jawAzimuth + Math.PI / 2);
    const preferred = seed ?? currentValues;
    const radialYaw = Math.atan2(-(target.z - mount[2]), target.x - mount[0]);
    const forward = new THREE.Vector3(
      Math.cos(radialYaw) * Math.cos(pitch), -Math.sin(pitch), -Math.sin(radialYaw) * Math.cos(pitch),
    ).normalize();
    // A square cube permits four equivalent jaw orientations. Try the one closest
    // to the seed first, instead of forcing a wrist flip at a symmetry boundary.
    const reference = constrainedIK!.forwardPose(preferred).rotation;
    const orientations = [0, 1, 2, 3].map((quarter) => {
      const yaw = jawAzimuth + quarter * Math.PI / 2;
      const opening = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
      opening.addScaledVector(forward, -opening.dot(forward)).normalize();
      const up = new THREE.Vector3().crossVectors(forward, opening).normalize();
      return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(forward, opening, up));
    }).sort((a, b) => a.angleTo(reference) - b.angleTo(reference));
    for (const orientation of orientations) {
      const solution = constrainedIK!.solve(target, orientation, analyticSeed ? [preferred, analyticSeed] : [preferred]);
      if (solution) return solution;
    }
    return null;
  },
  joints: [
    { name: "arm_joint1", label: "J1 base yaw", min: -120, max: 120, initial: 0, unit: "deg", urdfJoints: ["arm_joint1"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint2", label: "J2 shoulder", min: 0, max: 180, initial: 0, unit: "deg", urdfJoints: ["arm_joint2"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint3", label: "J3 elbow", min: -180, max: 0, initial: 0, unit: "deg", urdfJoints: ["arm_joint3"], toURDF: revolute, fromURDF: measured },
    { name: "arm_joint4", label: "J4 wrist pitch", min: -75, max: 75, initial: 0, unit: "deg", urdfJoints: ["arm_joint4"], toURDF: revolute, fromURDF: measured },
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
  grasp: { closedAt: 20 },
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
 * `GRIPPER_SHUT` deliberately commands the jaws *narrower* than the cube. The fingers cannot get
 * there — the cube is in the way — so the unreachable remainder of the command turns into contact
 * force, which is what holds the cube up. Command exactly the cube's width instead and the grip
 * is zero.
 */
const GRIPPER_SQUEEZE = 0.01;
const GRIPPER_OPEN = 100;
const GRIPPER_SHUT = spec.grasp.fullOpening
  ? THREE.MathUtils.clamp(((spec.props.cube - GRIPPER_SQUEEZE) / spec.grasp.fullOpening) * 100, 0, 95)
  : (spec.grasp.closedAt ?? 45);

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
  "abandoned_factory_canteen_01",
  "forest_slope",
  "industrial_sunset",
  "moonless_golf",
  "studio_small_09",
  "venice_sunset",
];
const PHYSICS_STEP = 1 / 480;
// Collision membership and filters for the arm, props and world.
const GROUP_ARM = 0x0001;
const GROUP_PROP = 0x0002;
const GROUP_WORLD = 0x0004;
const collisionGroups = (member: number, collidesWith: number) => (member << 16) | collidesWith;
const PROP_FREE_GROUPS = collisionGroups(GROUP_PROP, GROUP_ARM | GROUP_PROP | GROUP_WORLD);

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
type Task = {
  target: PropId;
  region: { shape: "circle"; color: "yellow"; center: [number, number]; radius: number };
  instruction: string;
};

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

let task: Task = {
  target: "red",
  region: { shape: "circle", color: "yellow", center: [0, 0], radius: 0.075 },
  instruction: "",
};
let frames: EpisodeFrame[] = [];
let dataset: Array<ReturnType<typeof buildEpisode>> = [];
let script: {
  plan: Script;
  time: number;
  tick: number;
  /** Wall-clock guard against a simulation or capture step getting stuck. */
  startedAt: number;
  /** Which leg of the stack is running, and how much episode time the earlier legs used. */
  stage: number;
  clock: number;
  wait?: number;
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
renderer.shadowMap.type = THREE.PCFShadowMap;
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
observationRenderer.shadowMap.enabled = true;
observationRenderer.shadowMap.type = THREE.PCFShadowMap;
observationRenderer.outputColorSpace = THREE.SRGBColorSpace;
observationRenderer.toneMapping = THREE.ACESFilmicToneMapping;
observationRenderer.toneMappingExposure = 1.05;

// Let the HDR supply the lighting colour. Neutral, low-intensity fill/key lights retain
// shadow definition without imposing the same warm cast on every environment.
scene.environmentIntensity = 1.2;
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x808080, 0.08);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 0.45);
keyLight.position.set(1.2, 2, 0.8);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
// Metre-scale scene: avoid quantized self-shadow bands on the broad desktop.
keyLight.shadow.camera.near = 0.1;
keyLight.shadow.camera.far = 8;
keyLight.shadow.bias = -0.0001;
keyLight.shadow.normalBias = 0.002;
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
// Render-target textures belong to their WebGL context. The preview renderer needs its own
// prefiltered environment; sharing the main renderer's GPU texture loses indirect lighting.
const observationPmrem = new THREE.PMREMGenerator(observationRenderer);
observationPmrem.compileEquirectangularShader();
let observationEnvironment: THREE.Texture | null = null;
const rgbeLoader = new RGBELoader();

const backdrops = await Promise.all(
  BACKDROPS.map(async (name) => {
    const texture = await rgbeLoader.loadAsync(`/assets/hdri/${name}.hdr`);
    texture.mapping = THREE.EquirectangularReflectionMapping;
    return {
      background: texture,
      environment: pmrem.fromEquirectangular(texture).texture,
      observationEnvironment: observationPmrem.fromEquirectangular(texture).texture,
    };
  }),
);

function useBackdrop(index: number) {
  const backdrop = backdrops[index];
  scene.background = backdrop.background;
  scene.environment = backdrop.environment;
  observationEnvironment = backdrop.observationEnvironment;
}

function renderObservationView(view: THREE.Camera) {
  const mainEnvironment = scene.environment;
  const debugVisible = colliderDebug.visible;
  try {
    colliderDebug.visible = false;
    scene.environment = observationEnvironment;
    observationRenderer.render(scene, view);
  } finally {
    scene.environment = mainEnvironment;
    colliderDebug.visible = debugVisible;
  }
}

// The arm and props share a dynamic world. Grasping uses jaw contact friction.
await RAPIER.init();
const physics = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
physics.timestep = PHYSICS_STEP;
physics.numSolverIterations = 8;
// Rapier 0.20 defaults to a 20 mm prediction band. At this gripper's scale it
// creates corner contacts far before touch, resisting a clear vertical descent.
physics.integrationParameters.normalizedPredictionDistance = 0.0005;
physics.integrationParameters.normalizedAllowedLinearError = 0.00005;
// Stiff contacts keep force-limited fingers on the cube surface instead of
// allowing several millimetres of compression under the commanded grip force.
physics.integrationParameters.contact_natural_frequency = 240;

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

const TARGET_RADIUS = Math.max(0.07, spec.props.cube * 1.65);
const targetZone = new THREE.Group();
targetZone.name = "target-zone";
const targetDisk = new THREE.Mesh(
  new THREE.CircleGeometry(TARGET_RADIUS, 48),
  new THREE.MeshBasicMaterial({ color: 0xe6bb42, transparent: true, opacity: 0.34, depthWrite: false }),
);
targetDisk.rotation.x = -Math.PI / 2;
const targetRing = new THREE.Mesh(
  new THREE.RingGeometry(TARGET_RADIUS - 0.004, TARGET_RADIUS, 48),
  new THREE.MeshBasicMaterial({ color: 0x9d6f00, side: THREE.DoubleSide }),
);
targetRing.rotation.x = -Math.PI / 2;
targetRing.position.y = 0.0005;
targetZone.add(targetDisk, targetRing);
scene.add(targetZone);

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
robot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), spec.heading ?? 0));
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

/**
 * Grasping is friction, not a trick.
 *
 * Earlier this was a kinematic attach: the cube was pinned to the jaw and its collision with the
 * arm switched off while held. That made the grip unfalsifiable — commanding the jaws past the
 * cube drove the fingers straight through it instead of stalling against it, and letting go threw
 * the cube because the jaw's velocity had to be handed over by hand.
 *
 * Now the fingers are force-limited position motors told to close tighter than the cube. They
 * stall on it, the leftover command becomes squeeze, and the cube is held by nothing but contact
 * friction between two jaw plates. It can therefore also slip, be knocked aside on approach, or
 * be dropped — all of which are real outcomes worth having in a dataset.
 */
let physicsAccumulator = 0;

/** The cube currently pinched between both jaws, recomputed once per step from contacts. */
let gripped: Prop | null = null;
let grippedReport: Prop | null = null;

/** Separated manifold points are not contact; gripping additionally requires force. */
function touching(a: RAPIER.Collider, b: RAPIER.Collider, requireForce = false) {
  // A cached manifold can retain points/normals from the start of an approach.
  // First require the current convex shapes themselves to be touching.
  if (!a.contactCollider(b, 0.0001)) return false;
  let hit = false;
  physics.contactPair(a, b, (manifold, flipped) => {
    const first = flipped ? b : a;
    const second = flipped ? a : b;
    const p1 = first.translation(), p2 = second.translation();
    const r1 = first.rotation(), r2 = second.rotation();
    const q1 = new THREE.Quaternion(r1.x, r1.y, r1.z, r1.w);
    const q2 = new THREE.Quaternion(r2.x, r2.y, r2.z, r2.w);
    const localNormal = manifold.localNormal1();
    const normal = new THREE.Vector3(localNormal.x, localNormal.y, localNormal.z).applyQuaternion(q1);
    if (manifoldHasContact(manifold, requireForce, (index) => {
      const local1 = manifold.localContactPoint1(index);
      const local2 = manifold.localContactPoint2(index);
      if (!local1 || !local2) return Infinity;
      const world1 = new THREE.Vector3(local1.x, local1.y, local1.z).applyQuaternion(q1).add(new THREE.Vector3(p1.x, p1.y, p1.z));
      const world2 = new THREE.Vector3(local2.x, local2.y, local2.z).applyQuaternion(q2).add(new THREE.Vector3(p2.x, p2.y, p2.z));
      return world2.sub(world1).dot(normal);
    })) hit = true;
  });
  return hit;
}

function contactAndMotorDiagnostic() {
  const contacts: unknown[] = [];
  for (const prop of props) {
    fingerColliders.forEach((finger, fingerIndex) => {
      physics.contactPair(prop.collider, finger, (manifold) => {
        contacts.push({
          prop: prop.id,
          finger: fingerIndex,
          currentDistanceMm: prop.collider.contactCollider(finger, 0.05)?.distance !== undefined
            ? prop.collider.contactCollider(finger, 0.05)!.distance * 1000 : null,
          distancesMm: Array.from({ length: manifold.numContacts() }, (_, i) => manifold.contactDist(i) * 1000),
          impulsesNs: Array.from({ length: manifold.numContacts() }, (_, i) => manifold.contactImpulse(i)),
          solverDistancesMm: Array.from({ length: manifold.numSolverContacts() }, (_, i) => manifold.solverContactDist(i) * 1000),
        });
      });
    });
  }
  const shoulder = armJoints?.get("arm_joint2");
  if (!shoulder) return { contacts };
  const rotation = shoulder.parent.rotation();
  const q = new THREE.Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
  const at = shoulder.parent.translation();
  const anchor = shoulder.anchor.clone().applyQuaternion(q).add(new THREE.Vector3(at.x, at.y, at.z));
  const axis = shoulder.axis.clone().applyQuaternion(q).normalize();
  const gravityTorque = shoulder.load.reduce((sum, body) => {
    const com = body.worldCom();
    return sum + new THREE.Vector3(com.x, com.y, com.z).sub(anchor)
      .cross(new THREE.Vector3(0, -9.81 * body.mass(), 0)).dot(axis);
  }, 0);
  const compensation = THREE.MathUtils.clamp(-gravityTorque, -shoulder.maxEffort, shoulder.maxEffort);
  return {
    contacts,
    shoulder: {
      gravityTorqueNm: gravityTorque,
      compensationNm: compensation,
      motorHeadroomNm: Math.max(0, shoulder.maxEffort - Math.abs(compensation)),
      effortLimitNm: shoulder.maxEffort,
      gains: shoulder.gains,
      proportionalTorqueNm: shoulder.gains[0] * revolute(currentValues.arm_joint2 - measuredValues.arm_joint2),
      carriedMassKg: shoulder.load.reduce((sum, body) => sum + body.mass(), 0),
      loadBodies: shoulder.load.length,
    },
  };
}

function findGrippedProp() {
  if (fingerColliders.length < 2) return null;
  return props.find((prop) => bothFingersGrip(fingerColliders.map((finger) => touching(prop.collider, finger, true)))) ?? null;
}

function updateGrasp() {
  gripped = findGrippedProp();
  if (gripped === grippedReport) return;
  if (gripped) statusElement.textContent = `GRIPPING ${gripped.label.toUpperCase()} CUBE`;
  else if (grippedReport) statusElement.textContent = `LET GO OF ${grippedReport.label.toUpperCase()} CUBE`;
  grippedReport = gripped;
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
    // Recompute compensation at each fixed physics step, not at render frequency.
    applyGravityCompensation();
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
 * Impulse joints form a maximal-coordinate chain and need enough solver iterations to keep
 * the lightweight wrist/shoulder bodies constrained under motor and gravity loads. Force-based
 * motors use separate joint gains and the effort limits declared by this asset's URDF.
 */
/** The link bolted to the desk. Both URDFs here name it the same thing. */
const ROOT_LINK = "base_link";
/**
 * Simulation-tuned force-based PD gains, in Nm/rad and Nm s/rad. The official A1Z SDK uses
 * [30,30,30,20,5,5] / [1,1,1,0.5,0.5,0.5] on hardware; its gains are a reference, not a
 * drop-in match for this constraint solver. See tools/check-arm-stability.mjs for hold checks.
 * https://github.com/userguide-galaxea/GALAXEA-A1Z
 */
const ARM_MOTOR_GAINS: Record<string, [number, number]> = {
  arm_joint1: [80, 4], arm_joint2: [200, 12], arm_joint3: [150, 8],
  arm_joint4: [60, 3], arm_joint5: [15, 1], arm_joint6: [15, 1],
};
// Force-based units: N/m and N s/m. Grip force remains capped separately.
const GRIPPER_MOTOR_GAINS: [number, number] = [6_000, 40];
/**
 * The finger motors get a real limit instead, because here the cap *is* the grip force. A 45 mm
 * cube at 700 kg/m3 weighs 0.63 N, and two jaw plates at a combined friction of about 1.0 hold
 * 2 x mu x F, so this is a wide margin over slipping without being enough to fling anything.
 */
const GRIP_FORCE = 30;

/** Jaw plate colliders, used to tell by contact whether something is actually being gripped. */
const fingerColliders: RAPIER.Collider[] = [];
/** Every body in the arm, so accumulated torques can be cleared before each step. */
const armBodies: RAPIER.RigidBody[] = [];
const armLinkBodies = new Map<string, RAPIER.RigidBody>();
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
  gains: [number, number];
  maxEffort: number;
  target: number;
  trim: number;
  lower: number;
  upper: number;
  /** Every body this joint carries — its child and everything beyond it. */
  load: RAPIER.RigidBody[];
};

/**
 * Link-local vertices of what this link alone draws, thinned to the point budget.
 *
 * Note the hand-rolled walk: the URDF tree nests each link under the joint that drives it, so a
 * plain `traverse` would sweep up every link downstream and hand back a hull enclosing the whole
 * arm from here on. Descending only until the next joint is what keeps one hull to one link.
 */
function linkHullPoints(link: THREE.Object3D, fromX = -Infinity) {
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
      if (vertex.x < fromX) continue;
      points.push(vertex.x, vertex.y, vertex.z);
    }
  }
  return new Float32Array(points);
}

/** Diagonalize the symmetric URDF inertia tensor into Rapier's principal-axis form. */
function linkMassProperties(link: URDFLink) {
  const { mass, origin, inertia: i } = link.inertial;
  const a = [[i.ixx, i.ixy, i.ixz], [i.ixy, i.iyy, i.iyz], [i.ixz, i.iyz, i.izz]];
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let iteration = 0; iteration < 32; iteration += 1) {
    let p = 0, q = 1;
    for (const [row, col] of [[0, 2], [1, 2]]) {
      if (Math.abs(a[row][col]) > Math.abs(a[p][q])) { p = row; q = col; }
    }
    if (Math.abs(a[p][q]) < 1e-12) break;
    const angle = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(angle), s = Math.sin(angle);
    const app = a[p][p], aqq = a[q][q], apq = a[p][q];
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
    for (let k = 0; k < 3; k += 1) {
      if (k !== p && k !== q) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = a[p][k] = c * akp - s * akq;
        a[k][q] = a[q][k] = s * akp + c * akq;
      }
      const vkp = v[k][p], vkq = v[k][q];
      v[k][p] = c * vkp - s * vkq;
      v[k][q] = s * vkp + c * vkq;
    }
  }
  const principal = new THREE.Vector3(a[0][0], a[1][1], a[2][2]);
  if (!(mass > 0) || ![principal.x, principal.y, principal.z].every((x) => Number.isFinite(x) && x > 0)) {
    throw new Error(`Invalid inertia for ${link.urdfName}`);
  }
  const basis = new THREE.Matrix4().set(
    ...[...v[0], 0, ...v[1], 0, ...v[2], 0, 0, 0, 0, 1] as Parameters<THREE.Matrix4["set"]>,
  );
  const [roll, pitch, yaw] = origin.rpy;
  const frame = new THREE.Quaternion().setFromEuler(new THREE.Euler(roll, pitch, yaw, "ZYX"))
    .multiply(new THREE.Quaternion().setFromRotationMatrix(basis)).normalize();
  return { mass, center: new THREE.Vector3().fromArray(origin.xyz), principal, frame };
}

/** Rotated joint origins retain the existing kinematic fallback. Build at URDF zero pose. */
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

  // A finger's blade sits entirely on its own side of the jaw, but its hinge bracket reaches
  // across the centreline. Hull the whole part and you get a wedge that fills the gap: the two
  // fingers' hulls overlap, anything between them touches both no matter how wide they are, and
  // closing the jaws shoves the object instead of gripping it. So fingers are hulled from the
  // blade outwards. Measured off this gripper; the bracket ends at x = 0.02 in the link frame.
  const BLADE_FROM_X = 0.02;
  const fingerLinks = new Set(
    Object.values(robot.joints)
      .filter((joint) => joint.jointType === "prismatic")
      .map((joint) => (joint.children.find((c) => (c as URDFLink).isURDFLink) as URDFLink | undefined)?.urdfName)
      .filter((name): name is string => Boolean(name)),
  );

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

    const points = linkHullPoints(link, fingerLinks.has(name) ? BLADE_FROM_X : -Infinity);
    const desc = points.length >= 12 ? RAPIER.ColliderDesc.convexHull(points) : null;
    if (desc) {
      const collider = physics.createCollider(
        desc.setFriction(0.9).setRestitution(0).setCollisionGroups(collisionGroups(GROUP_ARM, GROUP_PROP | GROUP_WORLD)),
        body,
      );
      if (link.inertial?.mass > 0) collider.setDensity(0);
    }
    if (link.inertial?.mass > 0) {
      const { mass, center, principal, frame } = linkMassProperties(link);
      body.setAdditionalMassProperties(mass, center, principal, frame, false);
      body.recomputeMassPropertiesFromColliders();
    }
    bodies.set(name, body);
    armLinkBodies.set(name, body);
    armBodies.push(body);
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
    joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
    // The only prismatic joints on these arms are the gripper fingers.
    const maxEffort = prismatic ? GRIP_FORCE : urdfJoint.limit.effort;
    if (!(maxEffort > 0)) throw new Error(`Missing effort limit for ${name}`);
    joint.setMotorMaxForce(maxEffort);
    if (prismatic) {
      for (let index = 0; index < child.numColliders(); index += 1) {
        fingerColliders.push(child.collider(index));
      }
    }
    // Everything from the child link outwards is what this joint has to hold up.
    const load: RAPIER.RigidBody[] = [];
    const gather = (link: URDFLink) => {
      const body = bodies.get(link.urdfName);
      if (body) load.push(body);
      link.traverse((object) => {
        const child = object as URDFLink;
        if (child !== link && child.isURDFLink && bodies.has(child.urdfName)) {
          load.push(bodies.get(child.urdfName)!);
        }
      });
    };
    gather(childLink);

    const gains = prismatic ? GRIPPER_MOTOR_GAINS : (ARM_MOTOR_GAINS[name] ?? [30, 1]);
    joints.set(name, {
      joint, parent, child, axis, anchor, prismatic, load, gains, maxEffort,
      target: 0, trim: 0, lower: urdfJoint.limit.lower, upper: urdfJoint.limit.upper,
    });
  }

  // This light-link serial chain needs more iterations than independent props. The headless
  // regression checks both residual oscillation and target error, not merely finite positions.
  physics.numSolverIterations = 256;
  return joints;
}

const IDENTITY = new THREE.Quaternion();
const armJoints = buildArmDynamics();

// Draw the solver's actual collision geometry, not another approximation of the STL.
const colliderDebug = new THREE.LineSegments(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0xff9b22, depthTest: false, depthWrite: false, toneMapped: false }),
);
colliderDebug.name = "gripper-collision-debug";
colliderDebug.frustumCulled = false;
colliderDebug.renderOrder = 1000;
colliderDebug.visible = false;
scene.add(colliderDebug);
let showColliderDebug = false;
const collisionDebugButton = document.querySelector<HTMLButtonElement>("#collision-debug-toggle")!;
collisionDebugButton.addEventListener("click", () => {
  showColliderDebug = !showColliderDebug;
  collisionDebugButton.setAttribute("aria-pressed", String(showColliderDebug));
  collisionDebugButton.textContent = showColliderDebug ? "GRIPPER HULL ON" : "GRIPPER HULL";
});

function updateColliderDebug() {
  colliderDebug.visible = showColliderDebug && activeView === "orbit";
  if (!colliderDebug.visible) return;
  const handles = new Set(fingerColliders.map((collider) => collider.handle));
  const { vertices } = physics.debugRender(undefined, (collider) => handles.has(collider.handle));
  const attribute = colliderDebug.geometry.getAttribute("position");
  if (!attribute || attribute.array.length !== vertices.length) {
    colliderDebug.geometry.dispose();
    colliderDebug.geometry = new THREE.BufferGeometry();
    colliderDebug.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vertices), 3));
  } else {
    attribute.array.set(vertices);
    attribute.needsUpdate = true;
  }
}

// Capture zero-pose joint origins before recording or commanding any joint angles.
const constrainedIK = spec.id === "a1z" ? new ConstrainedIK(
  robot.matrixWorld.clone(),
  JOINTS.filter((control) => control.unit === "deg").map((control) => {
    const joint = robot.joints[control.urdfJoints[0]];
    return {
      name: control.name,
      origin: new THREE.Matrix4().compose(joint.position, joint.quaternion, new THREE.Vector3(1, 1, 1)),
      axis: joint.axis.clone().normalize(),
      min: Math.max(revolute(control.min), joint.limit.lower),
      max: Math.min(revolute(control.max), joint.limit.upper),
    };
  }),
  new THREE.Vector3(A1Z_JAW_ANCHOR, 0, 0),
) : null;

const parentRotation = new THREE.Quaternion();
const childRotation = new THREE.Quaternion();
const relativeRotation = new THREE.Quaternion();
const relativeOffset = new THREE.Vector3();

const GRAVITY = new THREE.Vector3(0, -9.81, 0);
const axisWorld = new THREE.Vector3();
const jointOrigin = new THREE.Vector3();
const lever = new THREE.Vector3();
const weight = new THREE.Vector3();
const torque = new THREE.Vector3();
const bodyRotation = new THREE.Quaternion();

/**
 * Gravity compensation.
 *
 * For each joint, sum the gravity torque of every
 * body it carries about its own axis and apply the opposite as an actuator torque pair — equal
 * and opposite on child and parent, which is what a real joint does. The motors are then only
 * correcting the residual. The combined compensation and motor torque stay within the URDF
 * effort limit. This is a static feed-forward model, not full inverse dynamics.
 */
function applyGravityCompensation() {
  if (!armJoints) return;
  // addTorque accumulates until reset, and this runs once per substep — without clearing first,
  // a frame's worth of substeps stacks up to fifty times the intended torque and the arm flies.
  for (const body of armBodies) body.resetTorques(false);

  for (const entry of armJoints.values()) {
    // The fingers carry 138 g each and are not fighting gravity in any meaningful way.
    if (entry.prismatic) continue;

    const at = entry.parent.rotation();
    bodyRotation.set(at.x, at.y, at.z, at.w);
    axisWorld.copy(entry.axis).applyQuaternion(bodyRotation).normalize();

    const from = entry.parent.translation();
    jointOrigin.copy(entry.anchor).applyQuaternion(bodyRotation).add(new THREE.Vector3(from.x, from.y, from.z));

    let along = 0;
    for (const body of entry.load) {
      const com = body.worldCom();
      lever.set(com.x - jointOrigin.x, com.y - jointOrigin.y, com.z - jointOrigin.z);
      weight.copy(GRAVITY).multiplyScalar(body.mass());
      along += lever.cross(weight).dot(axisWorld);
    }

    const compensation = THREE.MathUtils.clamp(-along, -entry.maxEffort, entry.maxEffort);
    // Reserve torque headroom for the feed-forward compensation applied below.
    entry.joint.setMotorMaxForce(Math.max(0, entry.maxEffort - Math.abs(compensation)));
    torque.copy(axisWorld).multiplyScalar(compensation);
    entry.child.addTorque(torque, false);
    torque.negate();
    entry.parent.addTorque(torque, false);
    updateMotorTrim(entry, Math.max(0, entry.maxEffort - Math.abs(compensation)));
  }
}

/** Slow, bounded integral correction of steady joint error, separate from task targets.
 * Only integrate near the target and at low joint speed. Do not accumulate further
 * into a joint limit or beyond the remaining motor torque budget.
 */
function updateMotorTrim(entry: ArmJoint, headroom: number) {
  const actual = measureJoint(entry);
  const error = entry.target - actual;
  const parentVelocity = entry.parent.angvel();
  const childVelocity = entry.child.angvel();
  // axisWorld was computed from this joint's parent by applyGravityCompensation.
  const speed = (childVelocity.x - parentVelocity.x) * axisWorld.x
    + (childVelocity.y - parentVelocity.y) * axisWorld.y
    + (childVelocity.z - parentVelocity.z) * axisWorld.z;
  const limit = revolute(3);
  if (Math.abs(error) < revolute(5) && Math.abs(speed) < revolute(10)) {
    const candidate = THREE.MathUtils.clamp(entry.trim + error * 1.2 * PHYSICS_STEP, -limit, limit);
    const requested = entry.target + candidate;
    const bounded = THREE.MathUtils.clamp(requested, entry.lower, entry.upper);
    const effort = entry.gains[0] * (bounded - actual) - entry.gains[1] * speed;
    const previousEffort = entry.gains[0] * (entry.target + entry.trim - actual) - entry.gains[1] * speed;
    if (requested === bounded &&
        (Math.abs(effort) <= headroom || Math.abs(effort) < Math.abs(previousEffort))) {
      entry.trim = candidate;
    }
  }
  entry.joint.configureMotorPosition(
    THREE.MathUtils.clamp(entry.target + entry.trim, entry.lower, entry.upper),
    entry.gains[0], entry.gains[1],
  );
}

function clearMotorTrims() {
  if (armJoints) for (const entry of armJoints.values()) entry.trim = 0;
}

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
      const entry = armJoints.get(name);
      if (!entry) continue;
      entry.target = THREE.MathUtils.clamp(target, entry.lower, entry.upper);
      entry.joint.configureMotorPosition(
        THREE.MathUtils.clamp(entry.target + entry.trim, entry.lower, entry.upper),
        entry.gains[0],
        entry.gains[1],
      );
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

/** Reset both worlds together; do not accelerate the arm from its old pose at episode start. */
function resetArmPhysics(values: JointValues) {
  clearMotorTrims();
  for (const control of JOINTS) setURDFJoint(robot, control, values[control.name]);
  robot.updateMatrixWorld(true);
  const position = new THREE.Vector3();
  const rotation = new THREE.Quaternion();
  const scale = new THREE.Vector3();
  for (const [name, body] of armLinkBodies) {
    robot.links[name].matrixWorld.decompose(position, rotation, scale);
    body.setTranslation(position, true);
    body.setRotation(rotation, true);
    body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    body.setAngvel({ x: 0, y: 0, z: 0 }, true);
    body.resetForces(false);
    body.resetTorques(false);
  }
  physicsAccumulator = 0;
  gripped = grippedReport = null;
  Object.assign(measuredValues, values);
  commandArm(values);
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

type SensorView = "wrist" | "overhead";
let activeView: SensorView | "orbit" = "orbit";
const sensorCameras = new Map<SensorView, THREE.PerspectiveCamera>();
const sensorCanvases = new Map<SensorView, HTMLCanvasElement>();
const cameraButtons = document.querySelectorAll<HTMLButtonElement>("[data-camera]");
let lastSensorPreview = -Infinity;

/** Visual mounts are added after building physics, so they do not alter link hulls/inertia. */
function setupSensorCameras() {
  const create = (id: SensorView, parent: THREE.Object3D, position: Vec3, target: Vec3, fov: number) => {
    const sensor = new THREE.PerspectiveCamera(fov, 4 / 3, 0.005, 20);
    sensor.position.set(...position);
    // Wrist frame: +X is forward, Y is the jaw opening, +Z is image-up.
    // Keep the two fingers horizontal in the image instead of rolling the view by 90 degrees.
    if (id === "wrist") sensor.up.set(0, 0, 1);
    // Orient in the mounting link's coordinates before parenting the camera.
    sensor.lookAt(new THREE.Vector3(...target));
    parent.add(sensor);
    sensorCameras.set(id, sensor);
    const canvas = document.querySelector<HTMLCanvasElement>(`#camera-${id}`)!;
    canvas.width = OBSERVATION_WIDTH;
    canvas.height = OBSERVATION_HEIGHT;
    sensorCanvases.set(id, canvas);

    const housing = new THREE.Group();
    housing.position.copy(sensor.position);
    housing.quaternion.copy(sensor.quaternion);
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.026, 0.024),
      new THREE.MeshStandardMaterial({ color: 0x20262a, roughness: 0.5 }));
    body.position.z = 0.018;
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.008, 16),
      new THREE.MeshStandardMaterial({ color: 0x56b7bd, metalness: 0.6, roughness: 0.2 }));
    lens.rotation.x = Math.PI / 2;
    lens.position.z = 0.005;
    housing.add(body, lens);
    parent.add(housing);
    if (id === "overhead") {
      // Table-mounted stand, behind the camera so it stays out of the forward view.
      const stand = new THREE.Group();
      stand.name = "center-camera-stand";
      const metal = new THREE.MeshStandardMaterial({ color: 0x30383d, metalness: 0.65, roughness: 0.38 });
      const rubber = new THREE.MeshStandardMaterial({ color: 0x171b1d, roughness: 0.9 });
      const poleX = position[0] - 0.035;
      const standX = spec.heading ? position[0] : poleX;
      const standZ = position[2] - (spec.heading ? 0.035 : 0);
      const addPart = (geometry: THREE.BufferGeometry, material: THREE.Material, at: THREE.Vector3) => {
        const part = new THREE.Mesh(geometry, material);
        part.position.copy(at);
        part.castShadow = true;
        part.receiveShadow = true;
        stand.add(part);
        return part;
      };
      addPart(new THREE.BoxGeometry(0.13, 0.006, spec.heading ? 0.075 : 0.13), rubber, new THREE.Vector3(standX, 0.003, standZ));
      addPart(new THREE.BoxGeometry(0.12, 0.012, spec.heading ? 0.070 : 0.12), metal, new THREE.Vector3(standX, 0.012, standZ));
      const rearMount = new THREE.Vector3(0, 0, 0.035).applyQuaternion(sensor.quaternion).add(sensor.position);
      const poleTop = new THREE.Vector3(standX, rearMount.y, standZ);
      const height = poleTop.y - 0.018;
      addPart(new THREE.CylinderGeometry(0.012, 0.012, height, 20), metal,
        new THREE.Vector3(standX, 0.018 + height / 2, standZ));
      addPart(new THREE.CylinderGeometry(0.02, 0.02, 0.024, 20), metal,
        new THREE.Vector3(standX, 0.03, standZ));
      addPart(new THREE.SphereGeometry(0.016, 16, 12), metal, poleTop);
      const connector = rearMount.clone().sub(poleTop);
      const arm = addPart(new THREE.CylinderGeometry(0.009, 0.009, connector.length(), 16), metal,
        poleTop.clone().add(rearMount).multiplyScalar(0.5));
      arm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), connector.normalize());
      addPart(new THREE.SphereGeometry(0.012, 16, 12), metal, rearMount);
      scene.add(stand);
    }
    if (id === "wrist") {
      const anchor = new THREE.Vector3(0.065, 0, 0.02);
      const bracket = new THREE.Line(new THREE.BufferGeometry().setFromPoints([anchor, sensor.position]),
        new THREE.LineBasicMaterial({ color: 0x43494d }));
      parent.add(bracket);
    }
  };
  const wrist = robot.links.arm_link6 ?? robot.links.wrist_roll_link ?? robot.links.base_link;
  // Centred above the jaws, looking straight along the tool rather than diagonally at it.
  // The fingertips sit below the optical axis, leaving the central image clear for the object.
  create("wrist", wrist, [0.085, 0, 0.055], [0.30, 0, 0.055], 80);
  // Central head camera: behind the base line, above both arms, facing the work area.
  const headCameraX = Math.max(spec.mount[0] - 0.08, -DESK.width / 2 + 0.11);
  create("overhead", scene,
    spec.heading ? [-0.10, spec.mount[1] + 0.50, spec.mount[2] + 0.035] : [headCameraX, spec.mount[1] + 0.50, 0],
    spec.heading ? [0, 0.045, spec.mount[2] + 0.48] : [spec.mount[0] + 0.48, 0.045, 0], 72);

  for (const button of cameraButtons) {
    button.addEventListener("click", () => {
      activeView = button.dataset.camera as typeof activeView;
      orbitControls.enabled = activeView === "orbit";
      sceneElement.style.cursor = activeView === "orbit" ? "grab" : "default";
      for (const item of cameraButtons) item.setAttribute("aria-pressed", String(item.dataset.camera === activeView));
      document.querySelector("#camera-view-label")!.textContent = activeView === "orbit"
        ? "DRAG TO ORBIT / SCROLL TO ZOOM"
        : `${activeView === "overhead" ? "CENTER" : "GRIPPER"} CAMERA / LIVE`;
    });
  }
}

/** Reuse the observation renderer for both live camera previews. */
function renderSensorPreviews(now: number) {
  if (now - lastSensorPreview < 100) return;
  lastSensorPreview = now;
  for (const [id, sensor] of sensorCameras) {
    sensor.aspect = OBSERVATION_WIDTH / OBSERVATION_HEIGHT;
    sensor.updateProjectionMatrix();
    renderObservationView(sensor);
    const canvas = sensorCanvases.get(id)!;
    canvas.getContext("2d")!.drawImage(observationRenderer.domElement, 0, 0, canvas.width, canvas.height);
  }
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
  renderObservationView(observationCamera);
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
      grasped: gripped?.id ?? null,
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
    task: { target: task.target, region: task.region },
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

function rollTask(center: THREE.Vector2): Task {
  const target = PROP_SPECS[Math.floor(Math.random() * PROP_SPECS.length)];
  const instructions = [
    `Move the ${target.label} cube into the yellow target circle.`,
    `Place the ${target.label} cube inside the yellow circle.`,
    `Pick up the ${target.label} block and put it in the yellow target area.`,
  ];
  return {
    target: target.id,
    region: { shape: "circle", color: "yellow", center: [center.x, center.y], radius: TARGET_RADIUS },
    instruction: instructions[Math.floor(Math.random() * instructions.length)],
  };
}

/** A spot claimed on the table, with the radius it needs kept clear around it. */
type Placement = { at: THREE.Vector2; clearance: number };

/** Test the same 4:3 framing as the CENTER preview, with a margin around every cube. */
function placementVisibleFromCenter(at: THREE.Vector2) {
  const center = sensorCameras.get("overhead");
  if (!center) return false;
  const view = center.clone();
  view.aspect = OBSERVATION_WIDTH / OBSERVATION_HEIGHT;
  view.updateProjectionMatrix();
  view.updateMatrixWorld(true);
  const half = spec.props.cube / 2;
  // Cubes start with arbitrary yaw: use their circumscribed horizontal square.
  const radius = half * Math.SQRT2;
  for (const dx of [-radius, radius]) {
    for (const dz of [-radius, radius]) {
      for (const y of [0, spec.props.cube]) {
        const point = new THREE.Vector3(at.x + dx, y, at.y + dz).project(view);
        if (Math.abs(point.x) > 0.85 || Math.abs(point.y) > 0.85 || point.z < -1 || point.z > 1) return false;
      }
    }
  }
  // Frustum inclusion alone does not catch a cube hidden behind a link or the gripper.
  const occluders: THREE.Object3D[] = [];
  robot.traverse((object) => { if (object instanceof THREE.Mesh) occluders.push(object); });
  const origin = view.getWorldPosition(new THREE.Vector3());
  const ray = new THREE.Raycaster();
  for (const [dx, dz] of [[0, 0], [-half, -half], [-half, half], [half, -half], [half, half]]) {
    const direction = new THREE.Vector3(at.x + dx, spec.props.cube, at.y + dz).sub(origin);
    ray.set(origin, direction.clone().normalize());
    ray.near = view.near;
    ray.far = direction.length() - 0.002;
    if (ray.intersectObjects(occluders, false).length > 0) return false;
  }
  return true;
}

/**
 * Samples a spot in the arm's reachable annulus that clears everything placed so far. Returns
 * null rather than falling back to a fixed spot, so a cramped layout is retried instead of
 * silently overlapping.
 */
function samplePlacement(taken: Placement[], clearance: number) {
  const { min, max, yaw } = spec.reach;
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const radius = THREE.MathUtils.lerp(min, max, Math.random());
    // Sample the arm's right half-workspace, then rotate it into the mounting orientation.
    const angle = THREE.MathUtils.lerp(-yaw, 0, Math.random());
    const heading = spec.heading ?? 0;
    const offset = new THREE.Vector3(radius * Math.cos(angle), 0, -radius * Math.sin(angle))
      .applyAxisAngle(new THREE.Vector3(0, 1, 0), heading);
    const at = new THREE.Vector2(spec.mount[0] + offset.x, spec.mount[2] + offset.z);
    // An off-centre arm's reach can extend beyond the desktop. Keep every cube on the table.
    const edgeMargin = spec.props.cube / 2 + 0.015;
    if (Math.abs(at.x) > DESK.width / 2 - edgeMargin ||
        Math.abs(at.y) > DESK.depth / 2 - edgeMargin) continue;
    if (offset.x * Math.sin(heading) + offset.z * Math.cos(heading) < spec.props.cube + 0.02) continue;
    if (taken.every((other) => other.at.distanceTo(at) >= clearance + other.clearance)) {
      if (!placementVisibleFromCenter(at)) continue;
      return { at, clearance };
    }
  }
  return null;
}

/**
 * One spot per cube, or null if this attempt boxed itself in. Leave room around each cube
 * so approaching fingers and carried objects do not collide with neighbouring props.
 */
function sampleLayout() {
  // Keep generous separation while allowing three cubes in the narrower visible half-workspace.
  const clearance = spec.props.cube * 1.6;
  const taken: Placement[] = [];
  for (let index = 0; index < props.length; index += 1) {
    const placement = samplePlacement(taken, clearance);
    if (!placement) return null;
    taken.push(placement);
  }
  const target = samplePlacement(taken, TARGET_RADIUS);
  return target ? { props: taken, target } : null;
}

function newEpisode() {
  // Visibility must be evaluated at the episode's initial pose, not the previous final pose.
  resetPose();
  scene.updateMatrixWorld(true);
  let layout = sampleLayout();
  for (let retry = 0; !layout && retry < 20; retry += 1) layout = sampleLayout();
  if (!layout) {
    statusElement.textContent = "NO VISIBLE RIGHT-SIDE LAYOUT FOUND - ADJUST CENTER CAMERA OR REACH";
    return false;
  }

  const spots = layout.props.map((placement) => placement.at);
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
  targetZone.position.set(layout.target.at.x, 0.001, layout.target.at.y);
  // A fixed backdrop is a shortcut: the same wall in the same place is a free position cue, and
  // a policy will happily use it instead of looking at the cubes. Re-rolling it per episode
  // forces the visual encoder onto the desk and the props.
  useBackdrop(Math.floor(Math.random() * backdrops.length));
  task = rollTask(layout.target.at);
  instructionElement.value = task.instruction;
  frames = [];
  replayButton.disabled = true;
  downloadButton.disabled = true;
  updateStats(0);
  // Refresh the preview so it shows the scene you are actually looking at. Skipped while
  // generating, where the episode's own first capture lands a frame a moment later anyway.
  if (!generation) previewElement.src = renderObservation();
  statusElement.textContent = "NEW TASK — adjust a joint, then start recording";
  return true;
}

function taskSucceeded() {
  if (gripped) return false;
  const cube = spec.props.cube;
  const prop = props.find((entry) => entry.id === task.target)!;
  const [x, z] = task.region.center;
  const fullyInside = Math.hypot(prop.mesh.position.x - x, prop.mesh.position.z - z)
    <= task.region.radius - cube * Math.SQRT2 / 2;
  const onTable = Math.abs(prop.mesh.position.y - cube / 2) <= cube * 0.2;
  const velocity = prop.body.linvel();
  return fullyInside && onTable && Math.hypot(velocity.x, velocity.y, velocity.z) <= 0.03;
}

function updateTaskState() {
  const done = taskSucceeded();
  taskStateElement.textContent = done ? "In target" : "Not in target";
  taskStateElement.classList.toggle("is-success", done);
}

/** One leg of the scripted trajectory: ease to `pose` over `duration` seconds. */
type ScriptStep = {
  pose: JointValues;
  duration: number;
  cartesian?: { from: THREE.Vector3; to: THREE.Vector3; pitch: number; azimuth: number; gripper: number };
  gripHold?: { target: PropId; min: number; max: number };
  completion?: "joint";
};
type Script = { steps: ScriptStep[]; start: JointValues; total: number };

// Demonstration speed, not the arm's limit. At 5 Hz capture this puts roughly 6 degrees
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

function buildTransfer(sourceId: PropId, destination: THREE.Vector2): Script | null {
  if (!spec.solve) return null;
  const source = props.find((entry) => entry.id === sourceId)!;
  const cube = spec.props.cube;

  const sourceAt = source.body.translation();
  const sourceYaw = new THREE.Euler().setFromQuaternion(source.mesh.quaternion, "YXZ").y;

  const grasp = new THREE.Vector3(sourceAt.x, sourceAt.y, sourceAt.z);
  const place = new THREE.Vector3(
    destination.x,
    cube / 2 + 0.002,
    destination.y,
  );

  const plan = ((): Array<{ pose: JointValues; hold?: number; gripHold?: ScriptStep["gripHold"]; completion?: ScriptStep["completion"] }> | null => {
    // Keep the lateral transfer clearly above the cubes, while retaining lower
    // fallbacks for targets near the edge of the reachable workspace.
    for (let hover = 0.065; hover >= 0.0475; hover -= 0.00875) {
      const overSource = grasp.clone().setY(grasp.y + hover);
      const overPlace = place.clone().setY(place.y + hover);
      const pitch = Math.PI / 2;
        let ikSeed = { ...currentValues };
        const at = (point: THREE.Vector3, gripper: number) => {
          const solution = spec.solve!(spec.mount, point, pitch, sourceYaw, ikSeed);
          if (solution) ikSeed = solution;
          return solution ? { ...solution, gripper } : null;
        };
        // Hold poses to let the jaws establish contact and placed cubes settle.
        const attempt: Array<{ pose: JointValues | null; hold?: number; gripHold?: ScriptStep["gripHold"]; completion?: ScriptStep["completion"] }> = [
          { pose: at(overSource, GRIPPER_OPEN) },
          { pose: at(grasp, GRIPPER_OPEN) },
          { pose: at(grasp, GRIPPER_SHUT) },
          { pose: at(grasp, GRIPPER_SHUT), hold: 0.8, gripHold: { target: sourceId, min: 0.15, max: 0.8 } },
          { pose: at(overSource, GRIPPER_SHUT) },
          { pose: at(overPlace, GRIPPER_SHUT) },
          { pose: at(place, GRIPPER_SHUT) },
          { pose: at(place, GRIPPER_SHUT), hold: 0.6 },
          { pose: at(place, GRIPPER_OPEN) },
          { pose: at(place, GRIPPER_OPEN), hold: 0.8 },
          { pose: at(overPlace, GRIPPER_OPEN) },
          { pose: at(overPlace, GRIPPER_OPEN), hold: 0.5 },
          // End every completed demonstration in the same neutral state it began in.
          { pose: { ...initialValues }, completion: "joint" },
        ];
        if (attempt.every((entry) => entry.pose !== null)) {
          return attempt as Array<{ pose: JointValues; hold?: number; gripHold?: ScriptStep["gripHold"]; completion?: ScriptStep["completion"] }>;
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
    return { pose, duration, gripHold: entry.gripHold, completion: entry.completion };
  });
  // Keep each vertical approach/retraction straight in task space. Joint interpolation
  // between two valid IK solutions otherwise sweeps the fingers sideways into the cube.
  const tcpAt = (pose: JointValues) => {
    return constrainedIK!.forwardPose(pose).position;
  };
  const trajectory: ScriptStep[] = steps;
  for (const index of [1, 4, 6, 10]) {
    const from = tcpAt(steps[index - 1].pose);
    const to = tcpAt(steps[index].pose);
    const azimuth = sourceYaw;
    const gripper = steps[index].pose.gripper;
    const toolForward = new THREE.Vector3(1, 0, 0)
      .applyQuaternion(constrainedIK!.forwardPose(steps[index].pose).rotation);
    const pitch = Math.asin(THREE.MathUtils.clamp(-toolForward.y, -1, 1));
    // Reject paths with unreachable intermediate points before starting the motion.
    let pathSeed = steps[index - 1].pose;
    for (let sample = 0; sample <= 20; sample += 1) {
      const solution = spec.solve(spec.mount, from.clone().lerp(to, sample / 20), pitch, azimuth, pathSeed);
      if (!solution) return null;
      pathSeed = solution;
    }
    trajectory[index].cartesian = { from, to, pitch, azimuth, gripper };
    const duration = Math.max(steps[index].duration, from.distanceTo(to) / 0.025);
    total += duration - steps[index].duration;
    steps[index].duration = duration;
  }
  return { steps: trajectory, start, total };
}

/** The commanded pose at a point in the script, eased between waypoints. */
function poseAt(script: Script, time: number): JointValues {
  let previous = script.start;
  let clock = 0;
  for (const step of script.steps) {
    if (time < clock + step.duration) {
      const alpha = THREE.MathUtils.smoothstep((time - clock) / step.duration, 0, 1);
      if (step.cartesian) {
        const { from, to, pitch, azimuth, gripper } = step.cartesian;
        const solution = spec.solve!(spec.mount, from.clone().lerp(to, alpha), pitch, azimuth, currentValues);
        if (!solution) throw new Error("Cartesian grasp path became unreachable");
        return { ...solution, gripper };
      }
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
  resetArmPhysics(initialValues);
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
} | {
  root: { name: string };
  localRun: string;
};

let writer: DiskWriter | null = null;

const canWriteToDisk = () => "showDirectoryPicker" in window;

async function postDataset(route: string, payload: unknown) {
  const response = await fetch(`/__dataset/${route}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error ?? "Dataset write failed");
  return result;
}

async function writeMetadata(contents: string) {
  const destination = writer!;
  if ("localRun" in destination) {
    await postDataset("meta", { run: destination.localRun, meta: JSON.parse(contents) });
  } else {
    await writeFile(destination.root, "meta.json", contents);
  }
}

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
  const destination = writer!;
  if ("localRun" in destination) {
    await postDataset("episode", { run: destination.localRun, index, episode });
    return;
  }
  const name = `episode_${String(index).padStart(5, "0")}`;
  const directory = await destination.episodes.getDirectoryHandle(name, { create: true });
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
}

async function writeDatasetMeta(summary: { completed: number; succeeded: number }) {
  await writeMetadata(
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
const EPISODE_TIMEOUT_MS = 60_000;

function startGeneratedEpisode() {
  for (let attempt = 0; attempt < 8; attempt += 1) {
    // A failed layout must never fall through to planning with the previous
    // task and frame buffer, otherwise consecutive episodes become duplicates.
    if (!newEpisode()) continue;
    const plan = buildTransfer(task.target, new THREE.Vector2(...task.region.center));
    if (plan) {
      script = { plan, time: 0, tick: 0, startedAt: performance.now(), stage: 0, clock: 0 };
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
  updateGrasp();

  if (script.tick % TICKS_PER_CAPTURE === 0) captureFrame(script.clock + script.time);
  script.tick += 1;
  let boundary = 0;
  for (const step of plan.steps) {
    boundary += step.duration;
    if (boundary <= script.time) continue;
    if (step.gripHold) {
      const stepStart = boundary - step.duration;
      const elapsed = Math.max(0, script.time - stepStart);
      script.wait = gripped?.id === step.gripHold.target ? (script.wait ?? 0) + GENERATION_DT : 0;
      if (script.wait >= step.gripHold.min) {
        const shortened = Math.max(GENERATION_DT, elapsed);
        plan.total -= step.duration - shortened;
        boundary = stepStart + shortened;
        step.duration = shortened;
        script.wait = 0;
        break;
      }
      if (elapsed >= step.gripHold.max) {
        throw new Error(`GRIP TIMEOUT - ${step.gripHold.target.toUpperCase()} cube did not establish a stable bilateral grip`);
      }
      break;
    }
    if (step.cartesian && step.cartesian.gripper === GRIPPER_OPEN &&
        step.cartesian.to.y < step.cartesian.from.y &&
        props.some((prop) => fingerColliders.some((finger) => touching(prop.collider, finger)))) {
      throw new Error("APPROACH COLLISION - a finger touched a cube before closing");
    }
    if (script.time + GENERATION_DT >= boundary) {
      // Redundant joint errors can cancel at the tool. Gate A1Z waypoints on the
      // actual TCP pose, which is the task-space quantity the IK was asked to reach.
      const gripperReached = step.pose.gripper < GRIPPER_OPEN
        || Math.abs(measuredValues.gripper - step.pose.gripper) <= 3;
      const jointReached = JOINTS.every((joint) => joint.unit === "percent"
        || Math.abs(measuredValues[joint.name] - step.pose[joint.name]) <= 0.5);
      let tcpReached = constrainedIK === null;
      let orientationReached = constrainedIK === null;
      const wristBody = armLinkBodies.get("arm_link6");
      const expectedPose = constrainedIK?.forwardPose(step.pose);
      if (expectedPose && wristBody) {
        const at = wristBody.translation();
        const rot = wristBody.rotation();
        const actualRotation = new THREE.Quaternion(rot.x, rot.y, rot.z, rot.w);
        const actual = new THREE.Vector3(A1Z_JAW_ANCHOR, 0, 0)
          .applyQuaternion(actualRotation)
          .add(new THREE.Vector3(at.x, at.y, at.z));
        tcpReached = actual.distanceTo(expectedPose.position) <= 0.001;
        orientationReached = actualRotation.angleTo(expectedPose.rotation) <= revolute(0.75);
      }
      const reached = gripperReached && (constrainedIK ? tcpReached && orientationReached : jointReached);
      const reachedForStep = step.completion === "joint"
        ? gripperReached && jointReached
        : reached;
      if (!reachedForStep) {
        script.wait = (script.wait ?? 0) + GENERATION_DT;
        if (script.wait > 3) {
          const expected = constrainedIK?.forwardPose(step.pose);
          const bodyAt = wristBody?.translation();
          const bodyRot = wristBody?.rotation();
          const actualRotation = bodyRot ? new THREE.Quaternion(bodyRot.x, bodyRot.y, bodyRot.z, bodyRot.w) : null;
          const actualTCP = bodyAt && actualRotation ? new THREE.Vector3(A1Z_JAW_ANCHOR, 0, 0)
            .applyQuaternion(actualRotation).add(new THREE.Vector3(bodyAt.x, bodyAt.y, bodyAt.z)) : null;
          const diagnostic = {
            physics: contactAndMotorDiagnostic(),
            stage: script.stage + 1,
            waypoint: plan.steps.indexOf(step) + 1,
            simulatedSeconds: round4(script.clock + script.time),
            waitSeconds: round4(script.wait),
            joints: JOINTS.map((joint) => ({
              name: joint.name,
              unit: joint.unit,
              target: round4(step.pose[joint.name]),
              command: round4(currentValues[joint.name]),
              actual: round4(measuredValues[joint.name]),
              error: round4(measuredValues[joint.name] - step.pose[joint.name]),
              servoTrimDeg: joint.unit === "deg" ? round4(measured(armJoints?.get(joint.urdfJoints[0])?.trim ?? 0)) : null,
              angularVelocity: armJoints?.get(joint.urdfJoints[0])?.child.angvel(),
            })),
            targetTCP: expected?.position.toArray(),
            actualTCP: actualTCP?.toArray(),
            tcpErrorMm: expected && actualTCP ? round4(expected.position.distanceTo(actualTCP) * 1000) : null,
            orientationErrorDeg: expected && actualRotation ? round4(measured(expected.rotation.angleTo(actualRotation))) : null,
            gripped: gripped?.id ?? null,
            fingerContacts: props.filter((prop) => fingerColliders.some((finger) => touching(prop.collider, finger))).map((prop) => prop.id),
          };
          console.warn("TRACKING_DIAGNOSTIC " + JSON.stringify(diagnostic));
          throw new Error(`TRACKING ERROR - stage ${diagnostic.stage}, waypoint ${diagnostic.waypoint}, TCP error ${diagnostic.tcpErrorMm} mm; see TRACKING_DIAGNOSTIC`);
        }
        script.clock += GENERATION_DT;
        return true;
      }
      script.wait = 0;
    }
    break;
  }
  script.time += GENERATION_DT;
  if (script.time < plan.total) return true;

  return false;
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
        statusElement.textContent = summary.completed > 0
          ? `${report} — RUN tools/to_lerobot.py ON ${finished.root.name}/`
          : `${report} — NO COMPLETED EPISODES SAVED`;
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
  const outputMode = document.querySelector<HTMLSelectElement>("#generation-output")!.value;
  if (outputMode === "local") {
    try {
      const result = await postDataset("start", {});
      writer = { root: { name: result.name }, localRun: result.run };
    } catch (error) {
      statusElement.textContent = `CANNOT WRITE PROJECT data/ - USE THE LOCAL VITE SERVER: ${error}`;
      return;
    }
  } else if (canWriteToDisk() && outputMode === "folder") {
    try {
      // Must be called straight off the click: the picker needs the user gesture.
      const root = await window.showDirectoryPicker({ mode: "readwrite" });
      writer = { root, episodes: await root.getDirectoryHandle("episodes", { create: true }) };
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
    : `GENERATING 0/${requested} - IN MEMORY / DOWNLOAD JSON WHEN COMPLETE`;
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
    if (performance.now() - script.startedAt >= EPISODE_TIMEOUT_MS) {
      finishGeneration("EPISODE TIMEOUT - the current episode did not finish within 60 seconds");
      return;
    }
    let advancing: boolean;
    try {
      advancing = advanceScript();
    } catch (error) {
      Object.assign(currentValues, measuredValues);
      Object.assign(targetValues, measuredValues);
      clearMotorTrims();
      commandArm(currentValues);
      finishGeneration(error instanceof Error ? error.message : String(error));
      return;
    }
    if (!advancing) {
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
    updateGrasp();
  }
  updateTaskState();
  orbitControls.update();
  // Mounted cameras need their parent link transforms updated before rendering.
  scene.updateMatrixWorld(true);
  renderSensorPreviews(now);
  updateColliderDebug();
  const viewCamera = activeView === "orbit" ? camera : sensorCameras.get(activeView)!;
  viewCamera.aspect = camera.aspect;
  viewCamera.updateProjectionMatrix();
  renderer.render(scene, viewCamera);

  if (recording && now - lastSample >= SAMPLE_INTERVAL) {
    captureFrame((now - recordStart) / 1000);
    lastSample = now;
  }
}

setupSensorCameras();
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
