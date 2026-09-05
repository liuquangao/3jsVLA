import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";
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
  toURDF: (value: number) => number;
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
    /** Gripper control percentages that close on / let go of an object. */
    closeBelow: number;
    openAbove: number;
  };
  /**
   * Props are scattered into an annulus in front of the base. `min`/`max` are radii in metres
   * and `yaw` a half-angle in radians; the band has to stay inside the arm's top-down grasp
   * envelope, and be roomy enough for two zones and three cubes to fit without overlapping.
   */
  reach: { min: number; max: number; yaw: number };
  /** Prop sizes, scaled to the arm. `cube` must stay under the gripper's jaw opening. */
  props: { cube: number; zoneInner: number; zoneOuter: number };
  /**
   * Closed-form IK, when this arm has one. Without it the scripted generator is unavailable
   * and the arm is joint-slider only. `pitch` is how far the tool points below horizontal and
   * `jawAzimuth` which way the jaw opening faces, both in radians.
   */
  solve?: (mount: Vec3, target: THREE.Vector3, pitch: number, jawAzimuth: number) => JointValues | null;
  joints: JointSpec[];
};

const revolute = (value: number) => THREE.MathUtils.degToRad(value);

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
  view: {
    camera: [1.16, 0.88, 1.16],
    target: [0, 0.26, 0],
    distance: [0.9, 4],
    obsCamera: [0.94, 0.72, 0.94],
    obsTarget: [-0.06, 0.18, 0],
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
  // The G1Z jaw tips meet at x = 0.183 in the arm_link6 frame and open to 60 mm, so a
  // 45 mm cube is pinched at 75% of the control range. Grip below 70%, let go above 82%.
  grasp: {
    link: "arm_link6",
    anchor: [0.17, 0, 0],
    region: [0.042, 0.032, 0.032],
    closeBelow: 70,
    openAbove: 82,
  },
  // A top-down grasp reaches out to 0.47 m at cube height before the ±75° wrist pitch runs out.
  reach: { min: 0.27, max: 0.43, yaw: THREE.MathUtils.degToRad(55) },
  props: { cube: 0.045, zoneInner: 0.07, zoneOuter: 0.085 },
  solve: solveA1Z,
  joints: [
    { name: "arm_joint1", label: "J1 base yaw", min: -120, max: 120, initial: 0, unit: "deg", urdfJoints: ["arm_joint1"], toURDF: revolute },
    { name: "arm_joint2", label: "J2 shoulder", min: 0, max: 180, initial: 100, unit: "deg", urdfJoints: ["arm_joint2"], toURDF: revolute },
    { name: "arm_joint3", label: "J3 elbow", min: -180, max: 0, initial: -100, unit: "deg", urdfJoints: ["arm_joint3"], toURDF: revolute },
    { name: "arm_joint4", label: "J4 wrist pitch", min: -75, max: 75, initial: 55, unit: "deg", urdfJoints: ["arm_joint4"], toURDF: revolute },
    { name: "arm_joint5", label: "J5 wrist yaw", min: -85, max: 85, initial: 0, unit: "deg", urdfJoints: ["arm_joint5"], toURDF: revolute },
    { name: "arm_joint6", label: "J6 wrist roll", min: -115, max: 115, initial: 0, unit: "deg", urdfJoints: ["arm_joint6"], toURDF: revolute },
    {
      name: "gripper",
      label: "Gripper",
      min: 0,
      max: 100,
      initial: 100,
      unit: "percent",
      urdfJoints: ["gripper_finger_left_joint", "gripper_finger_rIght_joint"],
      toURDF: (value) => (value / 100) * A1Z_FINGER_STROKE,
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
  grasp: {
    link: "gripper_frame_link",
    anchor: [0, 0, 0],
    region: [0.03, 0.03, 0.03],
    closeBelow: 22,
    openAbove: 38,
  },
  reach: { min: 0.13, max: 0.26, yaw: THREE.MathUtils.degToRad(55) },
  props: { cube: 0.025, zoneInner: 0.038, zoneOuter: 0.048 },
  joints: [
    { name: "shoulder_pan", label: "Shoulder pan", min: -110, max: 110, initial: 0, unit: "deg", urdfJoints: ["shoulder_pan"], toURDF: revolute },
    { name: "shoulder_lift", label: "Shoulder lift", min: -100, max: 100, initial: -28, unit: "deg", urdfJoints: ["shoulder_lift"], toURDF: revolute },
    { name: "elbow_flex", label: "Elbow flex", min: -97, max: 97, initial: 62, unit: "deg", urdfJoints: ["elbow_flex"], toURDF: revolute },
    { name: "wrist_flex", label: "Wrist flex", min: -95, max: 95, initial: -32, unit: "deg", urdfJoints: ["wrist_flex"], toURDF: revolute },
    { name: "wrist_roll", label: "Wrist roll", min: -157, max: 163, initial: 0, unit: "deg", urdfJoints: ["wrist_roll"], toURDF: revolute },
    {
      name: "gripper",
      label: "Gripper",
      min: 0,
      max: 100,
      initial: 70,
      unit: "percent",
      urdfJoints: ["gripper"],
      toURDF: (value) => THREE.MathUtils.degToRad(THREE.MathUtils.lerp(-10, 100, value / 100)),
    },
  ],
};

const ROBOTS = [A1Z, SO101];
const requestedRobot = new URLSearchParams(window.location.search).get("robot");
const spec = ROBOTS.find((entry) => entry.id === requestedRobot) ?? A1Z;
const JOINTS = spec.joints;

const CAPTURE_HZ = 5;
const SAMPLE_INTERVAL = 1000 / CAPTURE_HZ;

/** Work surface is the top of this box, at y = 0. */
const TABLE = { width: 1.18, thickness: 0.07, depth: 0.82 };
const PHYSICS_STEP = 1 / 120;
/** What the policy sees. The converter reads these back out of the dump's meta.json. */
const OBSERVATION_WIDTH = 256;
const OBSERVATION_HEIGHT = 192;

type PropId = "red" | "green" | "blue";
type ZoneShape = "circle" | "square";

const PROP_SPECS: Array<{ id: PropId; label: string; color: number }> = [
  { id: "red", label: "red", color: 0xd0472c },
  { id: "green", label: "green", color: 0x4a9257 },
  { id: "blue", label: "blue", color: 0x3a6ba5 },
];

// Zones are told apart by shape, not colour, so they never compete with the cube colours.
const ZONE_SPECS: Array<{ id: ZoneShape; label: string }> = [
  { id: "circle", label: "circle" },
  { id: "square", label: "square" },
];

type Prop = { id: PropId; label: string; color: number; mesh: THREE.Mesh; body: RAPIER.RigidBody };
type Zone = { id: ZoneShape; label: string; mesh: THREE.Mesh };
type Task = { object: PropId; target: ZoneShape; instruction: string };

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
const currentValues = { ...initialValues };
const targetValues = { ...initialValues };
const sliders = new Map<string, HTMLInputElement>();
const valueLabels = new Map<string, HTMLElement>();

let task: Task = { object: "red", target: "circle", instruction: "" };
let frames: EpisodeFrame[] = [];
let dataset: Array<ReturnType<typeof buildEpisode>> = [];
let script: { plan: Script; time: number; tick: number } | null = null;
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
scene.background = new THREE.Color("#cbc5b8");
scene.fog = new THREE.Fog("#cbc5b8", 2.8, 6);

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

const hemiLight = new THREE.HemisphereLight(0xfff8e7, 0x5d6259, 2.2);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xfff2d8, 3.6);
keyLight.position.set(1.2, 2, 0.8);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
keyLight.shadow.camera.left = -1.9;
keyLight.shadow.camera.right = 1.9;
keyLight.shadow.camera.top = 1.9;
keyLight.shadow.camera.bottom = -1.9;
scene.add(keyLight);

const tableMaterial = new THREE.MeshStandardMaterial({ color: 0xd7c8ac, roughness: 0.86 });
const tableTop = new THREE.Mesh(
  new THREE.BoxGeometry(TABLE.width, TABLE.thickness, TABLE.depth),
  tableMaterial,
);
tableTop.position.y = -TABLE.thickness / 2;
tableTop.receiveShadow = true;
scene.add(tableTop);

for (const x of [-0.51, 0.51]) {
  for (const z of [-0.33, 0.33]) {
    const leg = new THREE.Mesh(
      new THREE.BoxGeometry(0.055, 0.58, 0.055),
      new THREE.MeshStandardMaterial({ color: 0x262723, roughness: 0.75 }),
    );
    leg.position.set(x, -0.32, z);
    leg.castShadow = true;
    scene.add(leg);
  }
}

// Physics runs for the props only. The arm stays kinematic — it is driven straight from the
// joint controls and has no colliders — which keeps the recorded actions clean and dodges the
// soft, drooping serial chain that an impulse solver gives you for a force-driven arm. Grasping
// is a kinematic attach for the same reason: it never jitters and the pickup moment is legible.
await RAPIER.init();
const physics = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
physics.timestep = PHYSICS_STEP;
physics.numSolverIterations = 8;

// The table collider matches the visual box exactly, so the work surface really is y = 0.
physics.createCollider(
  RAPIER.ColliderDesc.cuboid(TABLE.width / 2, TABLE.thickness / 2, TABLE.depth / 2)
    .setTranslation(0, -TABLE.thickness / 2, 0)
    .setFriction(0.9),
);
// Floor, so a cube knocked off the table lands somewhere instead of falling forever.
physics.createCollider(
  RAPIER.ColliderDesc.cuboid(3, 0.05, 3).setTranslation(0, -0.66, 0).setFriction(0.9),
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
  physics.createCollider(
    RAPIER.ColliderDesc.cuboid(spec.props.cube / 2, spec.props.cube / 2, spec.props.cube / 2)
      .setFriction(1.1)
      .setRestitution(0.05)
      .setDensity(700),
    body,
  );

  return { ...propSpec, mesh, body };
});

/** Zones are markings on the table, not obstacles, so they get no collider. */
function zoneGeometry(shape: ZoneShape) {
  const { zoneInner: inner, zoneOuter: outer } = spec.props;
  if (shape === "circle") return new THREE.RingGeometry(inner, outer, 48);
  const outline = new THREE.Shape();
  outline.moveTo(-outer, -outer);
  outline.lineTo(outer, -outer);
  outline.lineTo(outer, outer);
  outline.lineTo(-outer, outer);
  outline.closePath();
  const hole = new THREE.Path();
  hole.moveTo(-inner, -inner);
  hole.lineTo(-inner, inner);
  hole.lineTo(inner, inner);
  hole.lineTo(inner, -inner);
  hole.closePath();
  outline.holes.push(hole);
  return new THREE.ShapeGeometry(outline);
}

const zones: Zone[] = ZONE_SPECS.map((zoneSpec) => {
  const mesh = new THREE.Mesh(
    zoneGeometry(zoneSpec.id),
    new THREE.MeshBasicMaterial({ color: 0x3a3f45, side: THREE.DoubleSide }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.001;
  scene.add(mesh);
  return { ...zoneSpec, mesh };
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
  statusElement.textContent = `GRIPPED ${prop.label.toUpperCase()} CUBE`;
}

/** Hands the prop back to the physics world, carrying the jaw's velocity with it. */
function releaseProp(deltaSeconds: number) {
  const prop = heldProp;
  if (!prop) return;
  heldProp = null;
  setPropDynamic(prop);
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
    if (opening > spec.grasp.openAbove) {
      releaseProp(deltaSeconds);
    } else {
      heldProp.body.setNextKinematicTranslation(jawPosition);
      heldProp.body.setNextKinematicRotation(jawRotation.clone().multiply(heldRotation));
    }
  } else if (opening < spec.grasp.closeBelow) {
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

function setRobotPose(values: JointValues) {
  for (const joint of JOINTS) {
    setURDFJoint(robot, joint, values[joint.name]);
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
      joint_positions: cloneJointValues(currentValues),
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
    format: "nanovla.episode.v1",
    robot: spec.id,
    instruction: instructionElement.value.trim(),
    task: { object: task.object, target: task.target },
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
      format: "nanovla.dataset.v1",
      robot: spec.id,
      capture_hz: CAPTURE_HZ,
      created_at: new Date().toISOString(),
      episodes: dataset,
    },
    `${spec.id}_dataset_${dataset.length}ep_${Date.now()}.json`,
  );
  statusElement.textContent = `DATASET DOWNLOADED — ${dataset.length} EPISODES`;
}

function pick<T>(items: T[]) {
  return items[Math.floor(Math.random() * items.length)];
}

/**
 * The instruction has to be the only thing that says which cube and which zone, otherwise the
 * policy can score perfectly while ignoring the language entirely. Three cubes and two zones
 * give six distinct tasks that share the same visual scene.
 */
function rollTask(): Task {
  const object = pick(PROP_SPECS);
  const target = pick(ZONE_SPECS);
  return {
    object: object.id,
    target: target.id,
    instruction: `Move the ${object.label} cube to the ${target.label}.`,
  };
}

/** A spot claimed on the table, with the radius it needs kept clear around it. */
type Placement = { at: THREE.Vector2; clearance: number };

/**
 * Samples a spot in the arm's reachable annulus that clears everything placed so far. Each
 * item carries its own clearance so zones (which are wide) keep their distance from each
 * other while cubes (which are small) can sit closer together. Returns null rather than
 * falling back to a fixed spot, so a cramped layout is retried instead of silently stacking.
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

/** Two zone spots then three cube spots, or null if this attempt boxed itself in. */
function sampleLayout() {
  const zoneClearance = spec.props.zoneOuter + 0.01;
  const cubeClearance = spec.props.cube;
  const taken: Placement[] = [];
  for (let index = 0; index < zones.length + props.length; index += 1) {
    const placement = samplePlacement(taken, index < zones.length ? zoneClearance : cubeClearance);
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
  zones.forEach((zone, index) => {
    zone.mesh.position.set(spots[index].x, 0.001, spots[index].y);
  });
  props.forEach((prop, index) => {
    const spot = spots[zones.length + index];
    prop.body.setTranslation({ x: spot.x, y: spec.props.cube / 2, z: spot.y }, true);
    prop.body.setRotation(
      new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.random() * Math.PI * 2),
      true,
    );
    prop.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    prop.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  });

  syncPropMeshes();
  task = rollTask();
  instructionElement.value = task.instruction;
  frames = [];
  replayButton.disabled = true;
  downloadButton.disabled = true;
  updateStats(0);
  resetPose();
  statusElement.textContent = "NEW TASK — adjust a joint, then start recording";
}

/**
 * The task counts as done when the named cube is at rest inside the named zone and no longer
 * held. This is the success signal an evaluation run would score against.
 */
function taskSucceeded() {
  const prop = props.find((entry) => entry.id === task.object)!;
  const zone = zones.find((entry) => entry.id === task.target)!;
  if (heldProp === prop) return false;
  if (Math.abs(prop.mesh.position.y - spec.props.cube / 2) > 0.02) return false;
  const velocity = prop.body.linvel();
  if (Math.hypot(velocity.x, velocity.y, velocity.z) > 0.03) return false;
  return (
    Math.hypot(prop.mesh.position.x - zone.mesh.position.x, prop.mesh.position.z - zone.mesh.position.z) <=
    spec.props.zoneInner
  );
}

function updateTaskState() {
  const done = taskSucceeded();
  taskStateElement.textContent = done ? "IN ZONE" : "NOT PLACED";
  taskStateElement.classList.toggle("is-success", done);
}

/** One leg of the scripted trajectory: ease to `pose` over `duration` seconds. */
type ScriptStep = { pose: JointValues; duration: number };
type Script = { steps: ScriptStep[]; start: JointValues; total: number };

// Demonstration speed, not the arm's limit. At 5 Hz capture this puts roughly 12 degrees
// between consecutive actions; raise CAPTURE_HZ if a policy needs finer steps than that.
const ARM_SPEED = 60; // deg/s
const GRIPPER_SPEED = 120; // percent/s
const GRIPPER_OPEN = 100;
const GRIPPER_SHUT = 45;

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
function buildScript(): Script | null {
  if (!spec.solve) return null;
  const prop = props.find((entry) => entry.id === task.object)!;
  const zone = zones.find((entry) => entry.id === task.target)!;

  const hover = THREE.MathUtils.lerp(0.07, 0.13, Math.random());
  /** Aim error, so a batch is not one motion repeated. Widen it to breed failures into the set. */
  const jitter = (spread: number) => (Math.random() - 0.5) * spread;
  const resting = spec.props.cube / 2;

  const cubePose = prop.body.translation();
  const cubeYaw = new THREE.Euler().setFromQuaternion(prop.mesh.quaternion, "YXZ").y;
  const placeYaw = Math.random() * Math.PI;

  const grasp = new THREE.Vector3(cubePose.x + jitter(0.008), resting, cubePose.z + jitter(0.008));
  const above = grasp.clone().setY(resting + hover);
  const overZone = new THREE.Vector3(
    zone.mesh.position.x + jitter(0.04),
    resting + hover,
    zone.mesh.position.z + jitter(0.04),
  );
  const drop = overZone.clone().setY(resting + THREE.MathUtils.lerp(0.008, 0.04, Math.random()));

  // Height costs pitch: straight down the jaw only clears 57 mm at the far edge of the
  // workspace, but 199 mm at 70°. So prefer a vertical grasp and tilt only as much as the
  // reach demands, sweeping down from a random steep start until every waypoint solves.
  const firstPitch = THREE.MathUtils.lerp(82, 90, Math.random());
  const plan = ((): Array<{ pose: JointValues; hold?: number }> | null => {
    for (let pitchDeg = firstPitch; pitchDeg >= 60; pitchDeg -= 4) {
      const pitch = revolute(pitchDeg);
      const at = (point: THREE.Vector3, azimuth: number, gripper: number) => {
        const solution = spec.solve!(spec.mount, point, pitch, azimuth);
        return solution ? { ...solution, gripper } : null;
      };
      // hold: repeat a pose so the attach registers, and so a dropped cube lands before the cut
      const attempt: Array<{ pose: JointValues | null; hold?: number }> = [
        { pose: at(above, cubeYaw, GRIPPER_OPEN) },
        { pose: at(grasp, cubeYaw, GRIPPER_OPEN) },
        { pose: at(grasp, cubeYaw, GRIPPER_SHUT) },
        { pose: at(grasp, cubeYaw, GRIPPER_SHUT), hold: 0.3 },
        { pose: at(above, cubeYaw, GRIPPER_SHUT) },
        { pose: at(overZone, placeYaw, GRIPPER_SHUT) },
        { pose: at(drop, placeYaw, GRIPPER_SHUT) },
        { pose: at(drop, placeYaw, GRIPPER_OPEN) },
        { pose: at(drop, placeYaw, GRIPPER_OPEN), hold: 0.45 },
        { pose: at(overZone, placeYaw, GRIPPER_OPEN) },
      ];
      if (attempt.every((entry) => entry.pose !== null)) {
        return attempt as Array<{ pose: JointValues; hold?: number }>;
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
        format: "nanovla.dump.v1",
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
    const plan = buildScript();
    if (plan) {
      script = { plan, time: 0, tick: 0 };
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
  updateGrasp(GENERATION_DT);
  stepPhysics(GENERATION_DT);

  if (script.tick % TICKS_PER_CAPTURE === 0) captureFrame(script.time);
  script.tick += 1;
  script.time += GENERATION_DT;
  return script.time < plan.total;
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
      const blend = 1 - Math.exp(-9 * deltaSeconds);
      for (const joint of JOINTS) {
        currentValues[joint.name] = THREE.MathUtils.lerp(
          currentValues[joint.name],
          targetValues[joint.name],
          blend,
        );
      }
    }
    setRobotPose(currentValues);
    updateGrasp(deltaSeconds);
    stepPhysics(deltaSeconds);
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
