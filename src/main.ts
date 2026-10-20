import RAPIER from "@dimforge/rapier3d-compat";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import URDFLoader, { type URDFLink, type URDFRobot } from "urdf-loader";
import { ConstrainedIK } from "./constrained-ik";
import { bothFingersGrip, manifoldHasContact } from "./contact-state";
import "./style.css";

type JointValues = Record<string, number>;

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
const OBSERVATION_WIDTH = 640;
const OBSERVATION_HEIGHT = 480;

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

const resetButton = document.querySelector<HTMLButtonElement>("#reset-button")!;
const newSceneButton = document.querySelector<HTMLButtonElement>("#new-scene-button")!;
const jawOpenButton = document.querySelector<HTMLButtonElement>("#jaw-open-button")!;
const jawCloseButton = document.querySelector<HTMLButtonElement>("#jaw-close-button")!;
const statusElement = document.querySelector<HTMLElement>("#status-message")!;
const robotBadgeElement = document.querySelector<HTMLElement>("#robot-badge")!;
robotBadgeElement.textContent = `${spec.label} / SIM`;

const initialValues = Object.fromEntries(JOINTS.map((joint) => [joint.name, joint.initial])) as JointValues;
/** What the arm is told to do. With dynamics on, what it actually does is `measuredValues`. */
const currentValues = { ...initialValues };
const targetValues = { ...initialValues };
const measuredValues = { ...initialValues };

let task: Task = {
  target: "red",
  region: { shape: "circle", color: "yellow", center: [0, 0], radius: 0.075 },
  instruction: "",
};

const scene = new THREE.Scene();

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
const observationPmrem = new THREE.PMREMGenerator(observationRenderer);
observationPmrem.compileEquirectangularShader();
const rgbeLoader = new RGBELoader();

const backdrops = await Promise.all(
  BACKDROPS.map(async (name) => {
    const texture = await rgbeLoader.loadAsync(`/assets/hdri/${name}.hdr`);
    texture.mapping = THREE.EquirectangularReflectionMapping;
    return {
      background: texture,
      environment: observationPmrem.fromEquirectangular(texture).texture,
    };
  }),
);

function useBackdrop(index: number) {
  const backdrop = backdrops[index];
  scene.background = backdrop.background;
  scene.environment = backdrop.environment;
}

/** Every picture in the interface is drawn here, so they are all lit identically. */
function renderObservationView(view: THREE.Camera) {
  observationRenderer.render(scene, view);
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

type SensorView = "wrist" | "overhead";
const sensorCameras = new Map<SensorView, THREE.PerspectiveCamera>();
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
  statusElement.textContent = "NEW SCENE — drag a joint, or click the side and top views";
  return true;
}

function resetPose() {
  Object.assign(currentValues, initialValues);
  Object.assign(targetValues, initialValues);
  resetArmPhysics(initialValues);
  setRobotPose(initialValues);
  statusElement.textContent = "POSE RESET";
}

/**
 * Two axis-aligned orthographic views that together name a point in space, plus the centre
 * camera for context.
 *
 * Orthographic projection is what makes two clicks enough: every ray through a pixel is parallel
 * to the view axis, so the coordinates read across the image do not depend on how far away the
 * surface under the cursor is. The top view therefore fixes X and Z, the side view fixes Y, and
 * no depth has to be guessed. The centre camera is deliberately not clickable; it is there to
 * say which cube is which, a question the flat grey orthographic views answer poorly.
 *
 * All three are drawn through `renderObservationView`, so they are lit exactly like the main
 * scene. A second WebGLRenderer cannot be used here: the environment map is a PMREM texture
 * belonging to one WebGL context, and sampling it from another silently drops the lighting.
 */
const PICK_ASPECT = OBSERVATION_WIDTH / OBSERVATION_HEIGHT;
/** Room around the workspace for the arm itself, and for picking just outside it. */
const PICK_MARGIN = 0.07;
/** Table to a little above the highest useful approach. */
const PICK_FLOOR = -0.03;
const PICK_CEILING = 0.40;

/**
 * The patch of table the arm can actually serve.
 *
 * Framing on the mount instead wastes most of both pictures: the reachable annulus is swept on
 * one side of the base, so the base sits at a corner of the useful area rather than its middle.
 * This walks the same annulus `samplePlacement` draws from, so the views follow whatever reach
 * and heading a robot declares instead of hard-coded numbers.
 */
const PICK_AREA = (() => {
  const { min, max, yaw } = spec.reach;
  const heading = spec.heading ?? 0;
  const box = new THREE.Box2();
  for (let step = 0; step <= 12; step += 1) {
    const angle = THREE.MathUtils.lerp(-yaw, 0, step / 12);
    for (const radius of [min, max]) {
      const offset = new THREE.Vector3(radius * Math.cos(angle), 0, -radius * Math.sin(angle))
        .applyAxisAngle(new THREE.Vector3(0, 1, 0), heading);
      box.expandByPoint(new THREE.Vector2(spec.mount[0] + offset.x, spec.mount[2] + offset.z));
    }
  }
  box.expandByScalar(PICK_MARGIN);
  return box;
})();

/** Smallest frame of the view's own aspect that holds `across` by `down`, centred on the area. */
function orthoView(
  aimedAt: THREE.Vector3, from: THREE.Vector3, up: THREE.Vector3, across: number, down: number,
) {
  const halfHeight = Math.max(down, across / PICK_ASPECT) / 2;
  const halfWidth = halfHeight * PICK_ASPECT;
  const view = new THREE.OrthographicCamera(
    -halfWidth, halfWidth, halfHeight, -halfHeight, 0.01, 12,
  );
  view.position.copy(from);
  view.up.copy(up);
  view.lookAt(aimedAt);
  view.updateMatrixWorld(true);
  return view;
}

const pickCentre = PICK_AREA.getCenter(new THREE.Vector2());
const pickSize = PICK_AREA.getSize(new THREE.Vector2());
const PICK_EYE_HEIGHT = (PICK_FLOOR + PICK_CEILING) / 2;

/** Looks along -Z, so screen right is +X and screen up is +Y. */
const sideCamera = orthoView(
  new THREE.Vector3(pickCentre.x, PICK_EYE_HEIGHT, pickCentre.y),
  new THREE.Vector3(pickCentre.x, PICK_EYE_HEIGHT, pickCentre.y + 3),
  new THREE.Vector3(0, 1, 0),
  pickSize.x,
  PICK_CEILING - PICK_FLOOR,
);
/**
 * Looks straight down. The up vector is -Z, not +Z: a camera's right axis is `up x forward`, so
 * +Z here would point screen-right at -X and mirror this view against the side view. -Z also
 * puts "away from the side view's eye" at the top of the picture, so the two views agree on
 * which way the scene faces.
 */
const topCamera = orthoView(
  new THREE.Vector3(pickCentre.x, 0, pickCentre.y),
  new THREE.Vector3(pickCentre.x, 3, pickCentre.y),
  new THREE.Vector3(0, 0, -1),
  pickSize.x,
  pickSize.y,
);

const sideCanvas = document.querySelector<HTMLCanvasElement>("#view-side")!;
const topCanvas = document.querySelector<HTMLCanvasElement>("#view-top")!;
const contextCanvas = document.querySelector<HTMLCanvasElement>("#view-context")!;
const pickReadout = {
  x: document.querySelector<HTMLElement>("#pick-x")!,
  y: document.querySelector<HTMLElement>("#pick-y")!,
  z: document.querySelector<HTMLElement>("#pick-z")!,
};
const pickMoveButton = document.querySelector<HTMLButtonElement>("#pick-move-button")!;
const pickClearButton = document.querySelector<HTMLButtonElement>("#pick-clear-button")!;
const pickStateElement = document.querySelector<HTMLElement>("#pick-state")!;

const pickMarker = new THREE.Group();
const pickDot = new THREE.Mesh(
  new THREE.SphereGeometry(0.008, 16, 12),
  new THREE.MeshBasicMaterial({ color: 0xec542f, toneMapped: false, depthTest: false }),
);
pickDot.renderOrder = 1001;
pickMarker.add(pickDot);
for (const axis of [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]) {
  const arm = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-axis[0], -axis[1], -axis[2]),
      new THREE.Vector3(axis[0], axis[1], axis[2]),
    ]),
    new THREE.LineBasicMaterial({ color: 0xec542f, toneMapped: false, depthTest: false }),
  );
  arm.renderOrder = 1001;
  pickMarker.add(arm);
}
pickMarker.visible = false;
scene.add(pickMarker);

/** What has been picked so far. Y comes from the side view, X and Z from the top view. */
let pickHeight: number | null = null;
let pickGround: { x: number; z: number } | null = null;
let lastPickRender = -Infinity;

function pickedPoint() {
  if (pickHeight === null || pickGround === null) return null;
  return new THREE.Vector3(pickGround.x, pickHeight, pickGround.z);
}

/** Pixel to world, through the clicked view's own projection. */
function unprojectClick(event: MouseEvent, canvas: HTMLCanvasElement, view: THREE.Camera) {
  const bounds = canvas.getBoundingClientRect();
  return new THREE.Vector3(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
    -1,
  ).unproject(view);
}

function setPickState(message: string, kind: "idle" | "ok" | "error" = "idle") {
  pickStateElement.textContent = message;
  pickStateElement.classList.toggle("is-success", kind === "ok");
  pickStateElement.classList.toggle("is-error", kind === "error");
}

function updatePickUI() {
  const point = pickedPoint();
  pickReadout.x.textContent = pickGround ? `${pickGround.x.toFixed(3)} m` : "—";
  pickReadout.y.textContent = pickHeight === null ? "—" : `${pickHeight.toFixed(3)} m`;
  pickReadout.z.textContent = pickGround ? `${pickGround.z.toFixed(3)} m` : "—";
  pickMoveButton.disabled = point === null;
  pickClearButton.disabled = pickHeight === null && pickGround === null;
  pickMarker.visible = point !== null;
  if (point) pickMarker.position.copy(point);
}

function clearPick() {
  pickHeight = null;
  pickGround = null;
  updatePickUI();
  setPickState("Click the top view for the floor position, the side view for the height.");
}

/** The picker only places the wrist, so the jaws are driven separately. */
function setJaws(open: boolean) {
  const jaw = JOINTS.find((joint) => joint.name === "gripper");
  if (!jaw) return;
  targetValues[jaw.name] = open ? jaw.max : jaw.min;
  setPickState(open ? "Jaws opening." : "Jaws closing.");
}

/**
 * Sends the gripper midpoint to the picked point.
 *
 * A straight-down approach is tried first because that is how the jaws reach a table. Shallower
 * pitches are fallbacks for points the wrist cannot stand over, and the angle that worked is
 * reported so an awkward solution is visible rather than silent.
 */
function moveGripperToPick() {
  const target = pickedPoint();
  if (!target) return;
  if (!spec.solve || !constrainedIK) {
    setPickState("This robot has no IK solver.", "error");
    return;
  }
  for (const degrees of [90, 75, 60, 45, 30]) {
    const solution = spec.solve(spec.mount, target, THREE.MathUtils.degToRad(degrees), 0, currentValues);
    if (!solution) continue;
    for (const joint of JOINTS) {
      // The jaw opening belongs to the operator; IK only places the wrist.
      if (joint.name === "gripper") continue;
      const value = solution[joint.name];
      if (value !== undefined) {
        targetValues[joint.name] = THREE.MathUtils.clamp(value, joint.min, joint.max);
      }
    }
    const residual = constrainedIK.forwardPose(solution).position.distanceTo(target) * 1000;
    setPickState(`Moving: ${degrees}° approach, ${residual.toFixed(1)} mm residual.`, "ok");
    statusElement.textContent =
      `GRIPPER TO ${target.x.toFixed(3)}, ${target.y.toFixed(3)}, ${target.z.toFixed(3)}`;
    return;
  }
  setPickState("Out of reach: no approach angle solves for that point.", "error");
}

sideCanvas.addEventListener("click", (event) => {
  // This view looks along -Z, so its ray is horizontal and only the height it crosses is new.
  pickHeight = unprojectClick(event, sideCanvas, sideCamera).y;
  updatePickUI();
  setPickState(pickGround
    ? "Point set. Move the gripper to it."
    : "Height set. Now click the floor position in the top view.");
});

topCanvas.addEventListener("click", (event) => {
  // This view looks straight down, so its ray is vertical and fixes the floor position.
  const at = unprojectClick(event, topCanvas, topCamera);
  pickGround = { x: at.x, z: at.z };
  updatePickUI();
  setPickState(pickHeight === null
    ? "Floor position set. Now click the height in the side view."
    : "Point set. Move the gripper to it.");
});

pickMoveButton.addEventListener("click", moveGripperToPick);
pickClearButton.addEventListener("click", clearPick);

/** Marks where the point sits in a view, so the two clicks can be checked against each other. */
function drawCross(canvas: HTMLCanvasElement, view: THREE.Camera, point: THREE.Vector3) {
  const at = point.clone().project(view);
  if (at.z < -1 || at.z > 1) return;
  const x = (at.x + 1) / 2 * canvas.width;
  const y = (1 - at.y) / 2 * canvas.height;
  const context = canvas.getContext("2d")!;
  context.save();
  context.strokeStyle = "#ec542f";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(x - 10, y); context.lineTo(x + 10, y);
  context.moveTo(x, y - 10); context.lineTo(x, y + 10);
  context.stroke();
  context.beginPath();
  context.arc(x, y, 6, 0, Math.PI * 2);
  context.stroke();
  context.restore();
}

/** Redrawn with the sensor previews; these views are for aiming, not for frame rate. */
function renderPickViews(now: number) {
  if (now - lastPickRender < 100) return;
  lastPickRender = now;
  const point = pickedPoint();
  for (const pair of [[sideCanvas, sideCamera], [topCanvas, topCamera]] as const) {
    renderObservationView(pair[1]);
    pair[0].getContext("2d")!.drawImage(observationRenderer.domElement, 0, 0, pair[0].width, pair[0].height);
    if (point) drawCross(pair[0], pair[1], point);
  }
  // Context only: the centre camera says which cube is which, the grey views do not.
  const centre = sensorCameras.get("overhead");
  if (centre) {
    renderObservationView(centre);
    contextCanvas.getContext("2d")!.drawImage(
      observationRenderer.domElement, 0, 0, contextCanvas.width, contextCanvas.height,
    );
    if (point) drawCross(contextCanvas, centre, point);
  }
}

let previousTime = performance.now();
function animate(now: number) {
  const deltaSeconds = Math.min((now - previousTime) / 1000, 0.1);
  previousTime = now;

  // Sliders and the view picker both command the motors through `targetValues`. The lag
  // between command and pose used to be a hand-rolled exponential blend; it now comes from
  // the arm's own inertia.
  const blend = armJoints ? 1 : 1 - Math.exp(-9 * deltaSeconds);
  for (const joint of JOINTS) {
    currentValues[joint.name] = THREE.MathUtils.lerp(
      currentValues[joint.name],
      targetValues[joint.name],
      blend,
    );
  }
  setRobotPose(currentValues);
  stepPhysics(deltaSeconds);
  syncArmFromPhysics();
  updateGrasp();
  // Mounted cameras need their parent link transforms updated before rendering.
  scene.updateMatrixWorld(true);
  renderPickViews(now);
  requestAnimationFrame(animate);
}

setupSensorCameras();
useBackdrop(0);
newEpisode();
stepPhysics(PHYSICS_STEP);
requestAnimationFrame(animate);

resetButton.addEventListener("click", resetPose);
newSceneButton.addEventListener("click", () => { newEpisode(); clearPick(); });
jawOpenButton.addEventListener("click", () => setJaws(true));
jawCloseButton.addEventListener("click", () => setJaws(false));
