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
  };
  action: {
    joint_targets: JointValues;
  };
};

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
  joints: JointSpec[];
};

const revolute = (value: number) => THREE.MathUtils.degToRad(value);

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
      initial: 60,
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

robotBadgeElement.textContent = `${spec.label} / SIM`;

const initialValues = Object.fromEntries(JOINTS.map((joint) => [joint.name, joint.initial])) as JointValues;
const currentValues = { ...initialValues };
const targetValues = { ...initialValues };
const sliders = new Map<string, HTMLInputElement>();
const valueLabels = new Map<string, HTMLElement>();

let frames: EpisodeFrame[] = [];
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

const observationCamera = new THREE.PerspectiveCamera(46, 4 / 3, 0.01, 10);
observationCamera.position.set(...spec.view.obsCamera);
observationCamera.lookAt(...spec.view.obsTarget);

const observationRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
observationRenderer.setSize(256, 192, false);
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
const tableTop = new THREE.Mesh(new THREE.BoxGeometry(1.18, 0.07, 0.82), tableMaterial);
tableTop.position.y = -0.035;
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

const cube = new THREE.Mesh(
  new THREE.BoxGeometry(0.1, 0.1, 0.1),
  new THREE.MeshStandardMaterial({ color: 0xdb4f31, roughness: 0.72 }),
);
cube.position.set(0.05, 0.05, 0.1);
cube.castShadow = true;
cube.receiveShadow = true;
scene.add(cube);

const targetRing = new THREE.Mesh(
  new THREE.RingGeometry(0.07, 0.085, 32),
  new THREE.MeshBasicMaterial({ color: 0x315e87, side: THREE.DoubleSide }),
);
targetRing.position.set(0, 0.001, -0.21);
targetRing.rotation.x = -Math.PI / 2;
scene.add(targetRing);

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

function captureFrame(now: number) {
  const image = renderObservation();
  frames.push({
    timestamp: Number(((now - recordStart) / 1000).toFixed(3)),
    observation: {
      image,
      joint_positions: cloneJointValues(currentValues),
    },
    action: {
      joint_targets: cloneJointValues(targetValues),
    },
  });
  previewElement.src = image;
  updateStats(now - recordStart);
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

function downloadEpisode() {
  const episode = {
    format: "nanovla.episode.v1",
    robot: spec.id,
    instruction: instructionElement.value.trim(),
    capture_hz: CAPTURE_HZ,
    created_at: new Date().toISOString(),
    frames,
  };
  const blob = new Blob([JSON.stringify(episode)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${spec.id}_episode_${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);
  statusElement.textContent = "EPISODE DOWNLOADED";
}

function resetPose() {
  Object.assign(currentValues, initialValues);
  Object.assign(targetValues, initialValues);
  updateJointUI(initialValues);
  setRobotPose(initialValues);
  statusElement.textContent = "POSE RESET";
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
  orbitControls.update();
  renderer.render(scene, camera);

  if (recording && now - lastSample >= SAMPLE_INTERVAL) {
    captureFrame(now);
    lastSample = now;
  }
}

buildJointControls();
resetPose();
resizeRenderer();
previewElement.src = renderObservation();
renderer.setAnimationLoop(animate);

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
replayButton.addEventListener("click", replayEpisode);
downloadButton.addEventListener("click", downloadEpisode);
resetButton.addEventListener("click", resetPose);
window.addEventListener("resize", resizeRenderer);
