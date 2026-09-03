import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import URDFLoader, { type URDFRobot } from "urdf-loader";
import "./style.css";

type JointName =
  | "shoulder_pan"
  | "shoulder_lift"
  | "elbow_flex"
  | "wrist_flex"
  | "wrist_roll"
  | "gripper";

type JointValues = Record<JointName, number>;

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

const JOINTS: Array<{
  name: JointName;
  label: string;
  min: number;
  max: number;
  initial: number;
}> = [
  { name: "shoulder_pan", label: "Shoulder pan", min: -110, max: 110, initial: 0 },
  { name: "shoulder_lift", label: "Shoulder lift", min: -100, max: 100, initial: -28 },
  { name: "elbow_flex", label: "Elbow flex", min: -97, max: 97, initial: 62 },
  { name: "wrist_flex", label: "Wrist flex", min: -95, max: 95, initial: -32 },
  { name: "wrist_roll", label: "Wrist roll", min: -157, max: 163, initial: 0 },
  { name: "gripper", label: "Gripper", min: 0, max: 100, initial: 70 },
];

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

const initialValues = Object.fromEntries(JOINTS.map((joint) => [joint.name, joint.initial])) as JointValues;
const currentValues = { ...initialValues };
const targetValues = { ...initialValues };
const sliders = new Map<JointName, HTMLInputElement>();
const valueLabels = new Map<JointName, HTMLElement>();

let frames: EpisodeFrame[] = [];
let recording = false;
let recordStart = 0;
let lastSample = 0;
let replaying = false;
let replayStart = 0;
let replayIndex = 0;

const scene = new THREE.Scene();
scene.background = new THREE.Color("#cbc5b8");
scene.fog = new THREE.Fog("#cbc5b8", 2.2, 4.5);

const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
camera.position.set(0.82, 0.58, 0.84);
camera.lookAt(0, 0.18, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
sceneElement.appendChild(renderer.domElement);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.target.set(0, 0.18, 0);
orbitControls.enableDamping = true;
orbitControls.minDistance = 0.7;
orbitControls.maxDistance = 3;
orbitControls.maxPolarAngle = Math.PI * 0.49;

const observationCamera = new THREE.PerspectiveCamera(46, 4 / 3, 0.01, 10);
observationCamera.position.set(0.68, 0.5, 0.68);
observationCamera.lookAt(0, 0.16, 0);

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
keyLight.shadow.camera.left = -1.5;
keyLight.shadow.camera.right = 1.5;
keyLight.shadow.camera.top = 1.5;
keyLight.shadow.camera.bottom = -1.5;
scene.add(keyLight);

const tableMaterial = new THREE.MeshStandardMaterial({ color: 0xd7c8ac, roughness: 0.86 });
const tableTop = new THREE.Mesh(new THREE.BoxGeometry(0.92, 0.07, 0.62), tableMaterial);
tableTop.position.y = -0.035;
tableTop.receiveShadow = true;
scene.add(tableTop);

for (const x of [-0.38, 0.38]) {
  for (const z of [-0.23, 0.23]) {
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
cube.position.set(0.19, 0.05, 0.07);
cube.castShadow = true;
cube.receiveShadow = true;
scene.add(cube);

const targetRing = new THREE.Mesh(
  new THREE.RingGeometry(0.07, 0.085, 32),
  new THREE.MeshBasicMaterial({ color: 0x315e87, side: THREE.DoubleSide }),
);
targetRing.position.set(0.22, 0.001, -0.17);
targetRing.rotation.x = -Math.PI / 2;
scene.add(targetRing);

const urdfLoader = new URDFLoader();
urdfLoader.parseCollision = false;
statusElement.textContent = "LOADING SO-101 URDF";
const robot = await urdfLoader.loadAsync("/assets/robots/so101/so101_new_calib.urdf");
robot.rotation.x = -Math.PI / 2;
robot.position.set(-0.2, 0.001, 0);
robot.traverse((object) => {
  if (object instanceof THREE.Mesh) {
    object.castShadow = true;
    object.receiveShadow = true;
  }
});
scene.add(robot);

function setURDFJoint(robotModel: URDFRobot, name: JointName, value: number) {
  if (name === "gripper") {
    const gripperRadians = THREE.MathUtils.degToRad(THREE.MathUtils.lerp(-10, 100, value / 100));
    robotModel.setJointValue(name, gripperRadians);
    return;
  }
  robotModel.setJointValue(name, THREE.MathUtils.degToRad(value));
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

function formatJointValue(name: JointName, value: number) {
  return name === "gripper" ? `${Math.round(value)}%` : `${Math.round(value)}°`;
}

function setRobotPose(values: JointValues) {
  for (const joint of JOINTS) {
    setURDFJoint(robot, joint.name, values[joint.name]);
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

function captureFrame(now: number) {
  observationRenderer.render(scene, observationCamera);
  const image = observationRenderer.domElement.toDataURL("image/jpeg", 0.72);
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
    robot: "so101",
    instruction: instructionElement.value.trim(),
    capture_hz: CAPTURE_HZ,
    created_at: new Date().toISOString(),
    frames,
  };
  const blob = new Blob([JSON.stringify(episode)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `so101_episode_${Date.now()}.json`;
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
observationRenderer.render(scene, observationCamera);
previewElement.src = observationRenderer.domElement.toDataURL("image/jpeg", 0.72);
renderer.setAnimationLoop(animate);

recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
replayButton.addEventListener("click", replayEpisode);
downloadButton.addEventListener("click", downloadEpisode);
resetButton.addEventListener("click", resetPose);
window.addEventListener("resize", resizeRenderer);
