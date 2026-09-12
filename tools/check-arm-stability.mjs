// Headless regression check using the application's actual arm construction and controller.
// Run with Node 24: node tools/check-arm-stability.mjs
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { URDFRobot, URDFLink, URDFJoint } from 'urdf-loader/src/URDFClasses.js';

await RAPIER.init();
const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const xml = readFileSync(new URL('../public/assets/robots/a1z/A1Z_G1Z.urdf', import.meta.url), 'utf8');
const attribute = (text, key) => text.match(new RegExp(`\\b${key}="([^"]*)"`))?.[1];
const tag = (text, name) => text.match(new RegExp(`<${name}\\b[^>]*>`))?.[0] ?? '';
const vector = (text, key) => (attribute(text, key) ?? '0 0 0').split(/\s+/).map(Number);
const robot = new URDFRobot();
robot.links = {};
robot.joints = {};
const stl = new STLLoader();
for (const match of xml.matchAll(/<link\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/link>/g)) {
  const [, name, body] = match;
  const link = new URDFLink();
  link.urdfName = name;
  const inertial = body.match(/<inertial>([\s\S]*?)<\/inertial>/)[1];
  const origin = tag(inertial, 'origin');
  link.inertial = {
    mass: Number(attribute(tag(inertial, 'mass'), 'value')),
    origin: { xyz: vector(origin, 'xyz'), rpy: vector(origin, 'rpy') },
    inertia: Object.fromEntries(['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz'].map(key =>
      [key, Number(attribute(tag(inertial, 'inertia'), key))])),
  };
  const filename = attribute(tag(body, 'mesh'), 'filename').replace('package://A1Z_G1Z/', '');
  const bytes = readFileSync(new URL(`../public/assets/robots/a1z/${filename}`, import.meta.url));
  link.add(new THREE.Mesh(stl.parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))));
  robot.links[name] = link;
}
for (const match of xml.matchAll(/<joint\s+name="([^"]+)"\s+type="([^"]+)"[^>]*>([\s\S]*?)<\/joint>/g)) {
  const [, name, type, body] = match;
  const joint = new URDFJoint();
  joint.urdfName = name;
  joint.jointType = type;
  joint.position.fromArray(vector(tag(body, 'origin'), 'xyz'));
  const [r, p, y] = vector(tag(body, 'origin'), 'rpy');
  joint.quaternion.setFromEuler(new THREE.Euler(r, p, y, 'ZYX'));
  joint.axis = new THREE.Vector3().fromArray(vector(tag(body, 'axis'), 'xyz'));
  joint.limit = Object.fromEntries(['lower', 'upper', 'effort', 'velocity'].map(key =>
    [key, Number(attribute(tag(body, 'limit'), key))]));
  robot.links[attribute(tag(body, 'parent'), 'link')].add(joint);
  joint.add(robot.links[attribute(tag(body, 'child'), 'link')]);
  robot.joints[name] = joint;
}
robot.add(robot.links.base_link);
robot.rotation.x = -Math.PI / 2;

const physics = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
physics.timestep = 1 / 480;
physics.createCollider(RAPIER.ColliderDesc.cuboid(1, 0.05, 0.4736).setTranslation(0, -0.05, 0).setFriction(0.9));
const specs = source.slice(source.indexOf('type JointValues'), source.indexOf('const ROBOTS ='));
const dynamics = source.slice(source.indexOf('const ROOT_LINK ='), source.indexOf('function buildJointControls()'));
const bootstrap = `
const spec = A1Z;
robot.position.set(...spec.mount);
robot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), spec.heading ?? 0));
const JOINTS = spec.joints;
const initialValues = Object.fromEntries(JOINTS.map(j => [j.name, j.initial]));
const measuredValues = {...initialValues};
const GROUP_ARM = 1, GROUP_PROP = 2, GROUP_WORLD = 4;
const collisionGroups = (member, filter) => (member << 16) | filter;
let physicsAccumulator = 0, gripped = null, grippedReport = null;
const statusElement = {textContent: ''};
`;
const api = new Function('THREE', 'RAPIER', 'robot', 'physics', stripTypeScriptTypes(
  specs + bootstrap + dynamics, { mode: 'transform' }) + `
  return { initialValues, resetArmPhysics, commandArm, applyGravityCompensation,
    syncArmFromPhysics, measuredValues, armJoints, linkMassProperties };
`)(THREE, RAPIER, robot, physics);
assert.equal(api.armJoints.size, 8);

// The imported principal moments/frame must reconstruct the complete URDF tensor.
for (const link of Object.values(robot.links)) {
  const { principal, frame } = api.linkMassProperties(link);
  const rotation = new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(frame));
  const tensor = rotation.clone().multiply(new THREE.Matrix3().set(principal.x, 0, 0, 0, principal.y, 0, 0, 0, principal.z))
    .multiply(rotation.clone().transpose());
  const i = link.inertial.inertia;
  const expected = new THREE.Matrix3().set(i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz);
  assert.ok(tensor.elements.every((x, index) => Math.abs(x - expected.elements[index]) < 1e-9), link.urdfName);
}

function run(label, target, seconds = 8) {
  api.commandArm(target);
  const low = {}, high = {};
  let maxSpeed = 0;
  for (let tick = 0; tick < seconds * 480; tick++) {
    api.applyGravityCompensation();
    physics.step();
    api.syncArmFromPhysics();
    for (const [name, entry] of api.armJoints) {
      const velocity = entry.child.angvel();
      assert.ok(Number.isFinite(velocity.x + velocity.y + velocity.z), `${label}: ${name} finite`);
      if (tick >= (seconds - 2) * 480 && !entry.prismatic) {
        const value = api.measuredValues[name];
        low[name] = Math.min(low[name] ?? Infinity, value);
        high[name] = Math.max(high[name] ?? -Infinity, value);
        maxSpeed = Math.max(maxSpeed, Math.hypot(velocity.x, velocity.y, velocity.z));
      }
    }
  }
  const swing = Math.max(...Object.keys(low).map(name => high[name] - low[name]));
  const error = Math.max(...Object.keys(low).map(name => Math.abs(api.measuredValues[name] - target[name])));
  console.log(JSON.stringify({label, swingDeg: swing, maxSpeedRadS: maxSpeed, errorDeg: error,
    errors: Object.fromEntries(Object.keys(low).map(name => [name, api.measuredValues[name] - target[name]]))}));
  assert.ok(swing < 0.5, `${label}: persistent oscillation ${swing} deg`);
  assert.ok(error < 3, `${label}: holding error ${error} deg`);
}
api.resetArmPhysics(api.initialValues);
run('startup hold', api.initialValues);
const moved = {...api.initialValues, arm_joint1: 20, arm_joint2: 110, arm_joint3: -110, arm_joint4: 45};
run('joint move and settle', moved, 12);
api.resetArmPhysics(api.initialValues);
api.syncArmFromPhysics();
assert.ok(Object.keys(moved).every(name => Math.abs(api.measuredValues[name] - api.initialValues[name]) < 0.001));
run('reset hold', api.initialValues);
physics.free();
