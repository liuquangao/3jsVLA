import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { stripTypeScriptTypes } from 'node:module';
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { URDFRobot, URDFLink, URDFJoint } from 'urdf-loader/src/URDFClasses.js';

await RAPIER.init();
const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const ikSource = readFileSync(new URL('../src/constrained-ik.ts', import.meta.url), 'utf8');
const contactSource = readFileSync(new URL('../src/contact-state.ts', import.meta.url), 'utf8');
const manifoldHasContact = new Function(stripTypeScriptTypes(contactSource.replace(/export /g,''),{mode:'transform'})+';return manifoldHasContact;')();
const ConstrainedIK = new Function('THREE', stripTypeScriptTypes(ikSource.replace(/^import .*;\r?\n/m, '').replace('export class', 'class'), {mode:'transform'}) + ';return ConstrainedIK;')(THREE);
const xml = readFileSync(new URL('../public/assets/robots/a1z/A1Z_G1Z.urdf', import.meta.url), 'utf8');
const attribute = (text, key) => text.match(new RegExp(`\\b${key}="([^"]*)"`))?.[1];
const tag = (text, name) => text.match(new RegExp(`<${name}\\b[^>]*>`))?.[0] ?? '';
const vector = (text, key) => (attribute(text, key) ?? '0 0 0').split(/\s+/).map(Number);
const robot = new URDFRobot();
robot.links = {}; robot.joints = {};
const stl = new STLLoader();
for (const [, name, body] of xml.matchAll(/<link\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/link>/g)) {
  const link = new URDFLink(); link.urdfName = name;
  const inertial = body.match(/<inertial>([\s\S]*?)<\/inertial>/)[1];
  const origin = tag(inertial, 'origin');
  link.inertial = { mass: Number(attribute(tag(inertial, 'mass'), 'value')),
    origin: {xyz: vector(origin, 'xyz'), rpy:vector(origin, 'rpy')},
    inertia: Object.fromEntries(['ixx','ixy','ixz','iyy','iyz','izz'].map(key => [key, Number(attribute(tag(inertial,'inertia'),key))])) };
  const filename = attribute(tag(body,'mesh'),'filename').replace('package://A1Z_G1Z/','');
  const bytes = readFileSync(new URL(`../public/assets/robots/a1z/${filename}`,import.meta.url));
  link.add(new THREE.Mesh(stl.parse(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength))));
  robot.links[name]=link;
}
for (const [,name,type,body] of xml.matchAll(/<joint\s+name="([^"]+)"\s+type="([^"]+)"[^>]*>([\s\S]*?)<\/joint>/g)) {
  const joint = new URDFJoint(); joint.urdfName=name; joint.jointType=type;
  joint.position.fromArray(vector(tag(body,'origin'),'xyz'));
  joint.quaternion.setFromEuler(new THREE.Euler(...vector(tag(body,'origin'),'rpy'),'ZYX'));
  joint.axis = new THREE.Vector3().fromArray(vector(tag(body,'axis'),'xyz'));
  joint.limit=Object.fromEntries(['lower','upper','effort','velocity'].map(key => [key,Number(attribute(tag(body,'limit'),key))]));
  robot.links[attribute(tag(body,'parent'),'link')].add(joint);
  joint.add(robot.links[attribute(tag(body,'child'),'link')]); robot.joints[name]=joint;
}
robot.add(robot.links.base_link); robot.rotation.x=-Math.PI/2;
const physics=new RAPIER.World({x:0,y:-9.81,z:0}); physics.timestep=1/480;
new Function('physics','PHYSICS_STEP',source.slice(source.indexOf('physics.timestep ='),source.indexOf('// A slab')))(physics,1/480);
const table=physics.createCollider(RAPIER.ColliderDesc.cuboid(1,.05,.4736).setTranslation(0,-.05,0).setFriction(.9));
const specs=source.slice(source.indexOf('type JointValues'),source.indexOf('const ROBOTS ='));
let dynamics=source.slice(source.indexOf('const ROOT_LINK ='),source.indexOf('function buildJointControls()'));
dynamics=dynamics.replace(/\/\/ Draw the solver's actual collision geometry[\s\S]*?(?=\/\/ Capture zero-pose)/,'');
const api=new Function('THREE','RAPIER','robot','physics','ConstrainedIK','manifoldHasContact',stripTypeScriptTypes(specs+`
const spec=A1Z, JOINTS=spec.joints, PHYSICS_STEP=1/480;
const initialValues=Object.fromEntries(JOINTS.map(j=>[j.name,j.initial]));
const currentValues={...initialValues}, measuredValues={...initialValues};
robot.position.set(...spec.mount);
robot.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,1,0),spec.heading??0));
const GROUP_ARM=1,GROUP_PROP=2,GROUP_WORLD=4;
const collisionGroups=(a,b)=>(a<<16)|b;
let physicsAccumulator=0,gripped=null,grippedReport=null;
const statusElement={textContent:''};
`+dynamics+source.slice(source.indexOf('function touching('),source.indexOf('function contactAndMotorDiagnostic(')),{mode:'transform'})+`
return {spec, initialValues,currentValues,measuredValues, resetArmPhysics,commandArm,applyGravityCompensation,syncArmFromPhysics,armJoints,armLinkBodies,fingerColliders,constrainedIK,touching};
`)(THREE,RAPIER,robot,physics,ConstrainedIK,manifoldHasContact);
const cube=physics.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(.075,.0225,-.145).setLinearDamping(.35).setAngularDamping(.55).setCcdEnabled(true));
const box=physics.createCollider(RAPIER.ColliderDesc.cuboid(.0225,.0225,.0225).setFriction(1.1).setRestitution(.05).setDensity(700).setCollisionGroups((2<<16)|7),cube);
if(process.env.NO_CONTACTS==='1') { box.setCollisionGroups(0); table.setCollisionGroups(0); }
if(process.env.NO_TRIM==='1') for(const entry of api.armJoints.values()) Object.defineProperty(entry,'trim',{get:()=>0,set:()=>{}});
api.resetArmPhysics(api.initialValues);
if(process.env.SOLVER) physics.numSolverIterations=Number(process.env.SOLVER);
if(process.env.PGS) physics.integrationParameters.numInternalPgsIterations=Number(process.env.PGS);
if(process.env.PREDICTION) physics.integrationParameters.normalizedPredictionDistance=Number(process.env.PREDICTION);
if(process.env.CONTACT_HZ) physics.integrationParameters.contact_natural_frequency=Number(process.env.CONTACT_HZ);
if(process.env.CONTACT_ERROR) physics.integrationParameters.normalizedAllowedLinearError=Number(process.env.CONTACT_ERROR);
console.log(JSON.stringify({settings:{solver:physics.numSolverIterations,pgs:physics.integrationParameters.numInternalPgsIterations,prediction:physics.integrationParameters.normalizedPredictionDistance,trim:process.env.NO_TRIM!=='1'}}));
const label = new Map([[table.handle,'table'],[box.handle,'cube']]);
for(const [name,body] of api.armLinkBodies) for(let i=0;i<body.numColliders();i++) label.set(body.collider(i).handle,name);
function snapshot(phase,t) {
  robot.updateMatrixWorld(true);
  const body=api.armLinkBodies.get('arm_link6'), p=body.translation(),q=body.rotation();
  const tcp=new THREE.Vector3(.17,0,0).applyQuaternion(new THREE.Quaternion(q.x,q.y,q.z,q.w)).add(new THREE.Vector3(p.x,p.y,p.z));
  const visual=new THREE.Vector3(.17,0,0).applyMatrix4(robot.links.arm_link6.matrixWorld);
  const contacts=[];
  for(const [name,body] of api.armLinkBodies) {
    if(name==='base_link') continue;
    for(let i=0;i<body.numColliders();i++) for(const other of [table,box]) physics.contactPair(body.collider(i),other,m=>{
      const points=Array.from({length:m.numContacts()},(_,k)=>({mm:m.contactDist(k)*1000,impulse:m.contactImpulse(k)}));
      if(points.some(p=>p.impulse>1e-8)) contacts.push({link:name,other:label.get(other.handle),freshMm:body.collider(i).contactCollider(other,.1)?.distance*1000,points});
    });
  }
  console.log(JSON.stringify({phase,t,tcp:tcp.toArray(),visualGapMm:tcp.distanceTo(visual)*1000,cube:cube.translation(),errors:Object.fromEntries(Object.keys(api.initialValues).map(n=>[n,api.measuredValues[n]-api.currentValues[n]])),trim:Object.fromEntries([...api.armJoints].filter(([,e])=>!e.prismatic).map(([n,e])=>[n,THREE.MathUtils.radToDeg(e.trim)])),contacts}));
}
let seed=api.initialValues;
const metrics={maxDescentErrorMm:0,minFingerTableGapMm:Infinity,maxGripPenetrationMm:0,openContacts:0,grippedAfterClose:false,liftMm:0};
for(const phase of ['hover','descend','close','lift']) {
  const start={...seed};
  for(let tick=0;tick<6*480;tick++) {
    const t=tick/480, alpha=THREE.MathUtils.smoothstep(Math.min(t/3,1),0,1);
    if(tick%8===0) {
      if(phase==='hover') {
        const goal=api.spec.solve(api.spec.mount,new THREE.Vector3(.075,.0725,-.145),Math.PI/2,0,seed);
        if(!goal) throw new Error('hover IK');
        seed=Object.fromEntries(Object.keys(start).map(k=>[k,k==='gripper'?100:THREE.MathUtils.lerp(start[k],goal[k],alpha)]));
      } else {
        const y=phase==='descend'?THREE.MathUtils.lerp(.0725,.0225,alpha):phase==='lift'?THREE.MathUtils.lerp(.0225,.0725,alpha):.0225;
        const goal=api.spec.solve(api.spec.mount,new THREE.Vector3(.075,y,-.145),Math.PI/2,0,seed);
        if(!goal) throw new Error(phase+' IK');
        seed={...goal,gripper:phase==='close'?THREE.MathUtils.lerp(100,58.333,alpha):phase==='lift'?58.333:100};
      }
      Object.assign(api.currentValues,seed); api.commandArm(seed);
    }
    api.applyGravityCompensation(); physics.step(); api.syncArmFromPhysics();
    if(tick%8===0 && phase==='descend') {
      const b=api.armLinkBodies.get('arm_link6'),p=b.translation(),q=b.rotation();
      const tcp=new THREE.Vector3(.17,0,0).applyQuaternion(new THREE.Quaternion(q.x,q.y,q.z,q.w)).add(new THREE.Vector3(p.x,p.y,p.z));
      metrics.maxDescentErrorMm=Math.max(metrics.maxDescentErrorMm,tcp.distanceTo(api.constrainedIK.forwardPose(seed).position)*1000);
      for(const finger of api.fingerColliders) {
        const gap=finger.contactCollider(table,.2);
        if(gap) metrics.minFingerTableGapMm=Math.min(metrics.minFingerTableGapMm,gap.distance*1000);
        if(api.touching(finger,box)) metrics.openContacts++;
      }
    }
    if(tick%8===0 && ['close','lift'].includes(phase)) for(const finger of api.fingerColliders) {
      const gap=finger.contactCollider(box,.05);
      if(gap) metrics.maxGripPenetrationMm=Math.max(metrics.maxGripPenetrationMm,-gap.distance*1000);
    }
    if(tick%480===0||tick===6*480-1) snapshot(phase,t);
  }
  if(phase==='close') metrics.grippedAfterClose=api.fingerColliders.every(f=>api.touching(f,box,true));
}
metrics.liftMm=(cube.translation().y-.0225)*1000;
console.log(JSON.stringify({metrics}));
if(process.argv.includes('--assert')) {
  assert.ok(metrics.maxDescentErrorMm<2,'descent tracking within 2 mm');
  assert.ok(metrics.minFingerTableGapMm>3,'fingers stay above the table');
  assert.equal(metrics.openContacts,0,'no early contact during open descent');
  assert.ok(metrics.maxGripPenetrationMm<.5,'grip penetration below 0.5 mm');
  assert.ok(metrics.grippedAfterClose,'current contacts identify bilateral grip');
  assert.ok(metrics.liftMm>40,'cube physically lifted by over 40 mm');
  console.log('PASS: smooth descent, table clearance, no ghost contact, shallow grip penetration, bilateral grip and physical lift.');
}
physics.free();
