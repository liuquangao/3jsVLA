import * as THREE from "three";

type Values = Record<string, number>;
type Joint = {
  name: string;
  origin: THREE.Matrix4;
  axis: THREE.Vector3;
  min: number;
  max: number;
};

const POSITION_TOLERANCE = 0.0005;
const ANGLE_TOLERANCE = THREE.MathUtils.degToRad(0.4);
const ROTATION_WEIGHT = 0.1;
const MAX_STEP = THREE.MathUtils.degToRad(6);
const DIFFERENCE_STEP = 1e-5;

function rotationError(target: THREE.Quaternion, actual: THREE.Quaternion) {
  const difference = target.clone().multiply(actual.clone().invert()).normalize();
  if (difference.w < 0) {
    difference.set(-difference.x, -difference.y, -difference.z, -difference.w);
  }
  const vector = new THREE.Vector3(difference.x, difference.y, difference.z);
  const length = vector.length();
  return length < 1e-12 ? vector.set(0, 0, 0)
    : vector.multiplyScalar(2 * Math.atan2(length, difference.w) / length);
}

/** Box-constrained convex QP: min 0.5*x'Hx - b'x, lower <= x <= upper. */
function solveBoxQP(h: number[][], b: number[], lower: number[], upper: number[]) {
  const x = b.map(() => 0);
  for (let sweep = 0; sweep < 80; sweep += 1) {
    let largestChange = 0;
    for (let i = 0; i < x.length; i += 1) {
      let residual = b[i];
      for (let j = 0; j < x.length; j += 1) if (j !== i) residual -= h[i][j] * x[j];
      const next = THREE.MathUtils.clamp(residual / h[i][i], lower[i], upper[i]);
      largestChange = Math.max(largestChange, Math.abs(next - x[i]));
      x[i] = next;
    }
    if (largestChange < 1e-8) break;
  }
  return x;
}

/** Sequential constrained least squares over the actual URDF kinematic chain.
 * Joint limits are hard bounds; pose is a convergence requirement. Damping and a
 * per-iteration trust region regularize singular configurations. This is not a
 * collision planner or a torque controller.
 */
export class ConstrainedIK {
  constructor(
    private readonly base: THREE.Matrix4,
    private readonly joints: Joint[],
    private readonly tcp: THREE.Vector3,
  ) {}

  private forward(q: number[]) {
    const transform = this.base.clone();
    const rotation = new THREE.Matrix4();
    for (let i = 0; i < this.joints.length; i += 1) {
      const joint = this.joints[i];
      transform.multiply(joint.origin).multiply(rotation.makeRotationAxis(joint.axis, q[i]));
    }
    return {
      position: this.tcp.clone().applyMatrix4(transform),
      rotation: new THREE.Quaternion().setFromRotationMatrix(transform),
    };
  }

  forwardPose(values: Values) {
    return this.forward(this.joints.map((joint) => THREE.MathUtils.degToRad(values[joint.name])));
  }

  solve(position: THREE.Vector3, orientation: THREE.Quaternion, seeds: Values[]): Values | null {
    for (const seed of seeds) {
      if (this.joints.some((joint) => !Number.isFinite(seed[joint.name]))) continue;
      let q = this.joints.map((joint) => THREE.MathUtils.clamp(
        THREE.MathUtils.degToRad(seed[joint.name]), joint.min, joint.max,
      ));
      for (let iteration = 0; iteration < 80; iteration += 1) {
        const actual = this.forward(q);
        const offset = position.clone().sub(actual.position);
        const angle = rotationError(orientation, actual.rotation);
        if (offset.length() <= POSITION_TOLERANCE && angle.length() <= ANGLE_TOLERANCE) {
          return Object.fromEntries(this.joints.map((joint, i) => [joint.name, THREE.MathUtils.radToDeg(q[i])]));
        }
        const error = [...offset.toArray(), ...angle.multiplyScalar(ROTATION_WEIGHT).toArray()];
        const columns = this.joints.map((_, i) => {
          const perturbed = [...q];
          perturbed[i] += DIFFERENCE_STEP;
          const moved = this.forward(perturbed);
          const linear = moved.position.sub(actual.position).divideScalar(DIFFERENCE_STEP);
          const angular = rotationError(moved.rotation, actual.rotation)
            .multiplyScalar(ROTATION_WEIGHT / DIFFERENCE_STEP);
          return [...linear.toArray(), ...angular.toArray()];
        });
        const dot = (a: number[], b: number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);
        const h = columns.map((a, i) => columns.map((b, j) => dot(a, b) + (i === j ? 1e-5 : 0)));
        const rhs = columns.map((column) => dot(column, error));
        const lower = this.joints.map((joint, i) => Math.max(-MAX_STEP, joint.min - q[i]));
        const upper = this.joints.map((joint, i) => Math.min(MAX_STEP, joint.max - q[i]));
        const step = solveBoxQP(h, rhs, lower, upper);
        const cost = dot(error, error);
        let accepted = false;
        for (let scale = 1; scale >= 1 / 32; scale /= 2) {
          const candidate = q.map((value, i) => value + step[i] * scale);
          const moved = this.forward(candidate);
          const nextCost = moved.position.distanceToSquared(position)
            + rotationError(orientation, moved.rotation).lengthSq() * ROTATION_WEIGHT ** 2;
          if (nextCost < cost) {
            q = candidate;
            accepted = true;
            break;
          }
        }
        if (!accepted) break;
      }
    }
    return null;
  }
}
