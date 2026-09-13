import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { stripTypeScriptTypes } from "node:module";

const source = await readFile(new URL("../src/contact-state.ts", import.meta.url), "utf8");
const { manifoldHasContact, bothFingersGrip } = await import(
  `data:text/javascript;base64,${Buffer.from(stripTypeScriptTypes(source)).toString("base64")}`
);
const manifold = (points) => ({
  numContacts: () => points.length,
  contactDist: (i) => points[i][0],
  contactImpulse: (i) => points[i][1],
});
const separated = manifold([[0.015, 0], [0.021, 0]]);
assert.equal(manifoldHasContact(separated), false);
assert.equal(manifoldHasContact(separated, true), false);
assert.equal(manifoldHasContact(manifold([])), false);
assert.equal(manifoldHasContact(manifold([[0.0001, 0]])), true);
assert.equal(manifoldHasContact(manifold([[0.0001001, 1]])), false);
assert.equal(manifoldHasContact(manifold([[-0.0001, 0]]), true), false);
assert.equal(manifoldHasContact(manifold([[0, 1e-8]]), true), false);
assert.equal(manifoldHasContact(manifold([[0, 2e-8]]), true), true);
assert.equal(manifoldHasContact(manifold([[0, 0], [0.015, 1]]), true), false);
assert.equal(manifoldHasContact(manifold([[NaN, 1], [Infinity, 1]]), true), false);
assert.equal(manifoldHasContact(manifold([[0, NaN]]), true), false);
const forceContact = manifold([[-0.00001, 0.0001]]);
const left = manifoldHasContact(forceContact, true);
assert.equal(manifoldHasContact(forceContact), true);
assert.equal(bothFingersGrip([left, false]), false);
assert.equal(bothFingersGrip([left, left]), true);
assert.equal(bothFingersGrip([left]), false);
assert.equal(bothFingersGrip([]), false);
assert.equal(bothFingersGrip([manifoldHasContact(separated, true), manifoldHasContact(separated, true)]), false);
assert.equal(manifoldHasContact(manifold([[0.02, 0.001]]), true, () => -0.00005), true);
assert.equal(manifoldHasContact(manifold([[-0.0001, 0.001]]), true, () => 0.01), false);
console.log("PASS: 19 contact checks, including current separation, cached points, impulse thresholds, same-point matching, bilateral grip and release.");
