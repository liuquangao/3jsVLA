// Rapier may retain separated points in a contact manifold.
export const CONTACT_TOLERANCE = 0.0001;
export const MIN_GRIP_IMPULSE = 1e-8;

type ContactManifold = {
  numContacts(): number;
  contactDist(index: number): number;
  contactImpulse(index: number): number;
};

export function manifoldHasContact(
  manifold: ContactManifold,
  requireForce = false,
  distanceAt: (index: number) => number = (index) => manifold.contactDist(index),
) {
  for (let index = 0; index < manifold.numContacts(); index += 1) {
    const distance = distanceAt(index);
    if (!Number.isFinite(distance) || distance > CONTACT_TOLERANCE) continue;
    if (!requireForce) return true;
    const impulse = manifold.contactImpulse(index);
    if (Number.isFinite(impulse) && impulse > MIN_GRIP_IMPULSE) return true;
  }
  return false;
}

export function bothFingersGrip(fingers: boolean[]) {
  return fingers.length >= 2 && fingers.every(Boolean);
}
