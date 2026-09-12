# Desk model

The table the robot works on.

- Source: https://polyhaven.com/a/metal_office_desk
- License: CC0 — public domain, no attribution required (recorded here anyway)
- Downloaded as glTF with 1k textures: `.gltf` + `.bin` + three maps (diff, nor_gl, arm)

Y-up, standing on the floor, 2.0 m wide x 0.947 m deep with the **work surface at y = 0.7875**.
`src/main.ts` drops it by exactly that so the surface lands on the y = 0 work plane, and builds
the physics collider from the same numbers — if those two ever disagree the cubes float above
the desk or sink into it.

The collider is a single box covering the desktop. The drawers and legs underneath get no
collision geometry because nothing ever touches them.

This model was chosen over the smaller `WoodenTable_01` for its depth: props are scattered out to
roughly +/-0.35 m in z, and a 0.66 m deep table would drop them off the edge.
