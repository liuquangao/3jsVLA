# Galaxea (星海图) A1Z + G1Z robot asset

6-DoF A1Z arm with the G1Z parallel gripper. `A1Z_G1Z.urdf` plus the 9 STL files it
references, taken from Galaxea's official user-guide asset repository.

- Source: https://github.com/userguide-galaxea/URDF/tree/galaxea/main/A1Z/A1Z_G1Z
- Vendor SDK (same joint names, CAN IDs, gravity model): https://github.com/userguide-galaxea/GALAXEA-A1Z
- License: the upstream `userguide-galaxea/URDF` repository declares no license.
  These meshes are redistributed here for local simulation only; check with Galaxea
  before shipping them in anything public.

The URDF was exported from SolidWorks, uses metres, and carries full inertials.
Joint names match the vendor SDK: `arm_joint1` … `arm_joint6` (CAN IDs 0x01 … 0x06).

| Joint | Axis | Limits (rad) | Limits (deg) |
| --- | --- | --- | --- |
| arm_joint1 | Z (yaw) | -2.094 … 2.094 | -120 … 120 |
| arm_joint2 | Y (shoulder pitch) | 0 … 3.142 | 0 … 180 |
| arm_joint3 | Y (elbow pitch) | -3.142 … 0 | -180 … 0 |
| arm_joint4 | Y (wrist pitch) | -1.309 … 1.309 | -75 … 75 |
| arm_joint5 | Z (wrist yaw) | -1.484 … 1.484 | -85 … 85 |
| arm_joint6 | X (wrist roll) | -2.007 … 2.007 | -115 … 115 |

## Local modification

Upstream declares both gripper fingers as `type="fixed"`, because the real G1Z is driven
over its own channel rather than as part of the arm chain. To make the gripper animate in
the browser, the two finger joints were changed to `prismatic` and given an axis and a
limit — nothing else in the file was touched:

```xml
<joint name="gripper_finger_left_joint" type="prismatic">
  ...
  <axis xyz="0 1 0" />
  <limit lower="0" upper="0.03" effort="60" velocity="0.15" />
</joint>
<!-- gripper_finger_rIght_joint is identical with axis xyz="0 -1 0" -->
```

Both fingers are driven by the single `gripper` control, so `0` is fully closed (the jaw
tips meet at y=0 in the `arm_link6` frame, which is exactly where upstream's fixed joints
place them) and `0.03` per finger is fully open.

**The 30 mm per-finger stroke is an assumption**, not a vendor figure — Galaxea does not
publish the G1Z stroke in this URDF. Change `A1Z_FINGER_STROKE` in `src/main.ts` and the
two `upper` values above if you get the real number.

## Colours

The exporter gave all nine links the same SolidWorks default grey
(`rgba="0.75294 0.75294 0.75294 1"`), so straight out of the URDF the arm renders as one
flat colourless mass. The URDF is left alone; the collector recolours it per link from the
`finish` map in `src/main.ts`, with values read off Galaxea's own product render
(`docs/images/A1Z.png` in the SDK repo): black anodised base plate and wrist modules,
charcoal shoulder and upper arm, brushed-aluminium forearm. Change the map, not the URDF.

Note the upstream spelling `gripper_finger_rIght_*` (capital `I`); it is kept as-is so the
URDF and the mesh filenames stay in sync with the source repository.
