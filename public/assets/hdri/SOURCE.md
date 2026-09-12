# Backdrops

Ten indoor panoramas, one picked at random per episode. Each is used twice: as the visible
background and, through a PMREM pass, as the scene's light source — so the arm actually picks up
the colour and direction of whichever room it is standing in.

- Source: https://polyhaven.com/hdris (each file is named after its asset id)
- License: CC0 — public domain, no attribution required (recorded here anyway)
- Downloaded at 1k `.hdr`, ~1.5 MB each

## Why randomise

A fixed backdrop is a shortcut. The same wall in the same place every episode is a free position
cue, and a visual policy will use it instead of looking at the cubes — the classic shortcut
learning failure, and one that training curves will not warn you about, because the policy scores
well right up until the background changes.

Adding or removing a panorama is a matter of dropping the `.hdr` in here and editing `BACKDROPS`
in `src/main.ts`. They are all loaded at startup rather than on demand, which costs a slower first
load but keeps `newEpisode` synchronous — generation runs on a fixed simulated clock and cannot
await anything mid-episode.
