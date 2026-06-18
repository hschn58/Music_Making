# Blender capture

A scriptable synthetic scene is the highest-information source we have: a game
engine *emits* the geometry the instrument needs — the observer's fovea is the
camera, and true per-pixel distance is the depth buffer — instead of forcing us
to recover it from a real photo. Photoreal PBR shading also gives a **dense,
continuous brightness histogram**, which is what the feature envelope is built
from. (A flat, low-palette world like Minecraft yields a spiky histogram and a
jagged envelope.)

## Render a capture

Blender ships its own Python, so run with the Blender binary, **not** the project
venv. Best performance is on a machine with a GPU (e.g. macOS Apple Silicon,
natively); the script falls back to Cycles-CPU so it also runs headless in a
container.

```bash
blender --background --python blender/render_scene.py -- OUT_DIR [N_FRAMES] [RES] [SAMPLES]
# example: 50 frames (1 s at 50 fps), 256 px, 16 samples
blender --background --python blender/render_scene.py -- captures/rock 50 256 16
```

### MVP scene

A camera dollies *laterally past* a single procedurally textured rock while
looking straight ahead, so the rock sweeps edge -> center -> edge. One clip
exercises both geometry knobs: the **foveal window** (eccentricity from screen
center) and the **1/sqrt(r) falloff** (distance varies, closest at mid-dolly).

## Output (`OUT_DIR/`)

| file | meaning |
|---|---|
| `frame_####.png` | sRGB render (Standard view transform — no tone-map between scene and histogram) |
| `depth_####.png` | 16-bit **Z-depth** (distance along the camera axis), normalized over the clip range; recovered to metres by the loader |
| `camera.json` | intrinsics (`fov_x`, resolution, fps) + per-frame camera-to-world matrix and gaze `forward` |

## Read it back

```python
from music_making.capture import load_capture
cap = load_capture("captures/rock")
f = cap.frames[0]
r = cap.range_map(f)        # Euclidean eye->pixel distance (for 1/sqrt(r))
```

`music_making.capture` also exposes the pure geometry — `pixel_eccentricity`,
`z_to_range` — which need no image libraries.

## Platform note

Official Blender ships for Linux **x86_64**, macOS, and Windows. On Linux
**arm64** there is no official binary or `pip install bpy` wheel; use the distro
package (`apt-get install blender`) for headless testing, or render on the host.
