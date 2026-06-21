"""A much richer walk-through scene than ``walk_features.py`` — the point is to feed the
pipeline a world with real *within-feature* variety so the per-feature color density E(f)
is wide (many catalogue peaks), not 1-2 lines. The levers:

  * MULTI-HUE textured materials (color ramp over noise) on the big features, so a single
    object's pixels span a range of hues -> a rich spectrum, including a full-rainbow
    "meadow" patch and a gray-noise "rocks" feature (broadband / noise pole).
  * more, BIGGER, BRIGHTER objects, lit properly (no dusk crush) so each feature reads.
  * the same object-index + Z-depth passes, so it plugs straight into capture.py.

    blender --background --python blender/walk_complex.py -- OUT_DIR [N_FRAMES] [RES] [SAMPLES]

Feature pass_index map (mirror in scripts/blender_feature_test.py FEATURES):
    1 ground  2 water  3 conifers  4 fire  5 red  6 yellow  7 purple  8 cyan  9 orange
    10 meadow(rainbow)  11 magenta  12 foliage  13 rocks(gray)  14 blossoms
Background (sky) stays index 0 = the void.
"""

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_scene import _write_camera_json  # noqa: E402

MAXIDX = 16


def _args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return (a[0] if a else "/tmp/walk_complex"), int(a[1]) if len(a) > 1 else 72, \
        int(a[2]) if len(a) > 2 else 200, int(a[3]) if len(a) > 3 else 24


def _flat_mat(name, color, rough=0.45, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def _ramp_mat(name, stops, scale=8.0, detail=8.0, rough=0.6, metallic=0.0):
    """Color-ramp over noise -> within-object hue variety. stops: [(pos,(r,g,b)), ...]."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    cr = ramp.color_ramp
    cr.elements[0].position = float(stops[0][0])
    cr.elements[0].color = (*stops[0][1], 1.0)
    cr.elements[1].position = float(stops[-1][0])
    cr.elements[1].color = (*stops[-1][1], 1.0)
    for pos, col in stops[1:-1]:
        e = cr.elements.new(float(pos))
        e.color = (*col, 1.0)
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def _emit_mat(name, color, strength=12.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    return m, em


def _add(prim, mat, idx, **kw):
    getattr(bpy.ops.mesh, prim)(**kw)
    o = bpy.context.active_object
    o.data.materials.append(mat)
    o.pass_index = idx
    bpy.ops.object.shade_smooth()
    return o


RAINBOW = [(0.0, (0.90, 0.05, 0.05)), (0.17, (0.95, 0.5, 0.05)), (0.33, (0.95, 0.9, 0.1)),
           (0.5, (0.1, 0.8, 0.2)), (0.67, (0.1, 0.7, 0.85)), (0.83, (0.1, 0.2, 0.9)),
           (1.0, (0.6, 0.1, 0.85))]
EARTH = [(0.0, (0.06, 0.12, 0.04)), (0.4, (0.2, 0.18, 0.07)), (0.7, (0.36, 0.26, 0.1)),
         (1.0, (0.5, 0.4, 0.16))]
WATER = [(0.0, (0.02, 0.15, 0.32)), (0.4, (0.05, 0.38, 0.52)), (0.75, (0.12, 0.6, 0.62)),
         (1.0, (0.75, 0.88, 0.92))]
GREEN = [(0.0, (0.03, 0.15, 0.04)), (0.5, (0.07, 0.32, 0.08)), (1.0, (0.18, 0.5, 0.12))]
FOLIAGE = [(0.0, (0.05, 0.25, 0.05)), (0.5, (0.25, 0.5, 0.1)), (1.0, (0.65, 0.78, 0.16))]
ROCK = [(0.0, (0.11, 0.11, 0.12)), (0.5, (0.3, 0.3, 0.32)), (1.0, (0.58, 0.58, 0.6))]
BLOSSOM = [(0.0, (0.96, 0.96, 0.96)), (0.5, (0.96, 0.6, 0.8)), (1.0, (0.85, 0.1, 0.5))]


def _world(scene):
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(32)
    sky.sun_rotation = math.radians(200)
    nt.links.new(sky.outputs[0], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.6


def main():
    out_dir, n_frames, res, samples = _args()
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _world(scene)

    # terrain + water (big, multi-hue)
    _add("primitive_plane_add", _ramp_mat("ground", EARTH, scale=14, detail=10, rough=0.9),
         1, size=90, location=(0, 0, 0))
    _add("primitive_plane_add", _ramp_mat("water", WATER, scale=9, rough=0.06, metallic=0.4),
         2, size=14, location=(6.0, 11.0, 0.02))

    # conifers (varied greens), several along the path
    for n, (x, y, h) in enumerate([(-5, 3, 3.2), (6, 0, 2.6), (-7, -2, 2.9),
                                   (4, 6, 2.2), (-3, 9, 3.4), (7, 8, 2.5)]):
        _add("primitive_cone_add", _ramp_mat(f"tree{n}", GREEN, scale=11, rough=0.85),
             3, radius1=0.9, depth=h, location=(x, y, h / 2), vertices=24)

    # fire (flickering emitter)
    fire_mat, em = _emit_mat("fire", (1.0, 0.45, 0.08))
    _add("primitive_uv_sphere_add", fire_mat, 4, radius=0.7, location=(-3.0, 1.0, 0.7))
    for f, s in [(0, 10), (n_frames // 3, 18), (2 * n_frames // 3, 9), (n_frames, 15)]:
        em.inputs["Strength"].default_value = s
        em.inputs["Strength"].keyframe_insert("default_value", frame=f)

    # bright saturated crystals (clean tones) — bigger so they read, staggered along path
    for (x, y), col, idx, r in [((-1.5, -4), (0.85, 0.05, 0.05), 5, 0.9),     # red
                                ((2.2, -1), (0.92, 0.85, 0.05), 6, 0.9),       # yellow
                                ((-2.8, 4), (0.62, 0.06, 0.78), 7, 1.1),       # purple (bigger)
                                ((1.4, 8), (0.05, 0.72, 0.78), 8, 0.95),       # cyan
                                ((3.8, 5.5), (0.95, 0.4, 0.05), 9, 1.0)]:      # orange (bigger)
        _add("primitive_ico_sphere_add", _flat_mat(f"x{idx}", col, rough=0.35),
             idx, subdivisions=3, radius=r, location=(x, y, r))

    # the RICH features: a rainbow meadow patch + magenta torus + foliage + gray rocks + blossoms
    _add("primitive_plane_add", _ramp_mat("meadow", RAINBOW, scale=30, detail=12, rough=0.7),
         10, size=9, location=(-0.5, 2.5, 0.05))
    _add("primitive_torus_add", _flat_mat("magenta", (0.82, 0.1, 0.6), rough=0.4),
         11, major_radius=0.9, minor_radius=0.32, location=(3.2, -3.5, 0.7))
    for n, (x, y) in enumerate([(-4.5, 5.5), (-5.5, 6.5)]):
        _add("primitive_ico_sphere_add", _ramp_mat(f"foliage{n}", FOLIAGE, scale=16, rough=0.8),
             12, subdivisions=3, radius=0.8, location=(x, y, 0.8))
    for n, (x, y, r) in enumerate([(2.0, -2.5, 0.7), (5.0, 3.0, 0.9), (-6.0, 1.0, 0.6)]):
        _add("primitive_ico_sphere_add", _ramp_mat(f"rock{n}", ROCK, scale=20, rough=0.95),
             13, subdivisions=2, radius=r, location=(x, y, r * 0.7))
    for n, (x, y) in enumerate([(-2.0, 7.0), (1.0, 4.0), (0.0, -1.5)]):
        _add("primitive_uv_sphere_add", _ramp_mat(f"bloom{n}", BLOSSOM, scale=22, rough=0.5),
             14, radius=0.45, location=(x, y, 0.45))

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 4.0
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(58), 0, math.radians(200))
    scene.collection.objects.link(so)

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_euler = (math.radians(76), 0, 0)
    cam_data.lens = 24.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 100.0
    cam.location = (0, -13, 1.7)
    cam.keyframe_insert("location", frame=0)
    cam.location = (0, 9, 1.7)
    cam.keyframe_insert("location", frame=max(1, n_frames - 1))
    for fc in cam.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.fps = 24
    scene.frame_start = 0
    scene.frame_end = max(0, n_frames - 1)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.render.filepath = os.path.join(out_dir, "_composite_")

    # --- compositor: RGB + Z-depth + object-index passes (same as walk_features) ----- #
    vl = scene.view_layers[0]
    vl.use_pass_z = True
    vl.use_pass_object_index = True
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")

    rgb = tree.nodes.new("CompositorNodeOutputFile")
    rgb.base_path = out_dir
    rgb.file_slots[0].path = "frame_"
    rgb.format.file_format = "PNG"
    rgb.format.color_mode = "RGB"
    tree.links.new(rl.outputs["Image"], rgb.inputs[0])

    norm = tree.nodes.new("CompositorNodeMapRange")
    norm.inputs["From Min"].default_value = cam_data.clip_start
    norm.inputs["From Max"].default_value = cam_data.clip_end
    norm.inputs["To Min"].default_value = 0.0
    norm.inputs["To Max"].default_value = 1.0
    norm.use_clamp = True
    tree.links.new(rl.outputs["Depth"], norm.inputs["Value"])
    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.base_path = out_dir
    depth.file_slots[0].path = "depth_"
    depth.format.file_format = "PNG"
    depth.format.color_mode = "BW"
    depth.format.color_depth = "16"
    tree.links.new(norm.outputs["Value"], depth.inputs[0])

    inorm = tree.nodes.new("CompositorNodeMapRange")
    inorm.inputs["From Min"].default_value = 0.0
    inorm.inputs["From Max"].default_value = float(MAXIDX)
    inorm.inputs["To Min"].default_value = 0.0
    inorm.inputs["To Max"].default_value = 1.0
    inorm.use_clamp = True
    tree.links.new(rl.outputs["IndexOB"], inorm.inputs["Value"])
    index = tree.nodes.new("CompositorNodeOutputFile")
    index.base_path = out_dir
    index.file_slots[0].path = "index_"
    index.format.file_format = "PNG"
    index.format.color_mode = "BW"
    index.format.color_depth = "16"
    tree.links.new(inorm.outputs["Value"], index.inputs[0])

    _write_camera_json(scene, cam, out_dir, res)
    bpy.ops.render.render(animation=True)
    print(f"[walk_complex] wrote {n_frames} frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
