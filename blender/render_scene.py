"""Render a scriptable synthetic scene to a *capture* the music pipeline can read.

Run with a Blender binary (NOT the project venv) — Blender ships its own Python:

    blender --background --python blender/render_scene.py -- OUT_DIR [N_FRAMES] [RES] [SAMPLES]

Why a synthetic scene instead of a real photo/video: a game engine *emits* the
geometry we want (the observer's fovea = the camera, true per-pixel distance =
the depth buffer) instead of forcing us to recover it. And photoreal PBR shading
gives a *dense, continuous* brightness histogram — the signal the feature
envelope is built from. (A flat-shaded, low-palette world like Minecraft has only
a few brightness values, so the histogram is spiky and the envelope comes out
jagged.)

The MVP scene is the smallest thing that validates the instrument end-to-end: a
camera dollies *laterally past* a single procedurally textured rock, looking
straight ahead. The rock therefore sweeps from one edge of frame, through the
center, to the other edge — so a single clip exercises BOTH the foveal window
(eccentricity from screen center) and the 1/sqrt(r) falloff (distance varies,
closest at mid-dolly).

Outputs into OUT_DIR:
    frame_####.png   sRGB render (Standard view transform: no tone-map surprises)
    depth_####.exr   linear float Z-depth (distance along the camera axis), per pixel
    camera.json      intrinsics + per-frame camera-to-world matrix (the fovea/path)
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out_dir = argv[0] if len(argv) > 0 else "/tmp/capture"
    n_frames = int(argv[1]) if len(argv) > 1 else 50
    res = int(argv[2]) if len(argv) > 2 else 256
    samples = int(argv[3]) if len(argv) > 3 else 16
    return out_dir, n_frames, res, samples


def _reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.05, 0.06, 0.08, 1.0)  # dim ambient sky
    bg.inputs["Strength"].default_value = 0.6
    return scene


def _rock():
    """A displaced icosphere with a fully procedural PBR material — no external
    assets, continuous tone so the brightness histogram is dense."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.5, location=(0, 0, 0))
    rock = bpy.context.active_object

    # craggy surface: a fractal-noise displacement modifier
    tex = bpy.data.textures.new("rock_disp", type="CLOUDS")
    tex.noise_scale = 0.55
    tex.noise_depth = 4
    disp = rock.modifiers.new("disp", "DISPLACE")
    disp.texture = tex
    disp.strength = 0.6
    bpy.ops.object.shade_smooth()

    mat = bpy.data.materials.new("rock")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")

    # base colour: noise -> colour ramp (mottled brown/grey, continuous)
    c_noise = nt.nodes.new("ShaderNodeTexNoise")
    c_noise.inputs["Scale"].default_value = 6.0
    c_noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.13, 0.11, 0.09, 1.0)
    ramp.color_ramp.elements[1].color = (0.55, 0.50, 0.44, 1.0)
    nt.links.new(tc.outputs["Object"], c_noise.inputs["Vector"])
    nt.links.new(c_noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # spatially varying roughness (real surface, not a flat sheen)
    r_noise = nt.nodes.new("ShaderNodeTexNoise")
    r_noise.inputs["Scale"].default_value = 15.0
    nt.links.new(tc.outputs["Object"], r_noise.inputs["Vector"])
    nt.links.new(r_noise.outputs["Fac"], bsdf.inputs["Roughness"])

    # fine grain via bump
    b_noise = nt.nodes.new("ShaderNodeTexNoise")
    b_noise.inputs["Scale"].default_value = 42.0
    b_noise.inputs["Detail"].default_value = 10.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    nt.links.new(tc.outputs["Object"], b_noise.inputs["Vector"])
    nt.links.new(b_noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    rock.data.materials.append(mat)
    return rock


def _sun():
    light = bpy.data.lights.new("sun", "SUN")
    light.energy = 3.0
    obj = bpy.data.objects.new("sun", light)
    obj.rotation_euler = (math.radians(50), 0.0, math.radians(35))
    bpy.context.scene.collection.objects.link(obj)


def _camera(scene, n_frames):
    """Camera stands back on -Y, looks straight down +Y (fixed forward), and
    dollies in X from -4 to +4 so the rock sweeps across the frame.

    First-person convention: the camera looks straight ahead, so the center of
    vision is simply the center of every frame (no separate focal point)."""
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)  # -Z of camera -> +Y world
    cam_data.lens = 24.0        # ~74 deg FOV: rock stays framed across the dolly
    cam_data.clip_start = 0.1
    cam_data.clip_end = 50.0    # modest range -> good 16-bit depth precision

    # dolly laterally past the rock; amplitude keeps it on-screen throughout so
    # eccentricity sweeps 0..~27 deg (edge -> center -> edge) within the frame
    cam.location = (-2.5, -5.0, 0.0)
    cam.keyframe_insert("location", frame=0)
    cam.location = (2.5, -5.0, 0.0)
    cam.keyframe_insert("location", frame=max(1, n_frames - 1))
    # constant dolly speed
    for fc in cam.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    return cam


def _compositor(scene, out_dir):
    """Route the rendered image and the Z-depth pass to per-frame files.

    Depth is written as a 16-bit PNG normalized over the camera clip range
    (recovered to metric Z in the loader) — float EXR would be cleaner, but it
    avoids an OpenEXR dependency that isn't always present."""
    cam = scene.camera.data
    scene.view_layers[0].use_pass_z = True
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

    # map [clip_start, clip_end] -> [0, 1] so it survives a 16-bit integer PNG
    norm = tree.nodes.new("CompositorNodeMapRange")
    norm.inputs["From Min"].default_value = cam.clip_start
    norm.inputs["From Max"].default_value = cam.clip_end
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


def _write_camera_json(scene, cam, out_dir, res):
    frames = []
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        m = cam.matrix_world
        fwd = (m.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        frames.append({
            "index": f,
            "matrix_world": [list(row) for row in m],
            "location": list(m.translation),
            "forward": list(fwd),
        })
    data = {
        "resolution": [res, res],
        "fov_x": cam.data.angle_x,
        "fov_y": cam.data.angle_y,
        "lens_mm": cam.data.lens,
        "sensor_width_mm": cam.data.sensor_width,
        "clip_start": cam.data.clip_start,
        "clip_end": cam.data.clip_end,
        "fps": scene.render.fps,
        "frames": frames,
    }
    with open(os.path.join(out_dir, "camera.json"), "w") as fh:
        json.dump(data, fh, indent=2)


def main():
    out_dir, n_frames, res, samples = _args()
    os.makedirs(out_dir, exist_ok=True)

    scene = _reset()
    _rock()
    _sun()
    cam = _camera(scene, n_frames)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"          # no GPU in CI/containers; portable
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False    # denoiser isn't in every Cycles build
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.fps = 50
    scene.frame_start = 0
    scene.frame_end = max(0, n_frames - 1)
    # measure honestly: no film tone-mapping between the scene and the histogram
    scene.view_settings.view_transform = "Standard"
    scene.render.filepath = os.path.join(out_dir, "_composite_")  # keep stray output contained

    _compositor(scene, out_dir)
    _write_camera_json(scene, cam, out_dir, res)
    bpy.ops.render.render(animation=True)
    print(f"[render_scene] wrote {n_frames} frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
