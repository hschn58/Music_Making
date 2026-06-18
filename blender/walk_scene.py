"""A richer scene to walk through — color variety is what makes the music rich.

A dusk clearing: warm Nishita sky, mottled green/brown ground, a blue water patch,
green conifers, an orange fire glow (gently flickering), and scattered saturated
objects (red/yellow/purple/cyan) for hue variety. A first-person camera walks
forward through it, so features approach, pass, and change — the time axis.

    blender --background --python blender/walk_scene.py -- OUT_DIR [N_FRAMES] [RES] [SAMPLES]
"""

import math
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_scene import _compositor, _write_camera_json  # noqa: E402


def _args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return (a[0] if a else "/tmp/walk"), int(a[1]) if len(a) > 1 else 28, \
        int(a[2]) if len(a) > 2 else 160, int(a[3]) if len(a) > 3 else 16


def _mat(name, color, rough=0.6, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def _ground_mat():
    m = bpy.data.materials.new("ground")
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Roughness"].default_value = 0.9
    tc = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 8.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.06, 0.12, 0.04, 1.0)   # mossy green
    ramp.color_ramp.elements[1].color = (0.22, 0.16, 0.08, 1.0)   # dry earth
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


def _add(prim, mat, **kw):
    getattr(bpy.ops.mesh, prim)(**kw)
    o = bpy.context.active_object
    o.data.materials.append(mat)
    bpy.ops.object.shade_smooth()
    return o


def _world(scene):
    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "NISHITA"
    sky.sun_elevation = math.radians(8)      # low sun -> warm dusk colours
    sky.sun_rotation = math.radians(200)
    nt.links.new(sky.outputs[0], nt.nodes["Background"].inputs["Color"])
    nt.nodes["Background"].inputs["Strength"].default_value = 0.3   # don't let the sky blow out


def _fire(scene, n_frames):
    m = bpy.data.materials.new("fire")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.45, 0.08, 1.0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    _add("primitive_uv_sphere_add", m, radius=0.5, location=(-2.2, 2.0, 0.5))
    # gentle flicker
    for f, s in [(0, 12), (n_frames // 3, 18), (2 * n_frames // 3, 10), (n_frames, 15)]:
        em.inputs["Strength"].default_value = s
        em.inputs["Strength"].keyframe_insert("default_value", frame=f)


def main():
    out_dir, n_frames, res, samples = _args()
    os.makedirs(out_dir, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    _world(scene)

    _add("primitive_plane_add", _ground_mat(), size=60, location=(0, 0, 0))
    _add("primitive_plane_add", _mat("water", (0.03, 0.18, 0.32), rough=0.05, metallic=0.4),
         size=10, location=(4.5, 9.0, 0.02))
    # conifers (green), varied
    for x, y, h in [(-4, 6, 3.0), (5, 1, 2.4), (-6, -1, 2.8), (3, 5, 2.0)]:
        _add("primitive_cone_add", _mat(f"tree{x}{y}", (0.05, 0.22, 0.06), rough=0.8),
             radius1=0.8, depth=h, location=(x, y, h / 2))
    # scattered saturated objects -> hue variety -> pitch variety
    for (x, y), col in [((-1, 0), (0.7, 0.05, 0.05)), ((2, 3), (0.8, 0.7, 0.05)),
                        ((-3, 4), (0.5, 0.05, 0.6)), ((1, -2), (0.05, 0.5, 0.6)),
                        ((4, 4), (0.8, 0.3, 0.05))]:
        _add("primitive_ico_sphere_add", _mat(f"o{x}{y}", col, rough=0.4),
             subdivisions=3, radius=0.35, location=(x, y, 0.35))
    _fire(scene, n_frames)

    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 3.0
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(70), 0, math.radians(200))
    scene.collection.objects.link(so)

    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_euler = (math.radians(74), 0, 0)   # look +Y, tilted down onto the ground/objects
    cam_data.lens = 28.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 80.0
    cam.location = (0, -11, 1.6)
    cam.keyframe_insert("location", frame=0)
    cam.location = (0, 7, 1.6)
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
    scene.view_settings.exposure = -1.0    # expose for the ground/objects, not the sky
    scene.render.filepath = os.path.join(out_dir, "_composite_")

    _compositor(scene, out_dir)
    _write_camera_json(scene, cam, out_dir, res)
    bpy.ops.render.render(animation=True)
    print(f"[walk_scene] wrote {n_frames} frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
