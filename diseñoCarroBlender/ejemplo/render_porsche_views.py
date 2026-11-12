import bpy
import math
from mathutils import Vector, Euler

def setup_render_and_shoot():
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if hasattr(bpy.types, 'RenderSettings') and 'BLENDER_EEVEE_NEXT' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 800
    scene.render.resolution_percentage = 100

    # Ensure lighting is vibrant and clean
    light_names = ["Key_Light", "Fill_Light", "Rim_Light", "Top_Light"]
    lights_data = [
        ("Key_Light", (220.0, 220.0, 300.0), 350.0, (1.0, 0.98, 0.95)),
        ("Fill_Light", (-250.0, 180.0, 200.0), 200.0, (0.9, 0.95, 1.0)),
        ("Rim_Light", (0.0, -320.0, 250.0), 300.0, (1.0, 1.0, 1.0)),
        ("Top_Light", (0.0, 0.0, 350.0), 250.0, (1.0, 1.0, 1.0))
    ]
    
    entorno_col = bpy.data.collections.get("Entorno")
    if not entorno_col:
        entorno_col = bpy.data.collections.new("Entorno")
        scene.collection.children.link(entorno_col)

    for lname, loc, power, col in lights_data:
        l_obj = bpy.data.objects.get(lname)
        if not l_obj:
            l_data = bpy.data.lights.new(name=lname, type='POINT')
            l_obj = bpy.data.objects.new(name=lname, object_data=l_data)
            entorno_col.objects.link(l_obj)
        l_obj.location = loc
        l_obj.data.energy = power
        l_obj.data.color = col

    # Camera setup
    cam = bpy.data.objects.get("Render_Cam")
    if not cam:
        cam_data = bpy.data.cameras.new(name="Render_Cam")
        cam = bpy.data.objects.new(name="Render_Cam", object_data=cam_data)
        entorno_col.objects.link(cam)
    scene.camera = cam

    # Set focal length
    cam.data.lens = 55.0

    output_dir = r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo"

    # Shot 1: Perspective 3/4 Front View (matching reference photo!)
    # Looking down and from the front-right
    cam.location = (240.0, 240.0, 190.0)
    # Track towards center of car at (0, 20, 65)
    target = Vector((0.0, 20.0, 65.0))
    direction = target - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()

    scene.render.filepath = f"{output_dir}\\porsche_911_isometric_front.png"
    bpy.ops.render.render(write_still=True)
    print("Rendered:", scene.render.filepath)

    # Shot 2: Side View (Wheel Clearance Demonstration)
    cam.location = (420.0, 0.0, 75.0)
    target = Vector((0.0, 0.0, 65.0))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    cam.data.lens = 70.0

    scene.render.filepath = f"{output_dir}\\porsche_911_side_clearance.png"
    bpy.ops.render.render(write_still=True)
    print("Rendered:", scene.render.filepath)

    # Shot 3: Top-Down View (Sunroof, windshield and chassis bounds)
    cam.location = (0.0, 0.0, 480.0)
    cam.rotation_euler = Euler((0.0, 0.0, math.radians(90)), 'XYZ')
    cam.data.lens = 65.0

    scene.render.filepath = f"{output_dir}\\porsche_911_top_view.png"
    bpy.ops.render.render(write_still=True)
    print("Rendered:", scene.render.filepath)

setup_render_and_shoot()
