import bpy
import bmesh
import math
from mathutils import Vector, Matrix, Euler

def get_or_create_material(name, base_color, roughness=0.4, metallic=0.0, specular=0.5, emission_color=None, emission_strength=0.0, alpha=1.0, is_glass=False):
    mat = bpy.data.materials.get(name)
    if not mat:
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = base_color
    bsdf.inputs['Roughness'].default_value = roughness
    if 'Metallic' in bsdf.inputs:
        bsdf.inputs['Metallic'].default_value = metallic
    if 'Specular IOR Level' in bsdf.inputs:
        bsdf.inputs['Specular IOR Level'].default_value = specular
    
    if alpha < 1.0 or is_glass:
        if 'Alpha' in bsdf.inputs:
            bsdf.inputs['Alpha'].default_value = alpha
        if 'Transmission Weight' in bsdf.inputs:
            bsdf.inputs['Transmission Weight'].default_value = 0.92 if is_glass else 0.0
        if 'IOR' in bsdf.inputs:
            bsdf.inputs['IOR'].default_value = 1.45
        if hasattr(mat, 'surface_render_method'):
            mat.surface_render_method = 'BLENDED'
    
    if emission_color and emission_strength > 0:
        if 'Emission Color' in bsdf.inputs:
            bsdf.inputs['Emission Color'].default_value = emission_color
            bsdf.inputs['Emission Strength'].default_value = emission_strength
        elif 'Emission' in bsdf.inputs:
            bsdf.inputs['Emission'].default_value = emission_color

    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    return mat

def create_mini_jeep_chassis():
    col_name = "Chasis_Mini_Jeep"
    sc = bpy.context.scene.collection
    
    col = bpy.data.collections.get(col_name)
    if not col:
        col = bpy.data.collections.new(col_name)
        sc.children.link(col)
        
    vl = bpy.context.view_layer
    def ensure_visible_lc(lc):
        lc.hide_viewport = False
        lc.exclude = False
        for child in lc.children:
            ensure_visible_lc(child)
    ensure_visible_lc(vl.layer_collection)
    
    # Clean previous Mini Jeep objects
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for m in list(bpy.data.meshes):
        if any(k in m.name for k in ["Jeep_", "MiniJeep"]):
            bpy.data.meshes.remove(m)
            
    old_col = bpy.data.collections.get("Chasis_Exoesqueleto_Offroad")
    if old_col:
        for obj in list(old_col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old_col)

    # 1. Automotive Materials
    mat_paint = get_or_create_material("Mat_Jeep_DesertSand", (0.58, 0.44, 0.28, 1.0), roughness=0.30, metallic=0.08, specular=0.6)
    mat_trim = get_or_create_material("Mat_Jeep_Black_Trim", (0.04, 0.04, 0.045, 1.0), roughness=0.55, metallic=0.05)
    mat_rollbar = get_or_create_material("Mat_Jeep_Rollbar", (0.10, 0.11, 0.13, 1.0), roughness=0.22, metallic=0.92)
    mat_chrome = get_or_create_material("Mat_Jeep_Chrome", (0.92, 0.93, 0.95, 1.0), roughness=0.08, metallic=0.98)
    mat_headlight = get_or_create_material("Mat_Jeep_Headlight_LED", (0.98, 0.98, 0.92, 1.0), roughness=0.08, emission_color=(1.0, 0.98, 0.90, 1.0), emission_strength=28.0)
    mat_amber = get_or_create_material("Mat_Jeep_TurnSignal", (0.95, 0.45, 0.02, 1.0), roughness=0.15, emission_color=(1.0, 0.45, 0.02, 1.0), emission_strength=14.0)
    mat_taillight = get_or_create_material("Mat_Jeep_Taillight", (0.95, 0.04, 0.04, 1.0), roughness=0.15, emission_color=(1.0, 0.02, 0.02, 1.0), emission_strength=20.0)
    mat_glass = get_or_create_material("Mat_Jeep_Windshield_Glass", (0.85, 0.92, 0.96, 1.0), roughness=0.05, alpha=0.35, is_glass=True)
    mat_tire = bpy.data.materials.get("Mat_TireRubber_PBR") or get_or_create_material("Mat_Jeep_TireRubber", (0.03, 0.03, 0.035, 1.0), roughness=0.65)
    mat_yellow = bpy.data.materials.get("Mat_YellowPlastic") or get_or_create_material("Mat_YellowPlastic", (0.98, 0.74, 0.02, 1.0), roughness=0.22)
    mat_red = get_or_create_material("Mat_Jeep_Shackle_Red", (0.90, 0.08, 0.08, 1.0), roughness=0.30, metallic=0.20)
    mat_jerry = get_or_create_material("Mat_Jeep_JerryCan_Red", (0.75, 0.10, 0.08, 1.0), roughness=0.35, metallic=0.10)

    # 2. Master Hierarchy Empty: Mini_Jeep_Assembly
    bpy.ops.object.empty_add(type='CUBE', radius=50.0, location=(0, 0, 0))
    master_empty = bpy.context.active_object
    master_empty.name = "Mini_Jeep_Assembly"
    col.objects.link(master_empty)
    if master_empty.name in sc.objects:
        sc.objects.unlink(master_empty)

    created_objects = []

    def register_part(obj, mat=None):
        if mat:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        obj.parent = master_empty
        if obj.name not in col.objects:
            col.objects.link(obj)
        if obj.name in sc.objects:
            sc.objects.unlink(obj)
        created_objects.append(obj)
        return obj

    # ==========================================
    # A. FRONT GRILL & HEADLIGHTS (ICONIC JEEP FACE)
    # ==========================================
    # Grill plate at Y = 117.0 mm (behind the headlights and slots)
    bm_grill = bmesh.new()
    vg0 = bm_grill.verts.new((-44.0, 117.0, 52.0))
    vg1 = bm_grill.verts.new(( 44.0, 117.0, 52.0))
    vg2 = bm_grill.verts.new(( 44.0, 117.0, 84.0))
    vg3 = bm_grill.verts.new((-44.0, 117.0, 84.0))
    bm_grill.faces.new([vg0, vg1, vg2, vg3])
    
    bmesh.ops.recalc_face_normals(bm_grill, faces=bm_grill.faces)
    me_grill = bpy.data.meshes.new("Mesh_Jeep_Grill")
    bm_grill.to_mesh(me_grill)
    bm_grill.free()
    obj_grill = bpy.data.objects.new("Jeep_Front_Grill", me_grill)
    register_part(obj_grill, mat_paint)

    # 7 Vertical Jeep Recessed Slots (Plates at Y = 118.2 mm)
    slot_xs = [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0]
    for i, sx in enumerate(slot_xs):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sx, 118.2, 68.0))
        slot = bpy.context.active_object
        slot.name = f"Jeep_Grill_Slot_{i+1}"
        slot.scale = (3.2, 2.0, 18.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(slot, mat_trim)

    # Dual Round Headlights with Chrome Bezels (Plates at Y = 118.5 - 120.5 mm)
    for sign, side in [(-1, "L"), (1, "R")]:
        hx = sign * 31.0
        hy = 118.5
        hz = 68.0
        # Chrome bezel ring
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=9.5, depth=4.0, location=(hx, hy + 2.0, hz))
        bezel = bpy.context.active_object
        bezel.name = f"Jeep_Headlight_Bezel_{side}"
        bezel.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(bezel, mat_chrome)

        # Emissive glass lens
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=8.0, depth=2.5, location=(hx, hy + 3.2, hz))
        lens = bpy.context.active_object
        lens.name = f"Jeep_Headlight_Lens_{side}"
        lens.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(lens, mat_headlight)

        # Round Amber Turn Signal below headlight
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=3.8, depth=2.5, location=(sign * 38.0, hy + 2.0, 57.0))
        amber = bpy.context.active_object
        amber.name = f"Jeep_TurnSignal_{side}"
        amber.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(amber, mat_amber)

    # ==========================================
    # B. FRONT STEEL OFF-ROAD BUMPER & WINCH SHACKLES
    # ==========================================
    # Front Bumper at Y = 127.5 mm, width 148 mm, Z in [24, 46]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 127.5, 35.0))
    fbump = bpy.context.active_object
    fbump.name = "Jeep_Front_Bumper"
    fbump.scale = (148.0, 14.0, 18.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(fbump, mat_trim)

    # Red D-Ring Recovery Shackles
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_torus_add(major_radius=4.5, minor_radius=1.6, location=(sign * 42.0, 136.0, 35.0))
        shackle = bpy.context.active_object
        shackle.name = f"Jeep_Bumper_Shackle_{side}"
        shackle.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(shackle, mat_red)

    # Winch Fairlead Plate
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 135.0, 35.0))
    fairlead = bpy.context.active_object
    fairlead.name = "Jeep_Winch_Fairlead"
    fairlead.scale = (32.0, 4.0, 10.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(fairlead, mat_chrome)

    # ==========================================
    # C. HOOD WITH CAMERA APERTURE & HOOD LATCHES
    # ==========================================
    # Tapered Jeep hood: Y from 70.0 (cowl) to 117.0 (grill)
    # Width at grill: X in [-44, 44], width at cowl: X in [-48, 48]
    bm_hood = bmesh.new()
    
    # Front edge at grill (Y = 117, Z = 84)
    vh_f0 = bm_hood.verts.new(( 0.0, 117.0, 85.5))  # Center power ridge
    vh_f1 = bm_hood.verts.new((22.0, 117.0, 84.5))
    vh_f2 = bm_hood.verts.new((44.0, 117.0, 84.0))

    # Mid hood (Y = 94, Z = 86.5)
    vh_m0 = bm_hood.verts.new(( 0.0,  94.0, 87.5))
    vh_m1 = bm_hood.verts.new((24.0,  94.0, 86.5))
    vh_m2 = bm_hood.verts.new((46.0,  94.0, 86.0))

    # Cowl front rim (Y = 82)
    vh_c0 = bm_hood.verts.new((14.0,  82.0, 82.5))  # Camera aperture corner
    vh_c1 = bm_hood.verts.new((26.0,  82.0, 87.5))
    vh_c2 = bm_hood.verts.new((48.0,  82.0, 87.0))

    # Cowl rear rim (Y = 68)
    vh_r0 = bm_hood.verts.new((14.0,  68.0, 83.0))  # Camera aperture rear
    vh_r1 = bm_hood.verts.new((26.0,  68.0, 89.0))
    vh_r2 = bm_hood.verts.new((48.0,  68.0, 88.5))

    # Center rear lip behind camera
    vh_rc = bm_hood.verts.new(( 0.0,  68.0, 90.0))
    vh_r_cut = bm_hood.verts.new((14.0, 68.0, 90.0))

    bm_hood.faces.new([vh_f0, vh_f1, vh_m1, vh_m0])
    bm_hood.faces.new([vh_f1, vh_f2, vh_m2, vh_m1])
    bm_hood.faces.new([vh_m0, vh_m1, vh_c1, bm_hood.verts.new((0.0, 94.0, 87.5))])
    bm_hood.faces.new([vh_m1, vh_m2, vh_c2, vh_c1])
    bm_hood.faces.new([vh_c1, vh_c2, vh_r2, vh_r1])
    bm_hood.faces.new([vh_rc, vh_r_cut, vh_r1, bm_hood.verts.new((0.0, 68.0, 89.0))])

    bmesh.ops.mirror(bm_hood, geom=bm_hood.verts[:] + bm_hood.edges[:] + bm_hood.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_hood, verts=bm_hood.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_hood, faces=bm_hood.faces)

    me_hood = bpy.data.meshes.new("Mesh_Jeep_Hood")
    bm_hood.to_mesh(me_hood)
    bm_hood.free()
    obj_hood = bpy.data.objects.new("Jeep_Hood", me_hood)
    mod_sol = obj_hood.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.5
    mod_sol.offset = -1.0
    register_part(obj_hood, mat_paint)

    # Classic Jeep Hood Latches (Left & Right)
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sign * 46.5, 96.0, 84.5))
        latch = bpy.context.active_object
        latch.name = f"Jeep_Hood_Latch_{side}"
        latch.scale = (2.2, 6.0, 5.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(latch, mat_trim)

    # ==========================================
    # D. ICONIC FLAT FENDERS (WILLYS / WRANGLER STYLE)
    # ==========================================
    for sign, side in [(-1, "L"), (1, "R")]:
        bm_ff = bmesh.new()
        xin = sign * 44.0
        xout = sign * 80.0
        
        # Top flat step surface
        v0 = bm_ff.verts.new((xin,  117.0, 76.0))
        v1 = bm_ff.verts.new((xout, 117.0, 76.0))
        v2 = bm_ff.verts.new((xout,  68.0, 76.0))
        v3 = bm_ff.verts.new((xin,   68.0, 76.0))
        bm_ff.faces.new([v0, v1, v2, v3])

        # Outer flare lip drop
        v4 = bm_ff.verts.new((xout + sign * 1.5, 117.0, 70.0))
        v5 = bm_ff.verts.new((xout + sign * 1.5,  68.0, 70.0))
        bm_ff.faces.new([v1, v4, v5, v2])

        # Front vertical skirt down to bumper
        v6 = bm_ff.verts.new((xin,  117.0, 52.0))
        v7 = bm_ff.verts.new((xout, 117.0, 52.0))
        bm_ff.faces.new([v0, v6, v7, v1])

        bmesh.ops.recalc_face_normals(bm_ff, faces=bm_ff.faces)
        me_ff = bpy.data.meshes.new(f"Mesh_Jeep_Fender_Front_{side}")
        bm_ff.to_mesh(me_ff)
        bm_ff.free()
        obj_ff = bpy.data.objects.new(f"Jeep_Fender_Front_{side}", me_ff)
        mod_sol = obj_ff.modifiers.new("Solidify", 'SOLIDIFY')
        mod_sol.thickness = 2.4
        mod_sol.offset = -1.0
        register_part(obj_ff, mat_trim)

        # Amber Side Marker Light on Front Fender Flare
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(xout + sign * 1.8, 92.0, 73.0))
        mkr = bpy.context.active_object
        mkr.name = f"Jeep_SideMarker_{side}"
        mkr.scale = (2.0, 8.0, 3.5)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(mkr, mat_amber)

        # Rear Trapezoidal Jeep Flare over rear wheels
        bm_rf = bmesh.new()
        x_body = sign * 56.0
        vr0 = bm_rf.verts.new((x_body, -20.0, 76.0))
        vr1 = bm_rf.verts.new((xout,   -28.0, 76.0))
        vr2 = bm_rf.verts.new((xout,   -80.0, 76.0))
        vr3 = bm_rf.verts.new((x_body, -88.0, 76.0))
        bm_rf.faces.new([vr0, vr1, vr2, vr3])

        vr4 = bm_rf.verts.new((xout + sign * 1.5, -28.0, 70.0))
        vr5 = bm_rf.verts.new((xout + sign * 1.5, -80.0, 70.0))
        bm_rf.faces.new([vr1, vr4, vr5, vr2])

        bmesh.ops.recalc_face_normals(bm_rf, faces=bm_rf.faces)
        me_rf = bpy.data.meshes.new(f"Mesh_Jeep_Fender_Rear_{side}")
        bm_rf.to_mesh(me_rf)
        bm_rf.free()
        obj_rf = bpy.data.objects.new(f"Jeep_Fender_Rear_{side}", me_rf)
        mod_sol = obj_rf.modifiers.new("Solidify", 'SOLIDIFY')
        mod_sol.thickness = 2.4
        mod_sol.offset = -1.0
        register_part(obj_rf, mat_trim)

    # ==========================================
    # E. BODY TUB & OPEN-DOOR COCKPIT (BATEA JEEP)
    # ==========================================
    bm_tub = bmesh.new()
    # Half X >= 0
    vt0 = bm_tub.verts.new((56.0,   68.0, 53.0))
    vt1 = bm_tub.verts.new((56.0,   12.0, 53.0))
    vt2 = bm_tub.verts.new((56.0,  -88.0, 53.0))
    vt3 = bm_tub.verts.new((56.0, -125.0, 53.0))

    vt4 = bm_tub.verts.new((56.0,   68.0, 86.0))
    vt5 = bm_tub.verts.new((56.0,   56.0, 62.0))
    vt6 = bm_tub.verts.new((56.0,   18.0, 62.0))
    vt7 = bm_tub.verts.new((56.0,   10.0, 78.0))
    vt8 = bm_tub.verts.new((56.0, -125.0, 78.0))

    bm_tub.faces.new([vt0, vt1, vt6, vt5, vt4])
    bm_tub.faces.new([vt1, vt2, vt3, vt8, vt7])

    # Tailgate face
    vt_tail_c0 = bm_tub.verts.new((0.0, -125.0, 53.0))
    vt_tail_c1 = bm_tub.verts.new((0.0, -125.0, 78.0))
    bm_tub.faces.new([vt_tail_c0, vt3, vt8, vt_tail_c1])

    bmesh.ops.mirror(bm_tub, geom=bm_tub.verts[:] + bm_tub.edges[:] + bm_tub.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_tub, verts=bm_tub.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_tub, faces=bm_tub.faces)

    me_tub = bpy.data.meshes.new("Mesh_Jeep_Body_Tub")
    bm_tub.to_mesh(me_tub)
    bm_tub.free()
    obj_tub = bpy.data.objects.new("Jeep_Body_Tub", me_tub)
    mod_sol = obj_tub.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.8
    mod_sol.offset = -1.0
    register_part(obj_tub, mat_paint)

    # Side Rock Rails / Steps
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12, radius=2.6, depth=72.0, location=(sign * 60.0, 25.0, 50.0)
        )
        slider = bpy.context.active_object
        slider.name = f"Jeep_Rock_Slider_{side}"
        slider.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(slider, mat_trim)

    # Classic Round Side Mirrors mounted on Cowl
    for sign, side in [(-1, "L"), (1, "R")]:
        # Mirror arm
        bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=1.4, depth=14.0, location=(sign * 54.0, 66.0, 92.0))
        arm = bpy.context.active_object
        arm.name = f"Jeep_Mirror_Arm_{side}"
        arm.rotation_euler = (0.0, sign * math.radians(65.0), 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(arm, mat_trim)

        # Mirror head round disc
        bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=5.5, depth=2.2, location=(sign * 60.5, 66.0, 95.0))
        mirror = bpy.context.active_object
        mirror.name = f"Jeep_Mirror_Head_{side}"
        mirror.rotation_euler = (math.radians(90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(mirror, mat_chrome)

    # ==========================================
    # F. WINDSHIELD FRAME & SAFETY GLASS
    # ==========================================
    bm_ws = bmesh.new()
    vwo0 = bm_ws.verts.new((-48.0, 68.0,  86.0))
    vwo1 = bm_ws.verts.new(( 48.0, 68.0,  86.0))
    vwo2 = bm_ws.verts.new(( 46.0, 63.5, 122.0))
    vwo3 = bm_ws.verts.new((-46.0, 63.5, 122.0))

    vwi0 = bm_ws.verts.new((-40.0, 67.5,  92.0))
    vwi1 = bm_ws.verts.new(( 40.0, 67.5,  92.0))
    vwi2 = bm_ws.verts.new(( 38.5, 64.0, 116.0))
    vwi3 = bm_ws.verts.new((-38.5, 64.0, 116.0))

    bm_ws.faces.new([vwo0, vwo1, vwi1, vwi0])
    bm_ws.faces.new([vwo1, vwo2, vwi2, vwi1])
    bm_ws.faces.new([vwo2, vwo3, vwi3, vwi2])
    bm_ws.faces.new([vwo3, vwo0, vwi0, vwi3])

    bmesh.ops.recalc_face_normals(bm_ws, faces=bm_ws.faces)
    me_ws = bpy.data.meshes.new("Mesh_Jeep_Windshield_Frame")
    bm_ws.to_mesh(me_ws)
    bm_ws.free()
    obj_ws = bpy.data.objects.new("Jeep_Windshield_Frame", me_ws)
    mod_sol = obj_ws.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 3.2
    mod_sol.offset = -1.0
    register_part(obj_ws, mat_paint)

    # Glass
    bm_glass = bmesh.new()
    vg0 = bm_glass.verts.new((-40.0, 67.5,  92.0))
    vg1 = bm_glass.verts.new(( 40.0, 67.5,  92.0))
    vg2 = bm_glass.verts.new(( 38.5, 64.0, 116.0))
    vg3 = bm_glass.verts.new((-38.5, 64.0, 116.0))
    bm_glass.faces.new([vg0, vg1, vg2, vg3])
    bmesh.ops.recalc_face_normals(bm_glass, faces=bm_glass.faces)
    me_glass = bpy.data.meshes.new("Mesh_Jeep_Windshield_Glass")
    bm_glass.to_mesh(me_glass)
    bm_glass.free()
    obj_glass = bpy.data.objects.new("Jeep_Windshield_Glass", me_glass)
    register_part(obj_glass, mat_glass)

    # ==========================================
    # G. TUBULAR SPORT ROLL BAR (CAGE WILYS/WRANGLER)
    # ==========================================
    cage_segments = []
    
    # Main B-Pillar Sport Hoop at Y = -12.0 mm, Z from 68 to 124 mm
    p_b_bl = Vector((-50.0, -12.0,  68.0))
    p_b_tl = Vector((-46.0, -12.0, 124.0))
    p_b_tr = Vector(( 46.0, -12.0, 124.0))
    p_b_br = Vector(( 50.0, -12.0,  68.0))

    for pa, pb in [(p_b_bl, p_b_tl), (p_b_tl, p_b_tr), (p_b_tr, p_b_br)]:
        vec = pb - pa
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=3.2, depth=vec.length, location=(0,0,0))
        c = bpy.context.active_object
        c.location = (pa + pb) / 2.0
        c.rotation_mode = 'QUATERNION'
        c.rotation_quaternion = Vector((0,0,1)).rotation_difference(vec.normalized())
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        cage_segments.append(c)

    # Top Forward Spreader Bars
    for sign in [-1, 1]:
        p_ws = Vector((sign * 42.0,  63.5, 121.0))
        p_hp = Vector((sign * 42.0, -12.0, 124.0))
        vec = p_hp - p_ws
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=2.8, depth=vec.length, location=(0,0,0))
        c = bpy.context.active_object
        c.location = (p_ws + p_hp) / 2.0
        c.rotation_mode = 'QUATERNION'
        c.rotation_quaternion = Vector((0,0,1)).rotation_difference(vec.normalized())
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        cage_segments.append(c)

    # Rear Diagonal Kicker Bars
    for sign in [-1, 1]:
        p_hp = Vector((sign * 42.0,  -12.0, 124.0))
        p_rd = Vector((sign * 50.0, -108.0,  78.0))
        vec = p_rd - p_hp
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=2.8, depth=vec.length, location=(0,0,0))
        c = bpy.context.active_object
        c.location = (p_hp + p_rd) / 2.0
        c.rotation_mode = 'QUATERNION'
        c.rotation_quaternion = Vector((0,0,1)).rotation_difference(vec.normalized())
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        cage_segments.append(c)

    bpy.ops.object.select_all(action='DESELECT')
    for c in cage_segments:
        c.select_set(True)
    bpy.context.view_layer.objects.active = cage_segments[0]
    bpy.ops.object.join()
    obj_rollcage = bpy.context.active_object
    obj_rollcage.name = "Jeep_Sport_Roll_Bar"
    register_part(obj_rollcage, mat_rollbar)

    # ==========================================
    # H. REAR BUMPER, TAILLIGHTS, JERRY CAN & SPARE WHEEL
    # ==========================================
    # Rear Steel Bumper at Y = -127.5 mm, Z = 35 mm
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -127.5, 35.0))
    rbump = bpy.context.active_object
    rbump.name = "Jeep_Rear_Bumper"
    rbump.scale = (148.0, 14.0, 18.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(rbump, mat_trim)

    # Classic Square Jeep Taillights
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sign * 50.0, -126.5, 68.0))
        tl = bpy.context.active_object
        tl.name = f"Jeep_Taillight_{side}"
        tl.scale = (12.0, 4.0, 14.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_part(tl, mat_taillight)

    # Mounted 5th Spare Wheel at Tailgate (offset slightly left to accommodate Jerry Can)
    # Centered at (-12, -137, 68), diameter 65 mm, width 22 mm
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=32.5, depth=22.0, location=(-12.0, -137.0, 68.0))
    spare_tire = bpy.context.active_object
    spare_tire.name = "Jeep_Spare_Tire"
    spare_tire.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(spare_tire, mat_tire)

    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=21.0, depth=3.0, location=(-12.0, -147.5, 68.0))
    spare_rim = bpy.context.active_object
    spare_rim.name = "Jeep_Spare_Rim"
    spare_rim.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(spare_rim, mat_yellow)

    bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=6.5, depth=4.0, location=(-12.0, -149.0, 68.0))
    spare_bolt = bpy.context.active_object
    spare_bolt.name = "Jeep_Spare_Mount_Bolt"
    spare_bolt.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(spare_bolt, mat_chrome)

    # Classic Military Jerry Can (Bidón Rojo de Combustible) mounted on the right rear
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(34.0, -132.0, 68.0))
    jerry = bpy.context.active_object
    jerry.name = "Jeep_Jerry_Can_Fuel"
    jerry.scale = (16.0, 10.0, 24.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(jerry, mat_jerry)

    # Jerry Can Handle & Cap
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(34.0, -132.0, 81.5))
    jerry_h = bpy.context.active_object
    jerry_h.name = "Jeep_Jerry_Can_Handle"
    jerry_h.scale = (12.0, 6.0, 3.5)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_part(jerry_h, mat_trim)

    print(f"=== REFINED MINI JEEP CHASSIS COMPLETED SUCCESSFULLY ===")
    print(f"Total parts created: {len(created_objects)}")
    return master_empty

if __name__ == "__main__":
    create_mini_jeep_chassis()
