import bpy
import bmesh
import math
from mathutils import Vector, Matrix, Euler

def get_or_create_material(name, base_color, roughness=0.4, metallic=0.0, emission_color=None, emission_strength=0.0):
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

def create_offroad_chassis_and_exoskeleton():
    sc = bpy.context.scene.collection
    col_name = "Chasis_Exoesqueleto_Offroad"
    col = bpy.data.collections.get(col_name)
    if not col:
        col = bpy.data.collections.new(col_name)
        sc.children.link(col)
    
    # Clean previous versions of our offroad objects
    for o in list(sc.objects):
        if any(k in o.name for k in ["Offroad_", "Fender_", "Cabin_", "Armored_", "Roof_", "Exoskeleton_", "Taillight_Cyber", "Headlight_Cyber", "Treads_Wheel", "Wheel_HubCap", "Hood_Intake", "Slider_Brass", "Front_BullBar", "Rear_Tow"]):
            bpy.data.objects.remove(o, do_unlink=True)
    for o in list(col.objects):
        bpy.data.objects.remove(o, do_unlink=True)
        
    for m in list(bpy.data.meshes):
        if any(k in m.name for k in ["Offroad", "Fender", "Bumper", "Exoskeleton", "Louvers", "Taillight", "Headlight", "Treads_", "Pillar", "Skid", "HubCap"]):
            bpy.data.meshes.remove(m)
    
    # 1. Materials
    # Tactical slate blue-grey armor (matching reference style)
    mat_armor = get_or_create_material("Mat_Offroad_Armor", (0.22, 0.28, 0.35, 1.0), roughness=0.38, metallic=0.12)
    # Sporty cyber coral-red accent
    mat_accent = get_or_create_material("Mat_Cyber_Accent", (0.92, 0.16, 0.14, 1.0), roughness=0.30, metallic=0.08)
    # Cyan cyber headlight LED
    mat_headlight = get_or_create_material("Mat_Cyber_Headlight", (0.35, 0.90, 1.0, 1.0), roughness=0.1, emission_color=(0.25, 0.85, 1.0, 1.0), emission_strength=22.0)
    # Red cyber taillight LED
    mat_taillight = get_or_create_material("Mat_Cyber_Taillight", (1.0, 0.05, 0.05, 1.0), roughness=0.15, emission_color=(1.0, 0.02, 0.02, 1.0), emission_strength=18.0)
    # Roll cage dark gunmetal steel
    mat_steel = get_or_create_material("Mat_RollCage_Steel", (0.24, 0.28, 0.34, 1.0), roughness=0.22, metallic=0.90)
    # Brass for structural standoffs
    mat_brass = bpy.data.materials.get("Mat_Brass") or get_or_create_material("Mat_Brass", (0.95, 0.78, 0.28, 1.0), roughness=0.2, metallic=1.0)
    # Dark tire rubber
    mat_tire = bpy.data.materials.get("Mat_TireRubber_PBR") or get_or_create_material("Mat_Tread_Lug", (0.025, 0.025, 0.028, 1.0), roughness=0.68)
    # Vibrant Yellow for wheel dish hubs (matching reference image!)
    mat_yellow = bpy.data.materials.get("Mat_YellowPlastic") or get_or_create_material("Mat_YellowPlastic", (0.98, 0.76, 0.02, 1.0), roughness=0.22)

    created_objects = []

    def register_obj(obj, mat=None):
        if mat:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
        if obj.name not in col.objects:
            col.objects.link(obj)
        if obj.name not in sc.objects:
            sc.objects.link(obj)
        created_objects.append(obj)
        return obj

    # ==========================================
    # A. FRONT BUMPER & SKID WEDGE
    # ==========================================
    bm_bump = bmesh.new()
    v0  = bm_bump.verts.new((0.0, 148.0, 36.0))    # Nose low center
    v1  = bm_bump.verts.new((48.0, 142.0, 36.0))
    v2  = bm_bump.verts.new((75.0, 130.0, 38.0))   # Nose low corner

    v3  = bm_bump.verts.new((0.0, 150.0, 50.0))    # Nose mid prow
    v4  = bm_bump.verts.new((50.0, 144.0, 50.0))
    v5  = bm_bump.verts.new((76.0, 132.0, 52.0))

    v6  = bm_bump.verts.new((0.0, 138.0, 62.0))    # Nose upper shelf
    v7  = bm_bump.verts.new((48.0, 134.0, 62.0))
    v8  = bm_bump.verts.new((74.0, 126.0, 60.0))

    v9  = bm_bump.verts.new((0.0, 120.0, 20.0))    # Skid plate bottom edge
    v10 = bm_bump.verts.new((45.0, 120.0, 20.0))
    v11 = bm_bump.verts.new((70.0, 120.0, 20.0))

    bm_bump.faces.new([v9, v10, v1, v0])
    bm_bump.faces.new([v10, v11, v2, v1])
    bm_bump.faces.new([v0, v1, v4, v3])
    bm_bump.faces.new([v1, v2, v5, v4])
    bm_bump.faces.new([v3, v4, v7, v6])
    bm_bump.faces.new([v4, v5, v8, v7])

    bmesh.ops.mirror(bm_bump, geom=bm_bump.verts[:] + bm_bump.edges[:] + bm_bump.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_bump, verts=bm_bump.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_bump, faces=bm_bump.faces)

    me_bump = bpy.data.meshes.new("Mesh_Offroad_Bumper")
    bm_bump.to_mesh(me_bump)
    bm_bump.free()
    obj_bump = bpy.data.objects.new("Front_Offroad_Bumper", me_bump)
    mod_sol = obj_bump.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.8
    mod_sol.offset = -1.0
    register_obj(obj_bump, mat_armor)

    # Cyan Cyber Headlights
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sign * 36.0, 144.5, 49.0))
        hl = bpy.context.active_object
        hl.name = f"Headlight_Cyber_{side}"
        hl.scale = (22.0, 4.0, 6.5)
        hl.rotation_euler = (0.0, 0.0, sign * math.radians(-14.0))
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_obj(hl, mat_headlight)

    # Front Bull-Bar Accent Trim
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 148.0, 36.0))
    bb_trim = bpy.context.active_object
    bb_trim.name = "Front_BullBar_Accent"
    bb_trim.scale = (42.0, 5.0, 4.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_obj(bb_trim, mat_accent)

    # ==========================================
    # B. GEOMETRIC FENDERS & SIDE ROCK SLIDERS
    # ==========================================
    for sign, side in [(-1, "L"), (1, "R")]:
        bm_side = bmesh.new()
        x_in  = sign * 72.0
        x_out = sign * 84.5
        x_lip = sign * 87.0

        # Front Fender (around Y=54, Wheel R=32.5, Top=65, Flare Top=77)
        ff_out = [
            (x_in,  96.0, 52.0),
            (x_out, 92.0, 55.0),
            (x_out, 76.0, 77.0),
            (x_out, 32.0, 77.0),
            (x_out, 16.0, 55.0),
            (x_in,  12.0, 52.0),
        ]
        ff_in = [
            (x_in,  96.0, 52.0),
            (x_in,  90.0, 64.0),
            (x_in,  74.0, 75.0),
            (x_in,  34.0, 75.0),
            (x_in,  18.0, 64.0),
            (x_in,  12.0, 52.0),
        ]
        v_ff_o = [bm_side.verts.new(p) for p in ff_out]
        v_ff_i = [bm_side.verts.new(p) for p in ff_in]
        for i in range(len(ff_out) - 1):
            bm_side.faces.new([v_ff_i[i], v_ff_i[i+1], v_ff_o[i+1], v_ff_o[i]])

        # Rear Fender (around Y=-54)
        rf_out = [
            (x_in,  -12.0, 52.0),
            (x_out, -16.0, 55.0),
            (x_out, -32.0, 77.0),
            (x_out, -76.0, 77.0),
            (x_out, -92.0, 55.0),
            (x_in,  -96.0, 52.0),
        ]
        rf_in = [
            (x_in,  -12.0, 52.0),
            (x_in,  -18.0, 64.0),
            (x_in,  -34.0, 75.0),
            (x_in,  -74.0, 75.0),
            (x_in,  -90.0, 64.0),
            (x_in,  -96.0, 52.0),
        ]
        v_rf_o = [bm_side.verts.new(p) for p in rf_out]
        v_rf_i = [bm_side.verts.new(p) for p in rf_in]
        for i in range(len(rf_out) - 1):
            bm_side.faces.new([v_rf_i[i], v_rf_i[i+1], v_rf_o[i+1], v_rf_o[i]])

        # Rock Slider Step (Y: 12 to -12)
        v_st_f_i = v_ff_i[5]
        v_st_f_o = v_ff_o[5]
        v_st_r_i = v_rf_i[0]
        v_st_r_o = v_rf_o[0]
        v_lip_f  = bm_side.verts.new((x_lip,  12.0, 48.0))
        v_lip_r  = bm_side.verts.new((x_lip, -12.0, 48.0))

        bm_side.faces.new([v_st_f_i, v_st_r_i, v_st_r_o, v_st_f_o])
        bm_side.faces.new([v_st_f_o, v_st_r_o, v_lip_r, v_lip_f])

        # Solid Lower Skirt Plate
        v_lp_bot_f = bm_side.verts.new((x_in,  12.0, 22.0))
        v_lp_bot_r = bm_side.verts.new((x_in, -12.0, 22.0))
        bm_side.faces.new([v_st_f_i, v_lp_bot_f, v_lp_bot_r, v_st_r_i])

        bmesh.ops.recalc_face_normals(bm_side, faces=bm_side.faces)
        me_side = bpy.data.meshes.new(f"Mesh_Fender_Armor_{side}")
        bm_side.to_mesh(me_side)
        bm_side.free()

        obj_side = bpy.data.objects.new(f"Fender_Armor_Side_{side}", me_side)
        mod_sol = obj_side.modifiers.new("Solidify", 'SOLIDIFY')
        mod_sol.thickness = 2.8
        mod_sol.offset = -1.0
        register_obj(obj_side, mat_armor)

        # Brass vertical structural standoffs under the slider
        for y_pos in [10.0, -10.0]:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=12,
                radius=3.2,
                depth=28.0,
                location=(sign * 78.0, y_pos, 35.0)
            )
            sp = bpy.context.active_object
            sp.name = f"Slider_Brass_Standoff_{side}_{y_pos:.0f}"
            register_obj(sp, mat_brass)

    # ==========================================
    # C. ARMORED PILLARS (A-PILLAR, B-PILLAR, C-PILLAR & ROOF SILLS)
    # ==========================================
    for sign, side in [(-1, "L"), (1, "R")]:
        bm_pil = bmesh.new()
        xr = sign * 48.0  # Roof width
        xf = sign * 72.0  # Fender / beltline width

        # 1. Raked A-Pillar (connecting front cowl to roof)
        va0 = bm_pil.verts.new((sign * 58.0, 70.0, 78.0))
        va1 = bm_pil.verts.new((sign * 66.0, 70.0, 78.0))
        va2 = bm_pil.verts.new((xr + sign * 6.0, 26.0, 116.0))
        va3 = bm_pil.verts.new((xr, 26.0, 116.0))
        bm_pil.faces.new([va0, va1, va2, va3])

        # 2. Longitudinal Roof Side Rail (connecting A-pillar, B-pillar, C-pillar)
        vr_f0 = va3
        vr_f1 = va2
        vr_r0 = bm_pil.verts.new((xr, -44.0, 116.0))
        vr_r1 = bm_pil.verts.new((xr + sign * 6.0, -44.0, 116.0))
        bm_pil.faces.new([vr_f0, vr_r0, vr_r1, vr_f1])

        # 3. Wide Upright B-Pillar Plate
        vb0 = bm_pil.verts.new((xf + sign * 2.0, -20.0, 52.0))
        vb1 = bm_pil.verts.new((xf + sign * 2.0,  -4.0, 52.0))
        vb2 = bm_pil.verts.new((xr + sign * 3.0,  -4.0, 116.0))
        vb3 = bm_pil.verts.new((xr + sign * 3.0, -20.0, 116.0))
        bm_pil.faces.new([vb0, vb1, vb2, vb3])

        # 4. Rear Quarter C-Pillar Panel with geometric window cutout
        vc_top_f = vr_r1
        vc_top_r = bm_pil.verts.new((xr + sign * 2.0, -68.0, 108.0))
        vc_mid_r = bm_pil.verts.new((xf + sign * 2.0, -96.0, 78.0))
        vc_mid_b = bm_pil.verts.new((xf + sign * 2.0, -122.0, 68.0))
        vc_bot_f = bm_pil.verts.new((xf + sign * 2.0, -44.0, 74.0))

        # Cutout window points
        vw_tf = bm_pil.verts.new((xr + sign * 1.5, -48.0, 106.0))
        vw_tr = bm_pil.verts.new((xr + sign * 1.5, -64.0, 102.0))
        vw_br = bm_pil.verts.new((xf + sign * 1.5, -82.0, 84.0))
        vw_bf = bm_pil.verts.new((xf + sign * 1.5, -48.0, 82.0))

        bm_pil.faces.new([vc_top_f, vc_top_r, vw_tr, vw_tf])
        bm_pil.faces.new([vc_top_r, vc_mid_r, vw_br, vw_tr])
        bm_pil.faces.new([vc_mid_r, vc_bot_f, vw_bf, vw_br])
        bm_pil.faces.new([vc_top_f, vw_tf, vw_bf, vc_bot_f])
        bm_pil.faces.new([vc_mid_r, vc_mid_b, bm_pil.verts.new((xf, -122.0, 52.0)), bm_pil.verts.new((xf, -96.0, 52.0))])

        bmesh.ops.recalc_face_normals(bm_pil, faces=bm_pil.faces)
        me_pil = bpy.data.meshes.new(f"Mesh_Cabin_Pillars_{side}")
        bm_pil.to_mesh(me_pil)
        bm_pil.free()

        obj_pil = bpy.data.objects.new(f"Cabin_Pillars_{side}", me_pil)
        mod_sol = obj_pil.modifiers.new("Solidify", 'SOLIDIFY')
        mod_sol.thickness = 2.6
        mod_sol.offset = -1.0
        register_obj(obj_pil, mat_armor)

    # ==========================================
    # D. HOOD, CAMERA PORT & RED INTAKE SCOOP
    # ==========================================
    bm_hood = bmesh.new()
    vh0 = bm_hood.verts.new((0.0, 126.0, 62.0))
    vh1 = bm_hood.verts.new((34.0, 124.0, 60.0))
    vh2 = bm_hood.verts.new((72.0, 120.0, 56.0))

    vh3 = bm_hood.verts.new((0.0, 104.0, 72.0))
    vh4 = bm_hood.verts.new((32.0, 102.0, 70.0))
    vh5 = bm_hood.verts.new((68.0, 100.0, 58.0))

    vh6 = bm_hood.verts.new((15.0, 84.0, 74.0))
    vh7 = bm_hood.verts.new((32.0, 84.0, 76.0))
    vh8 = bm_hood.verts.new((64.0, 84.0, 60.0))

    vh9  = bm_hood.verts.new((0.0, 70.0, 99.0))
    vh10 = bm_hood.verts.new((16.0, 70.0, 97.0))
    vh11 = bm_hood.verts.new((34.0, 70.0, 88.0))
    vh12 = bm_hood.verts.new((60.0, 70.0, 72.0))

    bm_hood.faces.new([vh0, vh1, vh4, vh3])
    bm_hood.faces.new([vh1, vh2, vh5, vh4])
    bm_hood.faces.new([vh3, vh4, vh7, vh6])
    bm_hood.faces.new([vh4, vh5, vh8, vh7])
    bm_hood.faces.new([vh6, vh7, vh11, vh10])
    bm_hood.faces.new([vh9, vh10, vh6, bm_hood.verts.new((0.0, 84.0, 74.0))])
    bm_hood.faces.new([vh7, vh8, vh12, vh11])

    bmesh.ops.mirror(bm_hood, geom=bm_hood.verts[:] + bm_hood.edges[:] + bm_hood.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_hood, verts=bm_hood.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_hood, faces=bm_hood.faces)

    me_hood = bpy.data.meshes.new("Mesh_Armored_Hood")
    bm_hood.to_mesh(me_hood)
    bm_hood.free()

    obj_hood = bpy.data.objects.new("Armored_Hood_Cowl", me_hood)
    mod_sol = obj_hood.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.6
    mod_sol.offset = -1.0
    register_obj(obj_hood, mat_armor)

    # Sporty Red Intake Scoop
    bm_scoop = bmesh.new()
    vs0 = bm_scoop.verts.new((0.0, 102.0, 74.0))
    vs1 = bm_scoop.verts.new((20.0, 100.0, 73.0))
    vs2 = bm_scoop.verts.new((24.0, 84.0, 82.0))
    vs3 = bm_scoop.verts.new((0.0, 84.0, 84.0))

    vs4 = bm_scoop.verts.new((0.0, 96.0, 85.0))
    vs5 = bm_scoop.verts.new((16.0, 94.0, 83.5))
    vs6 = bm_scoop.verts.new((18.0, 83.0, 91.0))
    vs7 = bm_scoop.verts.new((0.0, 83.0, 93.0))

    bm_scoop.faces.new([vs0, vs1, vs5, vs4])
    bm_scoop.faces.new([vs1, vs2, vs6, vs5])
    bm_scoop.faces.new([vs4, vs5, vs6, vs7])
    bmesh.ops.mirror(bm_scoop, geom=bm_scoop.verts[:] + bm_scoop.edges[:] + bm_scoop.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_scoop, verts=bm_scoop.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_scoop, faces=bm_scoop.faces)

    me_scoop = bpy.data.meshes.new("Mesh_Hood_Intake_Scoop")
    bm_scoop.to_mesh(me_scoop)
    bm_scoop.free()
    obj_scoop = bpy.data.objects.new("Hood_Intake_Scoop_Accent", me_scoop)
    register_obj(obj_scoop, mat_accent)

    # ==========================================
    # E. ROOF RACK WITH VENTILATED LOUVER BARS
    # ==========================================
    bm_slats = bmesh.new()
    for y_slat in range(-40, 26, 8):
        v0 = bm_slats.verts.new((-46.0, y_slat - 2.8, 115.5))
        v1 = bm_slats.verts.new(( 46.0, y_slat - 2.8, 115.5))
        v2 = bm_slats.verts.new(( 46.0, y_slat + 2.8, 117.8))
        v3 = bm_slats.verts.new((-46.0, y_slat + 2.8, 117.8))
        bm_slats.faces.new([v0, v1, v2, v3])

    bmesh.ops.recalc_face_normals(bm_slats, faces=bm_slats.faces)
    me_slats = bpy.data.meshes.new("Mesh_Roof_Louvers")
    bm_slats.to_mesh(me_slats)
    bm_slats.free()
    obj_slats = bpy.data.objects.new("Roof_Rack_Louvers", me_slats)
    mod_sol = obj_slats.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.0
    mod_sol.offset = -1.0
    register_obj(obj_slats, mat_armor)

    # Roll Cage Framework Bars
    cage_segments = []
    for sign in [-1, 1]:
        xr = sign * 48.0
        xf = sign * 58.0
        pts = [
            (Vector((xf, 70.0, 78.0)), Vector((xr, 26.0, 116.0))),   # A-pillar bar
            (Vector((xr, 26.0, 116.0)), Vector((xr, -44.0, 116.0))), # Roof rail bar
            (Vector((xr, -44.0, 116.0)), Vector((xf, -96.0, 76.0))), # C-pillar bar
            (Vector((xf, -96.0, 76.0)), Vector((xf, -125.0, 54.0))), # D-pillar bar
        ]
        for p_a, p_b in pts:
            vec = p_b - p_a
            bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=3.0, depth=vec.length, location=(0,0,0))
            c = bpy.context.active_object
            c.location = (p_a + p_b) / 2.0
            c.rotation_mode = 'QUATERNION'
            c.rotation_quaternion = Vector((0,0,1)).rotation_difference(vec.normalized())
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            cage_segments.append(c)

    # Front & Rear Crossbars
    for y_pos in [26.0, -44.0]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=6, radius=2.8, depth=96.0, location=(0.0, y_pos, 116.0))
        c = bpy.context.active_object
        c.rotation_euler = (0.0, math.radians(90.0), 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        cage_segments.append(c)

    bpy.ops.object.select_all(action='DESELECT')
    for c in cage_segments:
        c.select_set(True)
    bpy.context.view_layer.objects.active = cage_segments[0]
    bpy.ops.object.join()
    obj_exo = bpy.context.active_object
    obj_exo.name = "Exoskeleton_RollCage_Steel"
    register_obj(obj_exo, mat_steel)

    # ==========================================
    # F. REAR BATTERY BAY ARMOR & CYBER TAILLIGHTS
    # ==========================================
    bm_rear = bmesh.new()
    vr0  = bm_rear.verts.new((0.0, -70.0, 80.0))
    vr1  = bm_rear.verts.new((28.0, -70.0, 79.0))
    vr2  = bm_rear.verts.new((52.0, -70.0, 74.0))

    vr3  = bm_rear.verts.new((0.0, -112.0, 76.0))
    vr4  = bm_rear.verts.new((28.0, -112.0, 75.0))
    vr5  = bm_rear.verts.new((54.0, -112.0, 70.0))

    vr6  = bm_rear.verts.new((0.0, -128.0, 68.0))
    vr7  = bm_rear.verts.new((30.0, -128.0, 68.0))
    vr8  = bm_rear.verts.new((58.0, -128.0, 68.0))

    vr9  = bm_rear.verts.new((0.0, -128.0, 38.0))
    vr10 = bm_rear.verts.new((30.0, -128.0, 38.0))
    vr11 = bm_rear.verts.new((58.0, -128.0, 42.0))

    vr12 = bm_rear.verts.new((0.0, -118.0, 20.0))
    vr13 = bm_rear.verts.new((30.0, -118.0, 20.0))
    vr14 = bm_rear.verts.new((54.0, -118.0, 20.0))

    bm_rear.faces.new([vr0, vr1, vr4, vr3])
    bm_rear.faces.new([vr1, vr2, vr5, vr4])
    bm_rear.faces.new([vr3, vr4, vr7, vr6])
    bm_rear.faces.new([vr4, vr5, vr8, vr7])
    bm_rear.faces.new([vr6, vr7, vr10, vr9])
    bm_rear.faces.new([vr7, vr8, vr11, vr10])
    bm_rear.faces.new([vr9, vr10, vr13, vr12])
    bm_rear.faces.new([vr10, vr11, vr14, vr13])

    bmesh.ops.mirror(bm_rear, geom=bm_rear.verts[:] + bm_rear.edges[:] + bm_rear.faces[:], axis='X')
    bmesh.ops.remove_doubles(bm_rear, verts=bm_rear.verts, dist=0.1)
    bmesh.ops.recalc_face_normals(bm_rear, faces=bm_rear.faces)

    me_rear = bpy.data.meshes.new("Mesh_Rear_Battery_Armor")
    bm_rear.to_mesh(me_rear)
    bm_rear.free()
    obj_rear = bpy.data.objects.new("Rear_Battery_Bay_Armor", me_rear)
    mod_sol = obj_rear.modifiers.new("Solidify", 'SOLIDIFY')
    mod_sol.thickness = 2.6
    mod_sol.offset = -1.0
    register_obj(obj_rear, mat_armor)

    # Red Cyber Taillight Blocks
    for sign, side in [(-1, "L"), (1, "R")]:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(sign * 50.0, -130.0, 68.0))
        tl = bpy.context.active_object
        tl.name = f"Taillight_Cyber_{side}"
        tl.scale = (20.0, 8.0, 9.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_obj(tl, mat_taillight)

    # Rear Tow Accent Diffuser Bar
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, -130.0, 40.0))
    th = bpy.context.active_object
    th.name = "Rear_Tow_Diffuser_Accent"
    th.scale = (36.0, 5.0, 6.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    register_obj(th, mat_accent)

    # ==========================================
    # G. RUGGED OFF-ROAD PADDLE CLEATS & YELLOW DISH WHEEL HUBS
    # ==========================================
    wheel_positions = [
        ("Wheel_FL", -64.5,  54.0, 32.5, -1),
        ("Wheel_FR",  64.5,  54.0, 32.5,  1),
        ("Wheel_RL", -64.5, -54.0, 32.5, -1),
        ("Wheel_RR",  64.5, -54.0, 32.5,  1),
    ]

    r_outer = 32.5
    tread_h = 4.8
    tread_w = 23.0
    n_cleats = 14

    for name, wx, wy, wz, sign in wheel_positions:
        # 1. Paddle Cleats
        bm_cleat = bmesh.new()
        for i in range(n_cleats):
            angle = i * (2.0 * math.pi / n_cleats)
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
            
            cy = wy + sin_a * r_outer
            cz = wz + cos_a * r_outer
            ty = -cos_a
            tz = sin_a
            ny = sin_a
            nz = cos_a
            
            dx = tread_w / 2.0
            thick = 3.2
            
            v0 = bm_cleat.verts.new((wx - dx, cy - ty * thick, cz - tz * thick))
            v1 = bm_cleat.verts.new((wx + dx, cy - ty * thick, cz - tz * thick))
            v2 = bm_cleat.verts.new((wx + dx, cy + ty * thick, cz + tz * thick))
            v3 = bm_cleat.verts.new((wx - dx, cy + ty * thick, cz + tz * thick))
            
            tip_y = cy + ny * tread_h
            tip_z = cz + nz * tread_h
            v4 = bm_cleat.verts.new((wx - dx * 0.9, tip_y, tip_z))
            v5 = bm_cleat.verts.new((wx + dx * 0.9, tip_y, tip_z))
            
            bm_cleat.faces.new([v0, v1, v5, v4])
            bm_cleat.faces.new([v1, v2, v5])
            bm_cleat.faces.new([v2, v3, v4, v5])
            bm_cleat.faces.new([v3, v0, v4])
            bm_cleat.faces.new([v0, v3, v2, v1])

        bmesh.ops.recalc_face_normals(bm_cleat, faces=bm_cleat.faces)
        me_cleats = bpy.data.meshes.new(f"Mesh_Treads_{name}")
        bm_cleat.to_mesh(me_cleats)
        bm_cleat.free()
        obj_cleats = bpy.data.objects.new(f"Treads_{name}", me_cleats)
        register_obj(obj_cleats, mat_tire)

        # 2. Bold Yellow Dish Hub Disc on outer wheel face (like in reference image!)
        # Outer wheel face is at wx + sign * 12.0
        x_outer_face = wx + sign * 11.8
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=24,
            radius=20.5,
            depth=1.8,
            location=(x_outer_face, wy, wz)
        )
        hub = bpy.context.active_object
        hub.name = f"Wheel_HubCap_{name}"
        hub.rotation_euler = (0.0, math.radians(90.0), 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        register_obj(hub, mat_yellow)

    print("=== CYBER OFF-ROAD EXOSKELETON BUILD FULLY COMPLETE ===")
    print(f"Total objects registered: {len(created_objects)}")
    return created_objects

if __name__ == "__main__":
    create_offroad_chassis_and_exoskeleton()
