import bpy
import bmesh
import math
from mathutils import Vector, Matrix, Euler

def build_porsche_911_pro():
    # 1. Clean collection
    car_col = bpy.data.collections.get("Carrocería")
    if not car_col:
        car_col = bpy.data.collections.new("Carrocería")
        bpy.context.scene.collection.children.link(car_col)
    
    for obj in list(car_col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
        
    for m in list(bpy.data.meshes):
        if any(k in m.name.lower() for k in ["carroceria", "porsche", "faros", "pilotos", "bisel", "cutter"]):
            bpy.data.meshes.remove(m)

    # 2. Materials
    mat_blue = bpy.data.materials.get("Mat_Porsche_911_Blue_PLA")
    if not mat_blue:
        mat_blue = bpy.data.materials.new(name="Mat_Porsche_911_Blue_PLA")
        mat_blue.use_nodes = True
    
    nodes = mat_blue.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    # Vibrant rich blue matching user's PLA photo:
    bsdf.inputs['Base Color'].default_value = (0.04, 0.28, 0.88, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.38
    if 'Metallic' in bsdf.inputs:
        bsdf.inputs['Metallic'].default_value = 0.0

    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    node_out.location = (300, 0)
    mat_blue.node_tree.links.new(bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    # Headlight lens material
    mat_lens = bpy.data.materials.get("Mat_Porsche_Headlight")
    if not mat_lens:
        mat_lens = bpy.data.materials.new(name="Mat_Porsche_Headlight")
        mat_lens.use_nodes = True
        bsdf_l = [n for n in mat_lens.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        bsdf_l.inputs['Base Color'].default_value = (0.85, 0.92, 1.0, 1.0)
        bsdf_l.inputs['Roughness'].default_value = 0.15

    # 3. Build Base Continuous Porsche 911 Shell (Right Half, X >= 0)
    bm = bmesh.new()

    topo = [
        # 0: Y = 124 (Nose tip)
        {"L0": (0.0, 124.0, 62.0), "L1": (14.0, 124.0, 61.5), "L2": (30.0, 124.0, 60.0), "L3": (46.0, 124.0, 57.0), "L4": (58.0, 124.0, 53.5), "L5": (52.0, 124.0, 50.0)},
        # 1: Y = 116 (Valance)
        {"L0": (0.0, 116.0, 68.0), "L1": (17.0, 116.0, 67.5), "L2": (35.0, 116.0, 66.5), "L3": (52.0, 116.0, 62.0), "L4": (66.0, 116.0, 55.0), "L5": (60.0, 116.0, 50.0)},
        # 2: Y = 106 (Headlights brow)
        {"L0": (0.0, 106.0, 77.0), "L1": (19.0, 106.0, 76.5), "L2": (39.0, 106.0, 84.5), "L3": (56.0, 106.0, 80.5), "L4": (75.0, 106.0, 66.0), "L5": (68.0, 106.0, 50.0)},
        
        # 3: Y = 90 (Front arch front curve, >18mm clearance)
        {"L0": (0.0, 90.0, 84.5), "L1": (21.0, 90.0, 84.0), "L2": (41.0, 90.0, 87.5), "L3": (63.0, 90.0, 84.0), "L4": (88.0, 90.0, 72.0), "L5": None},
        # 4: Y = 76 (Cowl / Camera port)
        {"L0": (0.0, 76.0, 89.5), "L1": (22.0, 76.0, 89.0), "L2": (42.0, 76.0, 89.5), "L3": (66.0, 76.0, 86.0), "L4": (91.0, 76.0, 79.0), "L5": None},
        # 5: Y = 54 (Front axle apex / safari flare peak)
        {"L0": (0.0, 54.0, 97.0), "L1": (24.0, 54.0, 96.0), "L2": (42.0, 54.0, 102.0), "L3": (68.0, 54.0, 88.0), "L4": (93.5, 54.0, 84.0), "L5": None},
        # 6: Y = 32 (A-pillar / arch rear curve)
        {"L0": (0.0, 32.0, 115.0), "L1": (24.0, 32.0, 114.0), "L2": (40.0, 32.0, 115.0), "L3": (64.0, 32.0, 86.0), "L4": (91.0, 32.0, 79.0), "L5": None},
        # 7: Y = 18 (Roof front header)
        {"L0": (0.0, 18.0, 123.5), "L1": (24.0, 18.0, 122.5), "L2": (38.0, 18.0, 121.5), "L3": (56.0, 18.0, 85.0), "L4": (86.0, 18.0, 73.0), "L5": None},

        # 8: Y = 0 (Center crown / Rocker sill mid)
        {"L0": (0.0, 0.0, 126.0), "L1": (24.0, 0.0, 125.0), "L2": (38.0, 0.0, 123.5), "L3": (55.0, 0.0, 85.0), "L4": (75.0, 0.0, 66.0), "L5": (72.0, 0.0, 50.0)},

        # 9: Y = -18 (Roof rear header)
        {"L0": (0.0, -18.0, 123.5), "L1": (24.0, -18.0, 122.5), "L2": (38.0, -18.0, 121.5), "L3": (56.0, -18.0, 85.0), "L4": (86.0, -18.0, 73.0), "L5": None},
        # 10: Y = -34 (Fastback start / C-pillar)
        {"L0": (0.0, -34.0, 116.0), "L1": (24.0, -34.0, 115.0), "L2": (40.0, -34.0, 114.0), "L3": (64.0, -34.0, 86.0), "L4": (92.0, -34.0, 79.0), "L5": None},
        # 11: Y = -54 (Rear axle apex / rear flare peak)
        {"L0": (0.0, -54.0, 106.0), "L1": (24.0, -54.0, 105.0), "L2": (42.0, -54.0, 102.0), "L3": (68.0, -54.0, 88.0), "L4": (95.0, -54.0, 85.0), "L5": None},
        # 12: Y = -76 (Fastback slope / battery cover)
        {"L0": (0.0, -76.0, 94.0), "L1": (23.0, -76.0, 93.0), "L2": (42.0, -76.0, 91.0), "L3": (66.0, -76.0, 86.0), "L4": (92.0, -76.0, 79.0), "L5": None},
        # 13: Y = -92 (Rear decklid)
        {"L0": (0.0, -92.0, 86.0), "L1": (22.0, -92.0, 85.0), "L2": (41.0, -92.0, 83.0), "L3": (63.0, -92.0, 80.0), "L4": (87.0, -92.0, 72.0), "L5": None},
        
        # 14: Y = -110 (Rear light bar & ducktail lip)
        {"L0": (0.0, -110.0, 77.0), "L1": (20.0, -110.0, 76.0), "L2": (38.0, -110.0, 74.0), "L3": (55.0, -110.0, 71.0), "L4": (68.0, -110.0, 62.0), "L5": (65.0, -110.0, 50.0)},
        # 15: Y = -124 (Rear bumper terminus)
        {"L0": (0.0, -124.0, 68.0), "L1": (17.0, -124.0, 67.0), "L2": (34.0, -124.0, 65.0), "L3": (49.0, -124.0, 62.0), "L4": (60.0, -124.0, 56.0), "L5": (56.0, -124.0, 50.0)}
    ]

    v_grid = []
    for s_idx, d in enumerate(topo):
        row = {}
        for key in ["L0", "L1", "L2", "L3", "L4"]:
            row[key] = bm.verts.new(d[key])
        row["L5"] = bm.verts.new(d["L5"]) if d["L5"] else None
        v_grid.append(row)

    bm.verts.ensure_lookup_table()

    # Build 100% continuous solid skin for half the car:
    for s in range(len(topo) - 1):
        r1, r2 = v_grid[s], v_grid[s+1]
        # L0 -> L1 (centerline strip)
        bm.faces.new([r1["L0"], r1["L1"], r2["L1"], r2["L0"]])
        # L1 -> L2 (inner strip)
        bm.faces.new([r1["L1"], r1["L2"], r2["L2"], r2["L1"]])
        # L2 -> L3 (brow / pillar / roof rail strip)
        bm.faces.new([r1["L2"], r1["L3"], r2["L3"], r2["L2"]])
        # L3 -> L4 (shoulder / flare strip)
        bm.faces.new([r1["L3"], r1["L4"], r2["L4"], r2["L3"]])
        # L4 -> L5 (sill strip outside wheels)
        if r1["L5"] and r2["L5"]:
            bm.faces.new([r1["L4"], r1["L5"], r2["L5"], r2["L4"]])
        elif r1["L5"] and not r2["L5"]:
            bm.faces.new([r1["L4"], r1["L5"], r2["L4"]])
        elif not r1["L5"] and r2["L5"]:
            bm.faces.new([r1["L4"], r2["L5"], r2["L4"]])

    # Nose closing
    v_front_lip = bm.verts.new((0.0, 124.0, 50.0))
    r0 = v_grid[0]
    bm.faces.new([r0["L0"], r0["L1"], v_front_lip])
    bm.faces.new([r0["L1"], r0["L2"], v_front_lip])
    bm.faces.new([r0["L2"], r0["L3"], v_front_lip])
    bm.faces.new([r0["L3"], r0["L4"], v_front_lip])
    bm.faces.new([r0["L4"], r0["L5"], v_front_lip])

    # Rear closing
    v_rear_lip = bm.verts.new((0.0, -124.0, 50.0))
    r15 = v_grid[15]
    bm.faces.new([r15["L0"], v_rear_lip, r15["L1"]])
    bm.faces.new([r15["L1"], v_rear_lip, r15["L2"]])
    bm.faces.new([r15["L2"], v_rear_lip, r15["L3"]])
    bm.faces.new([r15["L3"], v_rear_lip, r15["L4"]])
    bm.faces.new([r15["L4"], v_rear_lip, r15["L5"]])

    bm.normal_update()

    mesh_base = bpy.data.meshes.new(name="Porsche_911_Base_Skin")
    bm.to_mesh(mesh_base)
    bm.free()

    car_obj = bpy.data.objects.new(name="Porsche_911_Carroceria_4WD", object_data=mesh_base)
    car_col.objects.link(car_obj)
    car_obj.data.materials.append(mat_blue)

    # 4. Mirror Modifier to get full symmetrical body
    mod_mir = car_obj.modifiers.new(name="Mirror_Porsche", type='MIRROR')
    mod_mir.use_axis[0] = True
    mod_mir.use_clip = True
    mod_mir.merge_threshold = 0.5

    # Apply mirror modifier
    bpy.context.view_layer.objects.active = car_obj
    bpy.ops.object.modifier_apply(modifier="Mirror_Porsche")

    # 5. Create Precise CAD Cutters:

    def make_cube_cutter(name, loc, size):
        bm_c = bmesh.new()
        bmesh.ops.create_cube(bm_c, size=1.0)
        # Scale to size
        for v in bm_c.verts:
            v.co.x *= size[0]
            v.co.y *= size[1]
            v.co.z *= size[2]
        m_c = bpy.data.meshes.new(name)
        bm_c.to_mesh(m_c)
        bm_c.free()
        obj_c = bpy.data.objects.new(name, m_c)
        obj_c.location = loc
        car_col.objects.link(obj_c)
        return obj_c

    # Cutter 1: Windshield Opening
    # Sloping rectangular prism cutting out the windshield glass area:
    # Cowl is at Y=76, Z=89. Roof header is at Y=22, Z=123.
    # Leaves solid 8.5mm A-pillars on sides (width X = +-33.0 mm)
    c_ws = make_cube_cutter("Cutter_Windshield", (0.0, 48.0, 106.0), (66.0, 48.0, 34.0))
    # Tilted back ~35 degrees matching windshield rake
    c_ws.rotation_euler = Euler((math.radians(35.0), 0.0, 0.0), 'XYZ')

    # Cutter 2: Sunroof Opening
    # Centered on the roof: width 44mm (X in [-22, 22]), length 30mm (Y in [-15, 15]), height 20mm
    c_sr = make_cube_cutter("Cutter_Sunroof", (0.0, 0.0, 125.0), (44.0, 30.0, 20.0))

    # Cutter 3: Camera Port on Hood
    # Cylinder dia 10.0 mm centered at (0, 80.5, 88.5) along Y
    bm_cam = bmesh.new()
    bmesh.ops.create_cone(bm_cam, cap_ends=True, cap_tris=False, segments=20, radius1=5.0, radius2=5.0, depth=40.0)
    # Rotate cylinder to point along Y
    for v in bm_cam.verts:
        # Swap Y and Z
        v.co = Vector((v.co.x, v.co.z, v.co.y))
    m_cam = bpy.data.meshes.new("Cutter_CamPort_Mesh")
    bm_cam.to_mesh(m_cam)
    bm_cam.free()
    c_cam = bpy.data.objects.new("Cutter_CameraPort", m_cam)
    c_cam.location = (0.0, 80.5, 88.5)
    car_col.objects.link(c_cam)

    # Cutter 4: Side Windows (Teardrop shape)
    # Right side window cutter
    c_sw_r = make_cube_cutter("Cutter_SideWindow_R", (48.0, 0.0, 104.0), (25.0, 52.0, 28.0))
    # Left side window cutter
    c_sw_l = make_cube_cutter("Cutter_SideWindow_L", (-48.0, 0.0, 104.0), (25.0, 52.0, 28.0))

    # Cutter 5: Lower Front Bumper Center Intake
    c_intake_ctr = make_cube_cutter("Cutter_Intake_Ctr", (0.0, 122.0, 54.0), (32.0, 15.0, 6.0))
    # Lower Front Bumper Side Intakes
    c_intake_r = make_cube_cutter("Cutter_Intake_R", (36.0, 120.0, 54.0), (16.0, 15.0, 6.0))
    c_intake_l = make_cube_cutter("Cutter_Intake_L", (-36.0, 120.0, 54.0), (16.0, 15.0, 6.0))

    # Apply Booleans sequentially
    cutters = [c_ws, c_sr, c_cam, c_sw_r, c_sw_l, c_intake_ctr, c_intake_r, c_intake_l]
    for idx, cutter in enumerate(cutters):
        mod_bool = car_obj.modifiers.new(name=f"Bool_{idx}", type='BOOLEAN')
        mod_bool.object = cutter
        mod_bool.operation = 'DIFFERENCE'
        mod_bool.solver = 'EXACT'
        bpy.context.view_layer.objects.active = car_obj
        bpy.ops.object.modifier_apply(modifier=f"Bool_{idx}")
        # Remove cutter object
        bpy.data.objects.remove(cutter, do_unlink=True)

    # 6. Add Sculpted Porsche 911 Oval Headlights
    # Right & Left headlights on the raised fender brows
    bm_hl = bmesh.new()
    for sign_x in [1.0, -1.0]:
        hl_center = Vector((sign_x * 39.0, 106.0, 84.5))
        hl_segs = 16
        hl_rx, hl_ry = 8.5, 12.0
        # Oval bezel ring
        ring_v = []
        for i in range(hl_segs):
            ang = 2.0 * math.pi * i / hl_segs
            lx = hl_rx * math.sin(ang)
            ly = hl_ry * math.cos(ang) * math.cos(math.radians(-22))
            lz = hl_ry * math.cos(ang) * math.sin(math.radians(-22))
            ring_v.append(bm_hl.verts.new(hl_center + Vector((sign_x * lx, ly, lz + 1.8))))
        # Lens apex
        apex = bm_hl.verts.new(hl_center + Vector((0.0, 0.0, 3.8)))
        for i in range(hl_segs):
            i_next = (i + 1) % hl_segs
            bm_hl.faces.new([ring_v[i], ring_v[i_next], apex])

    m_hl = bpy.data.meshes.new("Porsche_Headlights_Mesh")
    bm_hl.to_mesh(m_hl)
    bm_hl.free()
    hl_obj = bpy.data.objects.new("Porsche_Headlights", m_hl)
    car_col.objects.link(hl_obj)
    hl_obj.data.materials.append(mat_lens)

    # Join headlights into car body
    bpy.context.view_layer.objects.active = car_obj
    hl_obj.select_set(True)
    car_obj.select_set(True)
    bpy.ops.object.join()

    # 7. Add 4 Internal Mounting Bosses with 3.2mm holes
    # Standoffs: Front (X = +-55.60, Y = 99.96), Center (X = +-61.56, Y = 0.00)
    bm_tab = bmesh.new()
    mount_points = [
        ( 55.60, 99.96),
        (-55.60, 99.96),
        ( 61.56,  0.00),
        (-61.56,  0.00)
    ]
    tab_r_in = 1.6 # 3.2mm hole
    tab_r_out = 4.5 # 9.0mm boss
    tab_z_bot = 52.0
    tab_z_top = 55.0
    tab_segs = 16

    for mx, my in mount_points:
        ring_b_in, ring_b_out = [], []
        ring_t_in, ring_t_out = [], []
        for i in range(tab_segs):
            a = 2.0 * math.pi * i / tab_segs
            dx_in, dy_in = tab_r_in * math.cos(a), tab_r_in * math.sin(a)
            dx_out, dy_out = tab_r_out * math.cos(a), tab_r_out * math.sin(a)
            ring_b_in.append(bm_tab.verts.new((mx + dx_in, my + dy_in, tab_z_bot)))
            ring_b_out.append(bm_tab.verts.new((mx + dx_out, my + dy_out, tab_z_bot)))
            ring_t_in.append(bm_tab.verts.new((mx + dx_in, my + dy_in, tab_z_top)))
            ring_t_out.append(bm_tab.verts.new((mx + dx_out, my + dy_out, tab_z_top)))

        for i in range(tab_segs):
            i_next = (i + 1) % tab_segs
            bm_tab.faces.new([ring_t_in[i], ring_t_in[i_next], ring_t_out[i_next], ring_t_out[i]])
            bm_tab.faces.new([ring_b_in[i], ring_b_out[i], ring_b_out[i_next], ring_b_in[i_next]])
            bm_tab.faces.new([ring_b_in[i], ring_t_in[i], ring_t_in[i_next], ring_b_in[i_next]])
            bm_tab.faces.new([ring_b_out[i], ring_b_out[i_next], ring_t_out[i_next], ring_t_out[i]])

        # Structural rib to inner shell
        sign_x = 1.0 if mx > 0 else -1.0
        rib_v1 = bm_tab.verts.new((mx + sign_x * tab_r_out, my - 2.0, tab_z_bot))
        rib_v2 = bm_tab.verts.new((mx + sign_x * tab_r_out, my + 2.0, tab_z_bot))
        rib_v3 = bm_tab.verts.new((mx + sign_x * 12.0, my + 2.0, tab_z_top + 10.0))
        rib_v4 = bm_tab.verts.new((mx + sign_x * 12.0, my - 2.0, tab_z_top + 10.0))
        bm_tab.faces.new([rib_v1, rib_v2, rib_v3, rib_v4])

    m_tab = bpy.data.meshes.new("Porsche_Tabs_Mesh")
    bm_tab.to_mesh(m_tab)
    bm_tab.free()
    tab_obj = bpy.data.objects.new("Porsche_Tabs", m_tab)
    car_col.objects.link(tab_obj)
    tab_obj.data.materials.append(mat_blue)

    bpy.context.view_layer.objects.active = car_obj
    tab_obj.select_set(True)
    car_obj.select_set(True)
    bpy.ops.object.join()

    # 8. Solidify Modifier: Uniform 2.8 mm wall thickness
    mod_sol = car_obj.modifiers.new(name="Solidify_3DPrint", type='SOLIDIFY')
    mod_sol.thickness = 2.8
    mod_sol.offset = -1.0 # Inward
    mod_sol.use_even_offset = True
    mod_sol.use_quality_normals = True

    # 9. Subdivision Surface
    mod_sub = car_obj.modifiers.new(name="Subsurf_Curves", type='SUBSURF')
    mod_sub.levels = 1
    mod_sub.render_levels = 2

    # Smooth shading
    for poly in car_obj.data.polygons:
        poly.use_smooth = True

    print(f"Successfully generated PRO CAD Porsche 911 body: {car_obj.name}")
    print(f"  Dimensions: {car_obj.dimensions}")
    return car_obj

build_porsche_911_pro()
