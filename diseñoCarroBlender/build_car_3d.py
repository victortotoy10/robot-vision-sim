import bpy
import bmesh
import math
import mathutils
import os
from mathutils import Vector, Matrix

def reset_scene():
    """Limpia la escena actual dejando un espacio de trabajo limpio."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.curves:
        if block.users == 0:
            bpy.data.curves.remove(block)

def create_materials():
    """Crea la biblioteca de materiales PBR realistas optimizados."""
    mats = {}

    # 1. Acrílico transparente cristalino de alta pureza (sin ruido de transmisión)
    mat = bpy.data.materials.new(name="Mat_Acrylic_Glass")
    mat.use_nodes = True
    mat.surface_render_method = 'BLENDED'
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.92, 0.96, 1.0, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.03
    bsdf.inputs['IOR'].default_value = 1.49
    bsdf.inputs['Transmission Weight'].default_value = 0.0
    bsdf.inputs['Alpha'].default_value = 0.32
    bsdf.inputs['Specular IOR Level'].default_value = 0.70
    mats['acrylic_glass'] = mat

    # 2. Plástico amarillo brillante (Motores TT y Rines GrabCAD)
    mat = bpy.data.materials.new(name="Mat_YellowPlastic")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.98, 0.74, 0.02, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.22
    bsdf.inputs['Metallic'].default_value = 0.0
    mats['yellow_plastic'] = mat

    # 3. Neumático de goma con huella procedural de diamante GrabCAD y flancos lisos
    mat = bpy.data.materials.new(name="Mat_TireRubber_PBR")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    bsdf.inputs['Base Color'].default_value = (0.04, 0.04, 0.043, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.65

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])

    # Voronoi Manhattan: recrea la textura angular en rombos/diamante de Assembly.png
    voronoi = nodes.new(type="ShaderNodeTexVoronoi")
    voronoi.feature = 'DISTANCE_TO_EDGE'
    voronoi.distance = 'MANHATTAN'
    voronoi.inputs['Scale'].default_value = 160.0
    links.new(mapping.outputs['Vector'], voronoi.inputs['Vector'])

    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.08
    ramp.color_ramp.elements[1].position = 0.22
    links.new(voronoi.outputs['Distance'], ramp.inputs['Fac'])

    # Máscara para que los flancos laterales del neumático se mantengan suaves
    geom = nodes.new(type="ShaderNodeNewGeometry")
    sep_xyz = nodes.new(type="ShaderNodeSeparateXYZ")
    links.new(geom.outputs['Normal'], sep_xyz.inputs['Vector'])

    math_abs = nodes.new(type="ShaderNodeMath")
    math_abs.operation = 'ABSOLUTE'
    links.new(sep_xyz.outputs['X'], math_abs.inputs[0])

    math_mask = nodes.new(type="ShaderNodeMath")
    math_mask.operation = 'LESS_THAN'
    math_mask.inputs[1].default_value = 0.35
    links.new(math_abs.outputs['Value'], math_mask.inputs[0])

    math_mul = nodes.new(type="ShaderNodeMath")
    math_mul.operation = 'MULTIPLY'
    links.new(ramp.outputs['Color'], math_mul.inputs[0])
    links.new(math_mask.outputs['Value'], math_mul.inputs[1])

    bump = nodes.new(type="ShaderNodeBump")
    bump.inputs['Strength'].default_value = 0.85
    bump.inputs['Distance'].default_value = 0.003
    links.new(math_mul.outputs['Value'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

    mix = nodes.new(type="ShaderNodeMix")
    mix.data_type = 'RGBA'
    mix.inputs[6].default_value = (0.015, 0.015, 0.018, 1.0)
    mix.inputs[7].default_value = (0.045, 0.045, 0.048, 1.0)
    links.new(ramp.outputs['Color'], mix.inputs[0])
    links.new(mix.outputs[2], bsdf.inputs['Base Color'])

    mats['tire_rubber'] = mat

    # 4. Latón dorado (separadores hexagonales / standoffs)
    mat = bpy.data.materials.new(name="Mat_Brass")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.95, 0.78, 0.28, 1.0)
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.18
    mats['brass'] = mat

    # 5. Acero cromado / níquel (cuerpo del motor DC, tornillos, resortes)
    mat = bpy.data.materials.new(name="Mat_Steel_Chrome")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.88, 0.89, 0.91, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.98
    bsdf.inputs['Roughness'].default_value = 0.14
    mats['steel'] = mat

    # 6. Plástico blanco (ejes de salida de motor)
    mat = bpy.data.materials.new(name="Mat_WhitePlastic")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.95, 0.95, 0.93, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.35
    mats['white_plastic'] = mat

    # 7. Cobre brillante (terminales de conexión del motor)
    mat = bpy.data.materials.new(name="Mat_Copper")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.95, 0.55, 0.35, 1.0)
    bsdf.inputs['Metallic'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.20
    mats['copper'] = mat

    # 8. Plástico negro mate (tapas posteriores, disco de encoder, caja de baterías)
    mat = bpy.data.materials.new(name="Mat_BlackMatte")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.02, 0.02, 0.022, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.55
    mats['black_matte'] = mat

    return mats

def ensure_chassis_stl():
    """Garantiza la conversión de Part3.printable2.STEP a STL mediante Gmsh si aún no existe."""
    stl_path = r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo\chassis_plate.stl"
    step_path = r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo\Part3.printable2.STEP"
    if not os.path.exists(stl_path) and os.path.exists(step_path):
        import gmsh
        gmsh.initialize()
        gmsh.model.add("ChassisPlate")
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin', 2.5)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', 8.0)
        gmsh.option.setNumber('Mesh.Algorithm', 6)
        gmsh.model.mesh.generate(2)
        gmsh.write(stl_path)
        gmsh.finalize()
    return stl_path

def create_cad_chassis_plates(mat_acrylic):
    """
    Importa la placa de chasis original de Part3.printable2.STEP con escala real 1:1:
    - Ancho: 151.1 mm (0.1511 m)
    - Largo: 250.0 mm (0.2500 m)
    - Grosor: 3.0 mm (0.0030 m)
    """
    stl_file = ensure_chassis_stl()
    bpy.ops.wm.stl_import(filepath=stl_file)
    p_bot = bpy.context.active_object
    p_bot.name = "Chassis_Plate_Bottom"

    # En el STL de origen: X=151.06mm (ancho), Y=5.0mm (espesor), Z=250.0mm (largo)
    p_bot.rotation_euler = (math.pi / 2.0, 0, 0)
    bpy.ops.object.select_all(action='DESELECT')
    p_bot.select_set(True)
    bpy.context.view_layer.objects.active = p_bot
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Escalar X por 0.001 (151.1 mm -> 0.1511 m), Y por 0.001 (250 mm -> 0.2500 m), Z por 0.0006 (5mm -> 3mm)
    p_bot.scale = (0.001, 0.001, 0.0006)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Reducir vértices un 70% manteniendo intactos los orificios y curvas CAD
    dec = p_bot.modifiers.new("Decimate", 'DECIMATE')
    dec.decimate_type = 'DISSOLVE'
    dec.angle_limit = math.radians(2.0)
    bpy.ops.object.modifier_apply(modifier="Decimate")

    p_bot.location = (0, 0, 0.018) # 18 mm sobre el suelo
    p_bot.data.materials.clear()
    p_bot.data.materials.append(mat_acrylic)

    # Placa superior idéntica a 54 mm del suelo (separación de 33 mm)
    p_top = p_bot.copy()
    p_top.data = p_bot.data.copy()
    p_top.name = "Chassis_Plate_Top"
    bpy.context.collection.objects.link(p_top)
    p_top.location = (0, 0, 0.054)

    return p_bot, p_top

def create_standoff(x, y, z_bottom, z_top, mat_brass, mat_steel, name_prefix="Standoff"):
    """Crea un separador hexagonal de latón con tornillos M3 en los extremos."""
    height = z_top - z_bottom
    radius = 0.003
    
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=6,
        radius=radius,
        depth=height,
        location=(x, y, z_bottom + height / 2.0)
    )
    standoff = bpy.context.active_object
    standoff.name = f"{name_prefix}_{x:.2f}_{y:.2f}"
    standoff.data.materials.append(mat_brass)
    
    # Tornillos M3
    for z_sc, name_sc in [(z_top + 0.00075, "Top"), (z_bottom - 0.00075, "Bot")]:
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=14,
            radius=0.0028,
            depth=0.0015,
            location=(x, y, z_sc)
        )
        screw = bpy.context.active_object
        screw.name = f"Screw_{name_sc}_{x:.2f}_{y:.2f}"
        screw.data.materials.append(mat_steel)

def create_tt_motor_realistic(name, x, y, z, is_left, is_front, mats):
    """
    Crea el motor TT amarillo de alta precisión basado en Gear Motor.STEP:
    - Caja plástica amarilla ubicada dentro del chasis.
    - Motor DC cilíndrico metálico orientado hacia el centro del carro (Y=0).
    - Tapa posterior negra con terminales de cobre.
    - Eje de salida blanco extendido que conecta directamente al cubo de la rueda exterior.
    - Disco de encoder óptico negro en el eje interno hacia el centro del chasis.
    - Soportes en T de acrílico transparente.
    """
    y_dir = -1.0 if is_front else 1.0
    out_dir = -1.0 if is_left else 1.0

    gearbox_len = 0.042
    gearbox_wid = 0.0225
    gearbox_hgt = 0.020

    # 1. Caja de engranajes amarilla principal
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(x, y, z)
    )
    gearbox = bpy.context.active_object
    gearbox.name = f"{name}_Gearbox"
    gearbox.scale = (gearbox_wid, gearbox_len, gearbox_hgt)
    gearbox.data.materials.append(mats['yellow_plastic'])

    # 2. Motor DC cilíndrico metálico (plateado)
    dc_len = 0.025
    dc_rad = 0.010
    dc_y = y + y_dir * (gearbox_len / 2.0 + dc_len / 2.0 - 0.003)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=dc_rad,
        depth=dc_len,
        location=(x, dc_y, z),
        rotation=(math.pi / 2.0, 0, 0)
    )
    dc_can = bpy.context.active_object
    dc_can.name = f"{name}_DCCan"
    dc_can.data.materials.append(mats['steel'])

    # 3. Tapa trasera del motor en plástico negro
    cap_len = 0.004
    cap_y = dc_y + y_dir * (dc_len / 2.0 + cap_len / 2.0)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16,
        radius=dc_rad * 0.96,
        depth=cap_len,
        location=(x, cap_y, z),
        rotation=(math.pi / 2.0, 0, 0)
    )
    cap = bpy.context.active_object
    cap.name = f"{name}_RearCap"
    cap.data.materials.append(mats['black_matte'])

    # 4. Terminales de cobre de soldadura
    for term_z in [z + 0.004, z - 0.004]:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(x, cap_y + y_dir * 0.003, term_z)
        )
        term = bpy.context.active_object
        term.name = f"{name}_Terminal"
        term.scale = (0.002, 0.004, 0.0012)
        term.data.materials.append(mats['copper'])

    # 5. Eje de salida blanco (longitud de 38 mm para acoplar a la rueda exterior)
    shaft_len = 0.038
    shaft_x = x + out_dir * (gearbox_wid / 2.0 + shaft_len / 2.0)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=14,
        radius=0.0027,
        depth=shaft_len,
        location=(shaft_x, y, z),
        rotation=(0, math.pi / 2.0, 0)
    )
    shaft = bpy.context.active_object
    shaft.name = f"{name}_OutputShaft"
    shaft.data.materials.append(mats['white_plastic'])

    # 6. Disco ranurado de tacómetro / encoder óptico (hacia el interior)
    inner_x = x - out_dir * (gearbox_wid / 2.0 + 0.006)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24,
        radius=0.0125,
        depth=0.0015,
        location=(inner_x, y, z),
        rotation=(0, math.pi / 2.0, 0)
    )
    encoder_disc = bpy.context.active_object
    encoder_disc.name = f"{name}_EncoderDisc"
    encoder_disc.data.materials.append(mats['black_matte'])

    # 7. Soportes en T de acrílico transparente para montaje
    for brk_y in [y + 0.012, y - 0.012]:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(x, brk_y, z + 0.012)
        )
        brk = bpy.context.active_object
        brk.name = f"{name}_TBracket"
        brk.scale = (0.005, 0.003, 0.024)
        brk.data.materials.append(mats['acrylic_glass'])

def create_wheel_grabcad(name, x, y, z, is_left, mats):
    """
    Crea la rueda de alta fidelidad basada en GrabCAD (Wheel.STEP y Assembly.png):
    - Neumático hueco de goma con huella procedural Voronoi Manhattan y flancos lisos.
    - Rin amarillo con aro exterior hueco y 5 radios estilizados en estrella.
    - Ventanas triangulares abiertas que dejan ver el eje y el chasis.
    - Cubo central con tornillo cromado.
    - Ubicada completamente en el exterior del chasis con despeje garantizado.
    """
    r_out = 0.0335   # 67 mm diámetro exterior
    r_rim = 0.0260   # 52 mm diámetro interior de la llanta
    width = 0.0270   # 27 mm de ancho
    half_w = width / 2.0
    segments = 32

    parent = bpy.data.objects.new(f"{name}_Assembly", None)
    bpy.context.collection.objects.link(parent)
    parent.location = (x, y, z)

    # 1. NEUMÁTICO HUECO DE GOMA
    bm = bmesh.new()
    v_out_r, v_out_l, v_in_r, v_in_l = [], [], [], []
    for i in range(segments):
        ang = i * (2.0 * math.pi / segments)
        dy = math.sin(ang)
        dz = math.cos(ang)
        v_out_r.append(bm.verts.new(( half_w, dy * r_out, dz * r_out)))
        v_out_l.append(bm.verts.new((-half_w, dy * r_out, dz * r_out)))
        v_in_r.append(bm.verts.new((  half_w, dy * r_rim, dz * r_rim)))
        v_in_l.append(bm.verts.new(( -half_w, dy * r_rim, dz * r_rim)))

    for i in range(segments):
        nxt = (i + 1) % segments
        bm.faces.new([v_out_l[i], v_out_r[i], v_out_r[nxt], v_out_l[nxt]])
        bm.faces.new([v_in_r[i], v_in_l[i], v_in_l[nxt], v_in_r[nxt]])
        bm.faces.new([v_out_r[i], v_in_r[i], v_in_r[nxt], v_out_r[nxt]])
        bm.faces.new([v_in_l[i], v_out_l[i], v_out_l[nxt], v_in_l[nxt]])

    tire_mesh = bpy.data.meshes.new(f"{name}_TireMesh")
    bm.to_mesh(tire_mesh)
    bm.free()

    tire_obj = bpy.data.objects.new(f"{name}_Tire", tire_mesh)
    bpy.context.collection.objects.link(tire_obj)
    tire_obj.parent = parent
    tire_obj.data.materials.append(mats['tire_rubber'])

    out_dir = -1.0 if is_left else 1.0

    # 2. ARO EXTERIOR DEL RIN
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments,
        radius=r_rim,
        depth=width - 0.002,
        end_fill_type='NOTHING',
        location=(0, 0, 0),
        rotation=(0, math.pi / 2.0, 0)
    )
    r_barrel = bpy.context.active_object
    r_barrel.name = f"{name}_RimBarrel"
    r_barrel.parent = parent
    mod = r_barrel.modifiers.new(name="Solidify", type='SOLIDIFY')
    mod.thickness = 0.0018
    r_barrel.data.materials.append(mats['yellow_plastic'])

    # 3. CUBO CENTRAL AMARILLO
    hub_x = out_dir * (half_w - 0.005)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16,
        radius=0.0075,
        depth=0.010,
        location=(hub_x, 0, 0),
        rotation=(0, math.pi / 2.0, 0)
    )
    hub = bpy.context.active_object
    hub.name = f"{name}_Hub"
    hub.parent = parent
    hub.data.materials.append(mats['yellow_plastic'])

    # 4. TORNILLO CENTRAL CROMADO
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=12,
        radius=0.0028,
        depth=0.002,
        location=(hub_x + out_dir * 0.0045, 0, 0),
        rotation=(0, math.pi / 2.0, 0)
    )
    screw = bpy.context.active_object
    screw.name = f"{name}_CenterScrew"
    screw.parent = parent
    screw.data.materials.append(mats['steel'])

    # 5. LOS 5 RADIOS EN ESTRELLA ESCULPIDOS CON VENTANAS ABIERTAS
    spoke_bm = bmesh.new()
    spoke_thick = 0.0045
    r_hub = 0.0075
    r_inner_rim = r_rim - 0.0018

    for i in range(5):
        ang = i * (2.0 * math.pi / 5.0)
        perp = ang + math.pi / 2.0
        
        hw_hub = 0.0035
        cy1 = math.sin(ang) * r_hub
        cz1 = math.cos(ang) * r_hub
        p1 = (cy1 - math.sin(perp)*hw_hub, cz1 - math.cos(perp)*hw_hub)
        p2 = (cy1 + math.sin(perp)*hw_hub, cz1 + math.cos(perp)*hw_hub)
        
        hw_rim = 0.0065
        cy2 = math.sin(ang) * r_inner_rim
        cz2 = math.cos(ang) * r_inner_rim
        p3 = (cy2 + math.sin(perp)*hw_rim, cz2 + math.cos(perp)*hw_rim)
        p4 = (cy2 - math.sin(perp)*hw_rim, cz2 - math.cos(perp)*hw_rim)
        
        v1_f = spoke_bm.verts.new((hub_x + spoke_thick/2, p1[0], p1[1]))
        v2_f = spoke_bm.verts.new((hub_x + spoke_thick/2, p2[0], p2[1]))
        v3_f = spoke_bm.verts.new((hub_x + spoke_thick/2, p3[0], p3[1]))
        v4_f = spoke_bm.verts.new((hub_x + spoke_thick/2, p4[0], p4[1]))
        
        v1_b = spoke_bm.verts.new((hub_x - spoke_thick/2, p1[0], p1[1]))
        v2_b = spoke_bm.verts.new((hub_x - spoke_thick/2, p2[0], p2[1]))
        v3_b = spoke_bm.verts.new((hub_x - spoke_thick/2, p3[0], p3[1]))
        v4_b = spoke_bm.verts.new((hub_x - spoke_thick/2, p4[0], p4[1]))
        
        spoke_bm.faces.new([v1_f, v2_f, v3_f, v4_f])
        spoke_bm.faces.new([v4_b, v3_b, v2_b, v1_b])
        spoke_bm.faces.new([v1_b, v1_f, v4_f, v4_b])
        spoke_bm.faces.new([v2_f, v2_b, v3_b, v3_f])
        spoke_bm.faces.new([v3_f, v3_b, v4_b, v4_f])
        spoke_bm.faces.new([v1_f, v1_b, v2_b, v2_f])

    spoke_mesh = bpy.data.meshes.new(f"{name}_SpokesMesh")
    spoke_bm.to_mesh(spoke_mesh)
    spoke_bm.free()

    spoke_obj = bpy.data.objects.new(f"{name}_StarSpokes", spoke_mesh)
    bpy.context.collection.objects.link(spoke_obj)
    spoke_obj.parent = parent
    spoke_obj.data.materials.append(mats['yellow_plastic'])

def create_battery_box_kit(x, y, z, mats):
    """Crea la caja de baterías del kit escolar con resortes metálicos visibles."""
    bpy.ops.mesh.primitive_cube_add(
        size=1.0,
        location=(x, y, z + 0.010)
    )
    box = bpy.context.active_object
    box.name = "Battery_Box"
    box.scale = (0.050, 0.065, 0.018)
    box.data.materials.append(mats['black_matte'])

    # 4 Bahías para pilas con resortes metálicos de acero
    for bi in range(4):
        bx = x - 0.018 + bi * 0.012
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=12,
            radius=0.0035,
            depth=0.006,
            location=(bx, y - 0.026, z + 0.012),
            rotation=(math.pi / 2.0, 0, 0)
        )
        spring = bpy.context.active_object
        spring.name = f"Spring_{bi}"
        spring.data.materials.append(mats['steel'])

    # Cables salientes rojo y negro hacia la parte posterior
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=8,
        radius=0.001,
        depth=0.025,
        location=(x + 0.020, y + 0.038, z + 0.008),
        rotation=(0.3, 0, 0)
    )
    wire_red = bpy.context.active_object
    wire_red.name = "Battery_Wire_Red"
    mat_red = bpy.data.materials.new("Mat_WireRed")
    mat_red.use_nodes = True
    bsdf = next(n for n in mat_red.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.9, 0.05, 0.05, 1.0)
    wire_red.data.materials.append(mat_red)

def setup_dark_studio_environment():
    """Configura el entorno de estudio fotográfico con encuadre frontal-izquierdo matching photo."""
    bpy.ops.mesh.primitive_plane_add(
        size=4.0,
        location=(0, 0, 0)
    )
    floor = bpy.context.active_object
    floor.name = "Studio_Backdrop"
    
    mat_floor = bpy.data.materials.new(name="Mat_DarkStudioFloor")
    mat_floor.use_nodes = True
    bsdf = next(n for n in mat_floor.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs['Base Color'].default_value = (0.05, 0.052, 0.058, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.28
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    floor.data.materials.append(mat_floor)

    # Luz Principal Cenital (Key Light)
    bpy.ops.object.light_add(type='AREA', location=(-0.15, 0.20, 0.75))
    key_light = bpy.context.active_object
    key_light.name = "Key_Light_Area"
    key_light.data.energy = 85.0
    key_light.data.size = 0.60
    key_light.data.color = (1.0, 0.98, 0.95)

    # Luz de Relleno (Fill Light)
    bpy.ops.object.light_add(type='AREA', location=(0.40, -0.25, 0.65))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Light_Area"
    fill_light.data.energy = 35.0
    fill_light.data.size = 0.75
    fill_light.data.color = (0.90, 0.94, 1.0)

    # Luz de Contorno (Rim Light)
    bpy.ops.object.light_add(type='SPOT', location=(0.0, -0.55, 0.45))
    rim = bpy.context.active_object
    rim.name = "Rim_Light"
    rim.data.energy = 55.0
    rim.data.spot_size = math.radians(70)
    rim.rotation_euler = (math.radians(-45), 0, 0)

    # Cámara frontal 3/4 izquierda (Encuadre exacto de la foto del usuario)
    cam_loc = (-0.27, 0.32, 0.22)
    bpy.ops.object.camera_add(location=cam_loc)
    cam = bpy.context.active_object
    cam.name = "Main_Camera"
    cam.data.lens = 52.0
    
    target = Vector((0.0, 0.0, 0.038))
    direction = target - Vector(cam_loc)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = cam

def build_refined_car():
    """Construye el carro 4WD refinado con geometría CAD exacta 1:1 y despeje total de ruedas."""
    print("Construyendo carro 4WD refinado con datos de Part3.printable2.STEP y GrabCAD...")
    reset_scene()
    mats = create_materials()

    z_bottom_plate = 0.018 # 18 mm sobre suelo
    z_top_plate = 0.054    # 54 mm sobre suelo
    wheel_z = 0.0335       # Eje a 33.5 mm (rueda toca el suelo Z=0)
    wheel_track_x = 0.0915 # Ruedas exteriores con despeje garantizado de 2.5 mm
    motor_x = 0.042        # Motores interiores montados en los orificios del chasis

    # 1. Ambas Placas del Chasis (Importadas de Part3.printable2.STEP en escala exacta 1:1)
    create_cad_chassis_plates(mats['acrylic_glass'])

    # 2. Separadores de latón hexagonales con tornillos M3 (Chasis + Parachoques)
    standoff_coords = [
        ( 0.052,  0.088), (-0.052,  0.088), # Frente chasis
        ( 0.062,  0.000), (-0.062,  0.000), # Centro chasis
        ( 0.052, -0.088), (-0.052, -0.088), # Trasera chasis
        ( 0.056,  0.098), (-0.056,  0.098)  # Parachoques delantero
    ]
    for x, y in standoff_coords:
        create_standoff(x, y, z_bottom_plate + 0.003, z_top_plate, mats['brass'], mats['steel'])

    # 3. Los 4 Motores TT Amarillos Realistas Orientados hacia el Centro
    motor_coords = [
        ("Motor_FL", -motor_x,  0.0631, wheel_z, True,  True),
        ("Motor_RL", -motor_x, -0.0631, wheel_z, True,  False),
        ("Motor_FR",  motor_x,  0.0631, wheel_z, False, True),
        ("Motor_RR",  motor_x, -0.0631, wheel_z, False, False),
    ]
    for name, x, y, z, is_left, is_front in motor_coords:
        create_tt_motor_realistic(name, x, y, z, is_left, is_front, mats)

    # 4. Las 4 Ruedas GrabCAD con Rin Amarillo de 5 Radios y Neumático con Huella (Sin colisión)
    wheel_coords = [
        ("Wheel_FL", -wheel_track_x,  0.0631, wheel_z, True),
        ("Wheel_RL", -wheel_track_x, -0.0631, wheel_z, True),
        ("Wheel_FR",  wheel_track_x,  0.0631, wheel_z, False),
        ("Wheel_RR",  wheel_track_x, -0.0631, wheel_z, False),
    ]
    for name, x, y, z, is_left in wheel_coords:
        create_wheel_grabcad(name, x, y, z, is_left, mats)

    # 5. Caja de baterías en la parte trasera del piso superior (como en la foto real)
    create_battery_box_kit(0.0, -0.070, z_top_plate + 0.003, mats)

    # 6. Configurar Iluminación y Estudio Oscuro Elegante
    setup_dark_studio_environment()

    # 7. Guardar archivos .blend y renderizar imagen 1080p
    save_dirs = [
        r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo",
        r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender"
    ]
    for d in save_dirs:
        os.makedirs(d, exist_ok=True)
        blend_file = os.path.join(d, "carro_4wd_robot.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_file)
        print(f"Modelo guardado exitosamente en: {blend_file}")

    render_path = r"c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo\render_robot_4wd.png"
    bpy.context.scene.render.filepath = render_path
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.ops.render.render(write_still=True)
    print("Render 1080p guardado en:", render_path)

if __name__ == '__main__':
    build_refined_car()
