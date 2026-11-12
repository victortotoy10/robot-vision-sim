import re
import json

def parse_step(step_path):
    with open(step_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    points = {}
    for m in re.finditer(r'#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\)\s*\)', text):
        points[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

    vertices = {}
    for m in re.finditer(r'#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)', text):
        vertices[int(m.group(1))] = int(m.group(2))

    edge_curves = {}
    for m in re.finditer(r'#(\d+)\s*=\s*EDGE_CURVE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*\.([TF])\.\s*\)', text):
        edge_curves[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    oriented_edges = {}
    for m in re.finditer(r'#(\d+)\s*=\s*ORIENTED_EDGE\s*\(\s*\'[^\']*\'\s*,\s*\*\s*,\s*\*\s*,\s*#(\d+)\s*,\s*\.([TF])\.\s*\)', text):
        oriented_edges[int(m.group(1))] = (int(m.group(2)), m.group(3) == 'T')

    edge_loops = {}
    for m in re.finditer(r'#(\d+)\s*=\s*EDGE_LOOP\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)\s*\)', text):
        oes = [int(x.strip().replace('#', '')) for x in m.group(2).split(',') if x.strip()]
        edge_loops[int(m.group(1))] = oes

    face_bounds = {}
    for m in re.finditer(r'#(\d+)\s*=\s*FACE_(?:OUTER_)?BOUND\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*\.([TF])\.\s*\)', text):
        face_bounds[int(m.group(1))] = int(m.group(2))

    m_face = re.search(r'#4750\s*=\s*ADVANCED_FACE\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)', text)
    bounds = [int(x.strip().replace('#', '')) for x in m_face.group(1).split(',')]

    poly_loops = []
    for b in bounds:
        loop_id = face_bounds.get(b)
        if not loop_id or loop_id not in edge_loops:
            continue
        oes = edge_loops[loop_id]
        pts = []
        for oe_id in oes:
            if oe_id not in oriented_edges:
                continue
            ec_id, forward = oriented_edges[oe_id]
            if ec_id not in edge_curves:
                continue
            v_start, v_end = edge_curves[ec_id]
            v_chosen = v_start if forward else v_end
            pt_id = vertices.get(v_chosen)
            if pt_id and pt_id in points:
                p = points[pt_id]
                # In this STEP model: X is width, Z is length, Y is thickness
                pts.append((round(p[0], 3), round(p[2], 3)))
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            zs = [p[1] for p in pts]
            dx = max(xs) - min(xs)
            dz = max(zs) - min(zs)
            area = dx * dz
            poly_loops.append({
                'pts': pts,
                'bbox': [min(xs), max(xs), min(zs), max(zs)],
                'area': area,
                'count': len(pts)
            })

    poly_loops.sort(key=lambda it: it['area'], reverse=True)
    return poly_loops

if __name__ == '__main__':
    step_file = r'c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\ejemplo\Part3.printable2.STEP'
    loops = parse_step(step_file)
    print(f"Total loops parsed: {len(loops)}")
    print(f"Outer loop bbox: {loops[0]['bbox']} with {loops[0]['count']} points")
    
    # Save to json so Blender can use it directly!
    out_json = r'c:\Users\lenovo\Desktop\robot-vision-sim-main\diseñoCarroBlender\cad_loops.json'
    with open(out_json, 'w') as f:
        json.dump(loops, f, indent=2)
    print(f"Saved {len(loops)} loops to {out_json}")
