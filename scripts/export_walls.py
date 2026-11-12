import xml.etree.ElementTree as ET
import json
import os

dae_path = os.path.join(os.path.dirname(__file__), '..', 'meshes', 'walls.dae')
tree = ET.parse(dae_path)
root = tree.getroot()
ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}

pos_elem = root.find('.//c:float_array[@id="ID9_000-mesh-positions-array"]', ns)
pos_vals = [round(float(x), 4) for x in pos_elem.text.split()]
print(f'Total position floats: {len(pos_vals)} ({len(pos_vals)//3} vertices)')

triangles = root.findall('.//c:triangles', ns)
print(f'Total triangles elements: {len(triangles)}')

all_indices = []
for tri in triangles:
    p_elem = tri.find('c:p', ns)
    indices_raw = [int(x) for x in p_elem.text.split()]
    inputs = tri.findall('c:input', ns)
    stride = max(int(inp.attrib['offset']) for inp in inputs) + 1
    vertex_offset = 0
    for inp in inputs:
        if inp.attrib['semantic'] == 'VERTEX':
            vertex_offset = int(inp.attrib['offset'])
    
    verts = [indices_raw[i] for i in range(vertex_offset, len(indices_raw), stride)]
    all_indices.extend(verts)

print(f'Total vertex indices: {len(all_indices)} ({len(all_indices)//3} triangles)')

data = {
    'vertices': pos_vals,
    'indices': all_indices
}

out_path = os.path.join(os.path.dirname(__file__), '..', 'PaginaWebCarro', 'paredes_gazebo.js')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('export const paredesGazeboData = ' + json.dumps(data) + ';\n')

print(f'Exported {out_path} successfully!')

