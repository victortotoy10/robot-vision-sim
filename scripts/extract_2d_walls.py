import json
import math

with open('PaginaWebCarro/paredes_gazebo.js', 'r', encoding='utf-8') as f:
    txt = f.read()

idx = txt.find('{')
end_idx = txt.rfind('}')
wall_data = json.loads(txt[idx:end_idx+1])
verts = wall_data['vertices']
indices = wall_data['indices']

# Each vertex has X, Height(collada Z), Z(collada -Y)
# In Three.js:
# Three.js X = verts[3*i]
# Three.js Y = verts[3*i + 1] (height: 0.0 to 0.3)
# Three.js Z = -verts[3*i + 1] -> wait! In export_walls.py:
# verts was [x, y, z] from Collada
# In Three.js: X = raw X, Y = raw Z, Z = -raw Y.

print(f"Total triangles: {len(indices) // 3}")

# Let's extract all edges from the triangles that form vertical wall faces
# A vertical wall face has triangles where the normal is horizontal (i.e. normal.Y == 0)
# Or we can simply extract edges where Y goes from bottom (0.0) to top (0.3).

vertical_segments = []
for t in range(0, len(indices), 3):
    i0 = indices[t]
    i1 = indices[t+1]
    i2 = indices[t+2]
    
    # Coordinates in Three.js
    p0 = (verts[3*i0], -verts[3*i0+1], verts[3*i0+2]) # (X, Z, Height)
    p1 = (verts[3*i1], -verts[3*i1+1], verts[3*i1+2])
    p2 = (verts[3*i2], -verts[3*i2+1], verts[3*i2+2])
    
    # Triangle normal
    v01 = (p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2])
    v02 = (p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2])
    
    nx = v01[1]*v02[2] - v01[2]*v02[1]
    ny = v01[2]*v02[0] - v01[0]*v02[2] # In Three.js this is horizontal Z vs Height
    # In 2D plane (X, Z):
    # A wall is vertical if Height changes.
    # Bottom vertices have Height near 0, Top vertices have Height near 0.3.
    # The 2D footprint of the wall is the segment connecting bottom vertices!
    pass

# Let's find distinct bottom vertices (Height < 0.05)
bottom_pts = {}
for i in range(len(verts) // 3):
    gx = round(verts[3*i], 4)
    gz = round(-verts[3*i+1], 4) # Three.js Z
    gy = verts[3*i+2] # Height
    if gy < 0.05:
        bottom_pts[i] = (gx, gz)

print(f"Bottom vertices (Height < 0.05): {len(bottom_pts)}")

# Find edges between bottom vertices that belong to vertical faces
# A triangle has a bottom edge if two of its vertices are bottom vertices, and one is top!
wall_2d_edges = set()
for t in range(0, len(indices), 3):
    tri = [indices[t], indices[t+1], indices[t+2]]
    b_verts = [v for v in tri if v in bottom_pts]
    if len(b_verts) == 2:
        # This triangle has one bottom edge!
        pA = bottom_pts[b_verts[0]]
        pB = bottom_pts[b_verts[1]]
        dist = math.hypot(pB[0]-pA[0], pB[1]-pA[1])
        if dist > 0.01: # Avoid degenerate
            edge = tuple(sorted([pA, pB]))
            wall_2d_edges.add(edge)

print(f"Extracted 2D wall outline segments: {len(wall_2d_edges)}")

# Export these segments as a compact JSON / JS file
edge_list = [[[round(p[0][0], 3), round(p[0][1], 3)], [round(p[1][0], 3), round(p[1][1], 3)]] for p in wall_2d_edges]
print(f"Sample edge: {edge_list[0]}")

with open('PaginaWebCarro/paredes_2d.js', 'w', encoding='utf-8') as f:
    f.write('export const paredes2DData = ' + json.dumps(edge_list) + ';\n')

print("Exported PaginaWebCarro/paredes_2d.js successfully!")

