import json

# Inspect track
with open('PaginaWebCarro/pista_gazebo.js', 'r', encoding='utf-8') as f:
    txt = f.read()
idx = txt.find('[')
# Find the matching closing bracket or strip trailing semicolon
end_idx = txt.rfind(']')
track_data = json.loads(txt[idx:end_idx+1])
print(f'Track parts: {len(track_data)}')
for i, part in enumerate(track_data):
    verts = part['vertices']
    xs = verts[0::3]
    ys = verts[1::3]
    zs = verts[2::3]
    t_xs = xs
    t_zs = [-y for y in ys]
    print(f"Part {i} ({part['name']}): color={part['color']}, X in [{min(t_xs):.2f}, {max(t_xs):.2f}], Z in [{min(t_zs):.2f}, {max(t_zs):.2f}]")

# Inspect walls
with open('PaginaWebCarro/paredes_gazebo.js', 'r', encoding='utf-8') as f:
    txt = f.read()
idx = txt.find('{')
end_idx = txt.rfind('}')
wall_data = json.loads(txt[idx:end_idx+1])
verts = wall_data['vertices']
xs = verts[0::3]
ys = verts[1::3]
zs = verts[2::3]
t_xs = xs
t_zs = [-y for y in ys]
print(f"Walls: {len(xs)} vertices. X in [{min(t_xs):.2f}, {max(t_xs):.2f}], Height in [{min(zs):.2f}, {max(zs):.2f}], Z in [{min(t_zs):.2f}, {max(t_zs):.2f}]")

# Road geometry is usually geometry1_ref or geometry0_ref
# Let's find vertices of the road (color dark grey [0.2667, 0.2667, 0.2667])
road_part = None
for p in track_data:
    if p['color'][0] < 0.35 and p['color'][1] < 0.35:
        road_part = p
        break

if road_part:
    r_verts = road_part['vertices']
    r_xs = r_verts[0::3]
    r_zs = [-y for y in r_verts[1::3]]
    # Filter road vertices near X=0
    road_at_x0 = [(r_xs[i], r_zs[i]) for i in range(len(r_xs)) if -1 <= r_xs[i] <= 1]
    print(f"Road points near X in [-1, 1]: {len(road_at_x0)}")
    road_z_at_x0 = sorted(list(set([round(p[1], 2) for p in road_at_x0])))
    print(f"Road Z coordinates near X=0: {road_z_at_x0}")

# Walls near X in [-1, 1]
walls_at_x0 = [(t_xs[i], t_zs[i], zs[i]) for i in range(len(xs)) if -1 <= t_xs[i] <= 1]
walls_z_at_x0 = sorted(list(set([round(p[1], 2) for p in walls_at_x0])))
print(f"Wall Z coordinates near X=0: {walls_z_at_x0}")

