import xml.etree.ElementTree as ET

tree = ET.parse('meshes/walls.dae')
root = tree.getroot()
ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}

pos_elem = root.find('.//c:float_array[@id="ID9_000-mesh-positions-array"]', ns)
pos_vals = [float(x) for x in pos_elem.text.split()]
xs = pos_vals[0::3]
ys = pos_vals[1::3]
zs = pos_vals[2::3]

# In Three.js: X = x, Y = z, Z = -y
print(f"Total vertices in walls.dae: {len(xs)}")
straightaway_pts = [(xs[i], -ys[i], zs[i]) for i in range(len(xs)) if -2.0 <= -ys[i] <= 2.0]
print(f"Vertices with Three.js Z in [-2, 2]: {len(straightaway_pts)}")
for p in straightaway_pts[:30]:
    print(f"  Three.js X={p[0]:.2f}, Z={p[1]:.2f}, Height={p[2]:.2f}")

with open('worlds/racetrack.sdf', 'r') as f:
    print("\n--- racetrack.sdf link poses ---")
    for line in f:
        if 'model name=' in line or '<link' in line or '<pose' in line or '<mesh' in line:
            print("  ", line.strip())

