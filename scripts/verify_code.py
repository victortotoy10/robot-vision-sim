import os
import subprocess

print("Checking JS syntax with node...")
for js_file in ['PaginaWebCarro/paredes_2d.js', 'PaginaWebCarro/app.js']:
    res = subprocess.run(['node', '--check', js_file], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"OK: {js_file} syntax is clean!")
    else:
        print(f"ERROR in {js_file}:\n{res.stderr}")

print("Verification finished!")

