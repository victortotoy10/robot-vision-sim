import os

dirs_to_check = [
    r"c:\Users\lenovo\Desktop\robot-vision-sim-main\CarroFisico",
    r"C:\Users\lenovo\Desktop\CamaraEsp32\sample_project",
    r"C:\Users\lenovo\CarroFisicaCamara",
    r"C:\Users\lenovo\Desktop\CarroFisicaCamara" # just in case
]

for d in dirs_to_check:
    print("="*50)
    print("Checking:", d)
    if not os.path.exists(d):
        print("Does not exist!")
        continue
    for root, dirs, files in os.walk(d):
        rel = os.path.relpath(root, d)
        if ".git" in rel or "build" in rel or ".vscode" in rel:
            continue
        print(f"[{rel}]")
        for f in files:
            print(f"  {f}")

