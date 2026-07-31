#!/usr/bin/env bash
# ============================================================================
# install_python_deps.sh
#
# Instala las dependencias Python del PIPELINE CNN (PyTorch, pandas, numpy).
#
# Por defecto instala PyTorch en su versión CPU: es la opción correcta para
# laptops sin GPU dedicada. El entrenamiento (train_cnn) tarda más en CPU,
# pero el dataset de este proyecto es chico (imágenes 160x120, 80 épocas)
# y corre igual en minutos en una laptop modesta.
#
# Uso:
#   ./scripts/install_python_deps.sh          # PyTorch CPU (recomendado, laptop)
#   ./scripts/install_python_deps.sh --cuda    # PyTorch con GPU NVIDIA (si tenés una)
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"

python3 -m pip install --upgrade pip

if [[ "$MODE" == "--cuda" ]]; then
    echo "==> Instalando PyTorch con soporte CUDA (requiere GPU NVIDIA + drivers instalados)"
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
else
    echo "==> Instalando PyTorch (solo CPU) — recomendado para laptops sin GPU"
    echo "    Si tenés GPU NVIDIA y querés acelerar el entrenamiento, corré:"
    echo "    ./scripts/install_python_deps.sh --cuda"
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Instalando el resto de dependencias del pipeline CNN (pandas, numpy)"
python3 -m pip install -r "$REPO_ROOT/requirements.txt"

echo "==> Verificando instalación de PyTorch"
python3 -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('Dispositivo:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

cat <<'EOF'

============================================================================
Dependencias Python del pipeline CNN instaladas.

Siguiente paso: compilar el workspace ROS 2
    ./scripts/build_workspace.sh
============================================================================
EOF
