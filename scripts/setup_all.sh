#!/usr/bin/env bash
# ============================================================================
# setup_all.sh
#
# Instalación mínima y optimizada para laptops/VMs de recursos limitados:
# ROS 2 Humble (base, sin GUI), Gazebo Sim, PyTorch CPU, y compilación del
# workspace — solo lo necesario para correr el pipeline CNN de punta a punta.
#
# Uso:
#   chmod +x scripts/*.sh
#   ./scripts/setup_all.sh          # PyTorch CPU (recomendado, laptop sin GPU)
#   ./scripts/setup_all.sh --cuda   # PyTorch con GPU NVIDIA (si tenés una)
#
# Para instalar además las herramientas opcionales (RViz2, teleclado,
# visor de cámara):
#   ./scripts/install_optional_tools.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-cpu}"

"$REPO_ROOT/scripts/install_ros2_gazebo.sh"
"$REPO_ROOT/scripts/install_python_deps.sh" "$MODE"
"$REPO_ROOT/scripts/build_workspace.sh"

cat <<'EOF'

============================================================================
✅ Instalación mínima del pipeline CNN completa.

Ver README.md, sección "Guía de ejecución", para: levantar la simulación,
correr el piloto experto, grabar datos, entrenar la CNN y correr el
piloto autónomo por cámara.
============================================================================
EOF
