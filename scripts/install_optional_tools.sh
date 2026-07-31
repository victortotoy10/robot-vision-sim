#!/usr/bin/env bash
# ============================================================================
# install_optional_tools.sh
#
# Instala herramientas y dependencias OPCIONALES, que no hacen falta para
# el pipeline CNN principal (piloto experto -> grabadora -> entrenamiento ->
# piloto autónomo por cámara), pero sirven para depurar visualmente o para
# explorar las líneas de trabajo opcionales del proyecto (Aprendizaje por
# Refuerzo con PPO, seguidor de línea clásico con OpenCV).
#
# Corré esto DESPUÉS de scripts/install_ros2_gazebo.sh, solo si lo necesitás.
#
# Uso:
#   ./scripts/install_optional_tools.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Instalando RViz2, visor de cámara (rqt_image_view) y teleclado"
sudo apt update
sudo apt install -y \
    ros-humble-rviz2 \
    ros-humble-rqt-image-view \
    ros-humble-teleop-twist-keyboard

echo "==> Instalando dependencias Python de la línea de RL (Gymnasium + Stable-Baselines3)"
python3 -m pip install -r "$REPO_ROOT/requirements-optional.txt"

echo "==> Listo. Herramientas opcionales instaladas."
