#!/usr/bin/env bash
# ============================================================================
# install_optional_tools.sh
#
# Instala herramientas OPCIONALES que no hacen falta para el pipeline CNN
# principal (piloto experto -> grabadora -> entrenamiento -> piloto autónomo
# por cámara), pero sirven para depurar visualmente o para probar el
# seguidor de línea clásico por OpenCV (ver INFORME_VISION.md).
#
# Corré esto DESPUÉS de scripts/install_ros2_gazebo.sh, solo si lo necesitás.
#
# Uso:
#   ./scripts/install_optional_tools.sh
# ============================================================================
set -euo pipefail

echo "==> Instalando RViz2, visor de cámara (rqt_image_view) y teleclado"
sudo apt update
sudo apt install -y \
    ros-humble-rviz2 \
    ros-humble-rqt-image-view \
    ros-humble-teleop-twist-keyboard

echo "==> Listo. Herramientas opcionales instaladas."
