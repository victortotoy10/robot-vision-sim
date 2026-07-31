#!/usr/bin/env bash
# ============================================================================
# build_workspace.sh
#
# Compila el paquete sim_vision_test con colcon.
# Requiere que ROS 2 Humble ya esté instalado (scripts/install_ros2_gazebo.sh).
#
# Uso:
#   ./scripts/build_workspace.sh
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${ROS_DISTRO:-}" ]; then
    echo "==> Sourceando ROS 2 Humble"
    source /opt/ros/humble/setup.bash
fi

echo "==> Compilando paquete sim_vision_test"
colcon build --packages-select sim_vision_test --symlink-install

cat <<EOF

============================================================================
Build completo.

Activá el workspace en cada terminal nueva con:
    source $REPO_ROOT/install/setup.bash

Luego podés lanzar la simulación, por ejemplo:
    ros2 launch launch/robot_camera.launch.py world:=racetrack_decorated headless:=false

Ver README.md para la guía completa de ejecución (grabación de datos,
entrenamiento de la CNN, y piloto autónomo).
============================================================================
EOF
