#!/usr/bin/env bash
# ============================================================================
# install_ros2_gazebo.sh
#
# Instala SOLO lo necesario para correr el pipeline CNN del proyecto
# (piloto experto -> grabadora -> entrenamiento -> piloto autónomo):
# ROS 2 Humble en su variante liviana (ros-base, sin GUI de escritorio),
# Gazebo Sim (ros_gz) y las librerías de ROS que usan los nodos.
#
# Pensado para laptops/VMs de recursos limitados: NO instala
# ros-humble-desktop (que trae RViz, rqt, demos y tutoriales que no
# hacen falta para el pipeline CNN). Si además querés RViz2, teleclado
# o las herramientas de la línea de RL, corré aparte:
#   ./scripts/install_optional_tools.sh
#
# Probado sobre: Ubuntu 22.04 LTS (x86_64).
#
# Uso:
#   chmod +x scripts/install_ros2_gazebo.sh
#   ./scripts/install_ros2_gazebo.sh
# ============================================================================
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: este script está pensado para Ubuntu 22.04 (Linux). Abortando." >&2
    exit 1
fi

if ! grep -qi "jammy" /etc/os-release 2>/dev/null; then
    echo "ADVERTENCIA: este script fue probado en Ubuntu 22.04 (jammy)."
    echo "Tu sistema reporta: $(grep PRETTY_NAME /etc/os-release 2>/dev/null || echo desconocido)"
    read -p "¿Continuar de todas formas? [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

echo "==> 1/5: Configurando locale UTF-8"
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

echo "==> 2/5: Habilitando el repositorio APT de ROS 2"
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

echo "==> 3/5: Instalando ROS 2 Humble BASE (sin GUI de escritorio) + colcon + rosdep"
sudo apt install -y \
    ros-humble-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    git

if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

echo "==> 4/5: Instalando Gazebo Sim (Fortress) y el puente ros_gz"
sudo apt install -y \
    ros-humble-ros-gz-sim \
    ros-humble-ros-gz-bridge

echo "==> 5/5: Instalando librerías ROS que usan los nodos del pipeline CNN"
sudo apt install -y \
    ros-humble-robot-state-publisher \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    python3-opencv

echo "==> Verificando instalación"
source /opt/ros/humble/setup.bash
ros2 --version || true
gz sim --version || true

cat <<'EOF'

============================================================================
Instalación mínima de ROS 2 Humble + Gazebo Sim completa (~la mitad del
espacio en disco y paquetes de una instalación "desktop" completa).

Siguiente paso: instalar PyTorch/pandas/numpy (pipeline CNN)
    ./scripts/install_python_deps.sh

Opcional (RViz2, teleclado, visor de cámara, RL/PPO):
    ./scripts/install_optional_tools.sh

Recordá agregar esto a tu ~/.bashrc para no tener que sourcear siempre:
    source /opt/ros/humble/setup.bash
============================================================================
EOF
