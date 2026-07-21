# Informe del Proyecto: Simulación de Navegación y Visión por Computadora en ROS 2

Este documento contiene un análisis detallado sobre el desarrollo, arquitectura y optimizaciones del proyecto de navegación y procesamiento de visión artificial para el robot en el simulador Gazebo, utilizando ROS 2 (Humble/Jazzy) y OpenCV.

---

## 1. Objetivos del Proyecto y Requerimientos Técnicos

El sistema se ha diseñado bajo los lineamientos obligatorios de la guía de cátedra:
1. **Estructura ament_python (`sim_vision_test`):** Un paquete de ROS 2 en Python con dependencias explícitas de `rclpy`, `sensor_msgs`, `cv_bridge`, `image_transport` y `python3-opencv`.
2. **Puente de Comunicaciones (`ros_gz_bridge`):** Mapeo correcto de `/camera/image_raw` y `/clock` desde Gazebo Sim a ROS 2.
3. **Procesamiento de Región de Interés (ROI):** Redimensionamiento del frame de la cámara a $320 \times 240$ píxeles y extracción de la sección inferior del 40% (línea de seguimiento).
4. **Segmentación y Centroide:** Uso de HSV (`cv2.inRange`), cálculo de momentos espaciales (`cv2.moments`) para determinar el centroide (`cx`), cálculo del error de desviación en píxeles y reporte continuo de FPS en consola.

---

## 2. Optimización para Entornos de Máquina Virtual (VM)

Dado que la ejecución de entornos 3D en máquinas virtuales suele carecer de aceleración gráfica por hardware (GPU dedicada passthrough), se han implementado las siguientes estrategias de optimización para garantizar estabilidad e impedir la congelación del sistema:

* **Modo Headless en Lanzamiento:** Se añadió el parámetro `headless` a los archivos de lanzamiento (`robot_camera.launch.py`). Al activarse (`headless:=true`), Gazebo inicia el servidor en segundo plano (`-s -r <world>`) y deshabilita por completo la carga de la interfaz gráfica QT, reduciendo el uso de CPU/GPU en más de un 70%.
* **Resolución Nativa del Sensor Reducida:** Se modificó la resolución física de la cámara simulada en `urdf/my_robot.urdf` de $640 \times 480$ a $320 \times 240$ y su frecuencia de publicación a $15\text{ Hz}$. Esto reduce enormemente el volumen de renderizado de la cámara en Gazebo.
* **Salvaguarda X11 en el Nodo de Visión:** El nodo `vision_sim_node.py` solo intenta renderizar ventanas visuales con `cv2.imshow` si se cuenta con la variable de entorno `DISPLAY` activa y si el parámetro `show_image` es explícitamente fijado en `true`, previniendo fallas de segmentación en SSH o terminales de fondo.
* **OpenGL por Software:** En caso de fallas de renderizado en la GUI de Gazebo, se sugiere forzar el backend OpenGL por software del driver Mesa.

---

## 3. Análisis Diagnóstico: ¿Por qué no se visualizaba el carro o el circuito? (Paso a Paso)

Hemos detectado y resuelto los siguientes 3 problemas técnicos que impedían la correcta visualización del carro y la pista:

### Problema 1: El Carro no aparecía en RViz 2 (Ausencia de TF Bridging)
* **Causa:** En la simulación original, el plugin DiffDrive de Gazebo publicaba las transformadas de coordenadas (TF) del robot al tema `/model/my_robot/tf` de Gazebo. Sin embargo, este tema no estaba mapeado en el puente `ros_gz_bridge`. Sin estas transformadas, RViz no podía calcular la posición relativa de las ruedas, sensores o chasis, mostrando errores de renderizado.
* **Solución:** Modificamos `launch/robot_camera.launch.py` para mapear el tema de transformadas de Gazebo `/model/my_robot/tf` al tema estándar de ROS `/tf` mediante el puente:
  `'/model/my_robot/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'` e implementamos el remapeo a `/tf`.

### Problema 2: El Circuito desaparecía o parpadeaba (Z-Fighting)
* **Causa:** El plano de tierra de Gazebo está situado en $Z = 0.0$. Las líneas originales del circuito estaban definidas con un grosor de $1\text{ mm}$ ($0.001\text{ m}$) a una altura de $1\text{ mm}$ ($0.001\text{ m}$). En renderizadores de VM por software, esto genera un conflicto de profundidad gráfica (*Z-fighting*), haciendo que el suelo oculte visualmente las líneas o estas parpadeen hasta desaparecer.
* **Solución:** Incrementamos el espesor de las líneas del circuito a $1\text{ cm}$ ($0.01\text{ m}$) y elevamos sus centros a $5\text{ mm}$ ($0.005\text{ m}$) en `worlds/camera_world.sdf`. De esta forma, las caras superiores de las líneas reposan sólidamente sobre el suelo (a $10\text{ mm}$) eliminando el error visual.

### Problema 3: No había imagen de cámara en RViz 2
* **Causa:** La configuración de visualización de RViz (`robot.rviz`) estaba suscrita al tema `/camera/image`, pero el sensor físico del robot se modificó para publicar en `/camera/image_raw`.
* **Solución:** Corregimos la suscripción en el archivo `config/robot.rviz` a `/camera/image_raw`.

---

## 4. Comandos de Compilación y Ejecución

Sigue las siguientes instrucciones en tu terminal para compilar el espacio de trabajo y ejecutar la simulación en sus diferentes variantes.

### Paso Inicial: Compilar el Entorno
Abre una terminal en la raíz de tu espacio de trabajo (`/home/akenitoy/robot-vision-sim`) y ejecuta:
```bash
colcon build --packages-select sim_vision_test
source install/setup.bash
```

---

## 5. Flujo Operativo y Visualización (Paso a Paso)

A continuación se detallan los comandos de consola distribuidos en 3 terminales para poder ver visualmente el carro, el mundo y el procesamiento de la cámara:

### Terminal 1: Lanzamiento de la Simulación y Puente (Modo Con Interfaz Visual)
Para iniciar la simulación mostrando la ventana 3D de Gazebo (para ver físicamente al carro y el mundo):
```bash
# Sourcing y forzado de OpenGL por software por seguridad en la VM
export LIBGL_ALWAYS_SOFTWARE=1
source install/setup.bash

# Lanzamiento de Gazebo con interfaz visual activa
ros2 launch launch/robot_camera.launch.py headless:=false
```
*Si la simulación va extremadamente lenta y solo quieres procesar imágenes en el nodo de visión, cambia a `headless:=true` en el comando anterior.*

### Terminal 2: Nodo de Visión OpenCV (Con Ventana de Visualización de Cámara)
Para iniciar el nodo de procesamiento de visión y **ver visualmente la cámara del carro** (el ROI recortado y la máscara HSV de detección de línea):
```bash
source install/setup.bash

# Ejecución del nodo activando las ventanas de visualización de OpenCV
ros2 run sim_vision_test vision_sim_node --ros-args -p show_image:=true
```
Aparecerán dos ventanas de OpenCV en tu escritorio:
1. `"ROI original"`: Muestra el 40% inferior del frame con un círculo verde que indica el centroide detectado y una línea azul central de referencia.
2. `"Mascara HSV"`: Muestra en blanco y negro la segmentación de color lograda con `cv2.inRange`.

### Terminal 3: Teleoperación por Teclado
Para mover al carro de forma manual y observar cómo cambia el cálculo del error y la vista de la cámara:
```bash
source /opt/ros/humble/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
*Usa las teclas indicadas en pantalla (`i`, `j`, `k`, `l`, etc.) para conducir el robot.*

---

## 6. Visualización mediante RViz 2 (Alternativa Ligera)

Si la GUI de Gazebo consume demasiados recursos de procesamiento, puedes ejecutar la simulación en modo headless (Terminal 1 con `headless:=true`) y visualizar el robot y la cámara mediante RViz 2, que es mucho más eficiente:

```bash
# En una terminal adicional:
source install/setup.bash
rviz2 -d config/robot.rviz
```
Desde RViz podrás:
- Ver el modelo 3D del robot (`RobotModel`).
- Inspeccionar el tema de la cámara agregando un display de tipo `Image` suscrito a `/camera/image_raw`.
- Visualizar la nube de puntos o lecturas del sensor láser (`LaserScan`) suscritas al tema `/scan`.
