# Informe Técnico Detallado: Arquitectura de Conducción Autónoma en ROS 2 + Gazebo Sim

**Proyecto:** Simulación y Aprendizaje Autónomo de Vehículos Robóticos  
**Entorno:** ROS 2 Humble | Ignition Gazebo (Gazebo Sim) | PyTorch (CUDA Tesla T4)  
**Fecha de Actualización:** 2026-07-26  

---

## 1. Resumen Ejecutivo y Estado Actual

El proyecto ha evolucionado hacia un **sistema híbrido de conducción autónoma de 3 pilares** que combina la robótica determinista clásica con el aprendizaje profundo por IA:

1. **Piloto Autónomo Determinista (`artudo_wall_follower`):** Adaptación 1:1 en ROS 2 del algoritmo de seguimiento de paredes (*Wall-Following*) de `ar-tu-do-master`. Utiliza lecturas del LiDAR 2D y un **Controlador PID con Proyección Futura** para navegar de forma impecable y continua por la pista de carreras decorada en 3D (`racetrack_decorated.sdf`) sin necesidad de entrenamiento.
2. **Grabador Automático de Telemetría (`artudo_data_recorder`):** Captura en tiempo real (a 20 Hz / 50ms) la telemetría del piloto de `ar-tu-do-master` durante 20 o más vueltas continuas, generando un **dataset experto limpio y sin choques** (`artudo_expert_dataset.npz`) para entrenar redes neuronales por Aprendizaje Supervisado (Clonación de Comportamiento).
3. **Entorno de Aprendizaje por Refuerzo (PPO - Stable-Baselines3):** Entorno Gymnasium (`racetrack_env.py`) equipado con **5 correcciones críticas de estabilidad** (detección de atascos, penalización por proximidad, features de peligro y *Frame Stacking* de 4 cuadros) para entrenar políticas óptimas aceleradas por GPU en CUDA Tesla T4.

---

## 2. Mapa de Nodos del Paquete `sim_vision_test`

| Nodo / Executable | Archivo de Origen | Descripción y Función |
| :--- | :--- | :--- |
| **`artudo_wall_follower`** | `artudo_wall_follower_node.py` | Piloto autónomo basado en PID que conduce el auto calculando distancias a paredes en $-45^\circ$ y $-90^\circ$. |
| **`artudo_data_recorder`** | `artudo_data_recorder_node.py` | Capturador automático de datos. Intercepta `/scan` y `/cmd_vel` a 20Hz y guarda datasets `.npz`. |
| **`train_sb3`** | `train_sb3.py` | Entrenador PPO (Stable-Baselines3) con `VecFrameStack(n_stack=4)` ejecutándose en GPU Tesla T4. |
| **`evolutionary_trainer`** | `evolutionary_trainer.py` | Entrenador genético evolutivo (25 individuos por generación con mutación de pesos). |
| **`data_recorder_node`** | `data_recorder.py` | Grabador manual de imágenes de cámara (OpenCV) y comandos de teclado. |
| **`vision_sim_node`** | `vision_sim_node.py` | Procesador de visión por computadora OpenCV (Segmentación HSV y centroide ROI 40%). |

---

## 3. Explicación Detallada de los Componentes Clave

```mermaid
graph TD
    subgraph Simulador ["Entorno Gazebo Sim (Pista 3D Decorada)"]
        GZ["Gazebo Sim (racetrack_decorated.sdf)"]
    end

    subgraph Piloto ["Piloto Autónomo (ar-tu-do-master)"]
        GZ -->|Lectura /scan| PID["artudo_wall_follower_node"]
        PID -->|Control PID (steer, speed)| GZ
    end

    subgraph Grabador ["Grabador de Telemetría (Hands-Free)"]
        GZ -->|Tópico /scan| REC["artudo_data_recorder_node"]
        PID -->|Tópico /cmd_vel| REC
        REC -->|Muestras 20Hz| DATA["dataset_artudo_expert.npz"]
    end

    subgraph Entrenador ["Aprendizaje por Refuerzo PPO"]
        GZ -->|racetrack_env.py (14 dims)| STACK["VecFrameStack (56 dims)"]
        STACK --> PPO["train_sb3 (PPO en CUDA GPU T4)"]
    end
```

### 3.1. Algoritmo del Piloto Autónomo (`artudo_wall_follower`)

El piloto autónomo se basa en el principio de **predicción de distancia a la pared** derivado del proyecto `ar-tu-do-master` / F1TENTH:

1. **Muestreo Angular LiDAR:** En cada ciclo, toma la distancia a dos ángulos respecto al eje del vehículo: $a$ (a $-45^\circ$) y $b$ (a $-90^\circ$).
2. **Ángulo de Inclinación ($\alpha$):** Calcula el ángulo de orientación del vehículo con respecto a la pared:
   $$\alpha = \arctan2(a \cdot \cos(\theta) - b, \; a \cdot \sin(\theta)) \quad \text{donde } \theta = 45^\circ$$
3. **Distancia Predicha:** Estima a qué distancia estará la pared $0.8\text{m}$ más adelante:
   $$\text{distancia\_predicha} = (b \cdot \cos(\alpha)) + 0.8 \cdot \sin(\alpha)$$
4. **Controlador PID:** Ajusta el volante para mantener $\text{distancia\_predicha} \approx 1.0\text{m}$ (centro del carril).
5. **Velocidad Adaptativa:** En rectas acelera a $0.45\text{ m/s}$ y frena automáticamente en curvas cerradas a $0.18\text{ m/s}$ en función del ángulo del volante.

---

### 3.2. Grabación Automática de Telemetría (`artudo_data_recorder`)

Permite generar datasets de entrenamiento sin intervención humana:

* **Frecuencia:** Muestra cada $50\text{ms}$ ($20\text{ Hz}$).
* **Vector de Estado ($X_t$):** 8 sectores de distancia LiDAR (min-pooling) + velocidad lineal normalizada + ángulo de giro anterior.
* **Vector de Acción ($Y_t$):** $[\text{steer}, \; \text{speed}]$ emitidos por el piloto automático.
* **Volumen:** 20 vueltas continuas producen $\approx 15,000$ muestras puras sin choques.

---

### 3.3. Entorno PPO Mejorado (`racetrack_env.py` + `train_sb3.py`)

Para el aprendizaje por refuerzo, se implementaron **5 correcciones de estabilidad**:

1. **Reset por Atasco Temprano (Fix 1):** Si el desplazamiento es $< 1\text{cm}$ durante 30 steps ($\approx 1.5\text{s}$), el episodio termina con penalización de $-5.0$, evitando saturar el buffer con datos muertos.
2. **Penalización Gradual por Proximidad (Fix 2):** Si un obstáculo está a $< 30\text{cm}$, se aplica una penalización lineal que otorga un gradiente continuo a PPO antes de chocar.
3. **Features de Peligro Destacados (Fix 3):** Se expandió la observación de 12 a **14 dimensiones** añadiendo `min_lidar_frontal` y `min_lidar_global`.
4. **Frame Stacking (Fix 4):** Se integró `VecFrameStack(env, n_stack=4)` en `train_sb3.py` para darle percepción temporal de 4 cuadros ($14 \times 4 = 56$ dimensiones efectivas).
5. **Reward Simétrico por Retroceso (Fix 5):** Moverse en reversa se penaliza a $\Delta s \cdot 5.0$, eliminando el retroceso gratuito.

---

## 4. Guía Completa de Ejecución en AWS EC2 (Tesla T4)

Para poner en marcha la simulación y la grabación de 20 vueltas automáticas:

### 0️⃣ Paso 0: Limpieza, Pull y Compilación
```bash
pkill -9 -f train_sb3; pkill -9 -f artudo; pkill -9 -f ros2; pkill -9 -f ign

cd /home/ubuntu/robot-vision-sim
git pull origin main
source /opt/ros/humble/setup.bash
colcon build --packages-select sim_vision_test
source install/setup.bash
```

### 1️⃣ Terminal 1: Lanzar Simulación (Pista Decorada 3D Completa)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch launch/robot_camera.launch.py world:=racetrack_decorated headless:=true
```

### 2️⃣ Terminal 2: Visualizador de Cámara (Opcional)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
ros2 run rqt_image_view rqt_image_view
```
*(Seleccionar `/camera/image_raw` arriba a la izquierda).*

### 3️⃣ Terminal 3: Piloto Autónomo de `ar-tu-do-master`
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run sim_vision_test artudo_wall_follower
```

### 4️⃣ Terminal 4: Grabador Automático de Telemetría
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run sim_vision_test artudo_data_recorder
```
*(Dejar correr hasta superar las 15,000 muestras / 20 vueltas y presionar `Ctrl+C` para guardar).*
