# Informe Técnico Maestro: Conducción Autónoma por Clonación de Comportamiento Neuronal

**Proyecto:** Vehículo Autónomo de Carreras de Alta Velocidad  
**Plataforma de Simulación:** ROS 2 Humble | Ignition Gazebo (Gazebo Sim) | PyTorch (CUDA Tesla T4)  
**Dominio Académico:** Robótica Móvil, Control en Bucle Cerrado y Aprendizaje Supervisado  
**Fecha de Actualización:** 2026-07-30  

---

## 1. Resumen Ejecutivo y Marco Teórico de Control

Este proyecto implementa una arquitectura avanzada de **Conducción Autónoma por Clonación de Comportamiento (Behavioral Cloning)** basada en **Control Perceptivo en Bucle Cerrado (*Closed-Loop Perceptual Control*)**.

```mermaid
graph TD
    subgraph Sensores ["1. Percepción en Tiempo Real"]
        LIDAR["LiDAR 2D (/scan)"] --> STATE["Vector de Estado Normalizado X_t"]
        CAM["Cámara FPV (/camera/image_raw)"] --> FRAME["Streaming FPV a 30 FPS"]
    end

    subgraph Cerebro ["2. Inferencia Neuronal (PyTorch CUDA T4)"]
        STATE --> MLP["Red Neuronal MLP (Tanh Bounded)"]
        MLP -->|Mapeo Estado-Acción pi(St)| ACTION["Comandos Normalizados Y_t"]
    end

    subgraph Actuacion ["3. Control y Dinámica del Vehículo"]
        ACTION --> CMD["Tópico /cmd_vel (Twist)"]
        CMD --> GZ["Dinámica Físico-Cinemática en Gazebo Sim"]
        GZ -->|Modifica Posición y Paredes| Sensores
    end
```

---

## 2. Preguntas Fundamentales para la Exposición Académica

### ❓ Pregunta 1: ¿El Piloto Neuronal es "Ciego" y solo repite giros a ciegas?

**RESPUESTA CATEGÓRICA: NO, EL PILOTO NO ES CIEGO. OPERA EN BUCLE CERRADO (*CLOSED-LOOP*).**

#### 🔬 Justificación Técnica para los Jurados:
1. **Diferencia entre Control en Bucle Abierto (*Open-Loop*) y Bucle Cerrado (*Closed-Loop*):**
   * **Un sistema ciego (*Open-Loop*):** Simplemente reproduciría una lista fija de comandos temporales (ej. *"girar a la izquierda a los 3 segundos"*). Si mueves la pista o cambias al auto de lugar, el auto se estrellaría de inmediato porque no sabe dónde está.
   * **Nuestro Piloto Neuronal (*Closed-Loop*):** En **cada instante de tiempo $t$ (cada 50 milisegundos)**, la Red Neuronal toma las distancias vivas de las paredes emitidas por los sensores.
2. **Mapeo Perceptivo Estado-Acción ($\pi(S_t) \to A_t$):**
   * La Red Neuronal **NO memorizó la pista ni el tiempo**. Memorizó la **función matemática de navegación**: 
     $$\pi(\text{Distancia Pared Izquierda}, \, \text{Distancia Pared Derecha}) \longrightarrow \text{Ángulo del Volante}$$
   * Si trasladas el auto a otra posición, le cambias la forma al circuito o mueves las paredes, el vector de entrada $X_t$ cambia instantáneamente, y la Red Neuronal **recalcula y corrige la trayectoria de forma adaptativa**.

---

### ❓ Pregunta 2: ¿Cómo se utiliza la Cámara del Carro y la Toma de Frames?

#### 🔬 Detalle Técnico de la Cámara Frontal (`/camera/image_raw`):
1. **Streaming FPV de Alta Definición:** La cámara frontal del vehículo publica imágenes continuas a $30\text{ FPS}$ en el tópico de ROS 2 `/camera/image_raw` con codificación `bgr8`.
2. **Rol en la Arquitectura Híbrida:**
   * **Monitoreo FPV en Tiempo Real:** Permite la telemetría visual de alta fidelidad a través de `rqt_image_view` (Terminal 2) para observar el horizonte y la carretera desde la perspectiva del piloto.
   * **Extracción de Frames para Visión por Computadora (OpenCV / CNN):** Los fotogramas individuales son procesados a resolución $160 \times 120$ para la segmentación de imágenes y extracción de características en el pipeline visual.

---

## 3. Desglose Técnico Paso a Paso del Flujo de Trabajo

### 3.1. Fase 1: Percepción y Conducción Experta por Sensores (`artudo_wall_follower`)
* **Nodo Executable:** `sim_vision_test.artudo_wall_follower_node`
* **Muestreo Angular LiDAR:** En cada ciclo, el nodo toma los rayos LiDAR a $-45^\circ$ ($a$) y $-90^\circ$ ($b$).
* **Estimación Geométrica:** Calcula el ángulo de inclinación $\alpha$ del chasis respecto a la pared del circuito:
  $$\alpha = \arctan2(a \cdot \cos(45^\circ) - b, \; a \cdot \sin(45^\circ))$$
* **Predicción de Distancia a Futuro ($d_{\text{predicha}}$):** Estima la distancia a la pared $0.8\text{m}$ hacia adelante:
  $$d_{\text{predicha}} = (b \cdot \cos\alpha) + 0.8 \cdot \sin\alpha$$
* **Control Proporcional-Integral-Derivativo (PID):** Genera la corrección angular $\omega$ para mantener $d_{\text{predicha}} \approx 1.0\text{m}$ (centro del carril).

---

### 3.2. Fase 2: Grabación Automática de Telemetría (`artudo_data_recorder`)
* **Nodo Executable:** `sim_vision_test.artudo_data_recorder_node`
* **Frecuencia de Muestreo:** $20\text{ Hz}$ ($50\text{ms}$).
* **Estructura del Vector de Estado ($X_t \in \mathbb{R}^8$):**
  * 8 sectores angulares LiDAR con reducción *Min-Pooling* normalizados en el rango $[0.0, 1.0]$.
* **Estructura del Vector de Acción ($Y_t \in \mathbb{R}^2$):**
  * $[\text{steer}, \; \text{speed}]$ emitidos por el controlador en el instante $t$.
* **Almacenamiento:** Formato binario comprimido NumPy (`~/dataset_artudo/artudo_expert_dataset.npz`) con más de **170,000 muestras puras**.

---

### 3.3. Fase 3: Entrenamiento Supervisado por Clonación (`train_artudo_cloning`)
* **Script Executable:** `sim_vision_test.train_artudo_cloning`
* **Hardware:** GPU NVIDIA Tesla T4 con aceleración PyTorch CUDA.
* **Arquitectura de la Red Neuronal (MLP):**
  $$\text{Input}(8) \xrightarrow{\text{Linear}} \text{FC1}(64) \xrightarrow{\text{ReLU}} \text{FC2}(64) \xrightarrow{\text{ReLU}} \text{FC3}(32) \xrightarrow{\text{ReLU}} \text{Output}(2) \xrightarrow{\text{Tanh()}}$$
* **Función de Activación Final `Tanh()`:** Bounded output en $[-1.0, 1.0]$. Garantiza que las salidas des-normalizadas no superen los límites físicos ($\text{steer} \in [-0.70, 0.70]\text{ rad}$, $\text{speed} \ge 0.18\text{ m/s}$).
* **Función de Pérdida y Optimizador:** Error Cuadrático Medio (MSE Loss) con optimizador Adam ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-5}$).
* **Convergencia:** Loss $< 0.001$ alcanzado en 40 épocas ($< 30\text{ segundos}$ en GPU T4).

---

### 3.4. Fase 4: Despliegue del Piloto Neuronal Autónomo (`artudo_neural_pilot`)
* **Nodo Executable:** `sim_vision_test.artudo_neural_pilot_node`
* **Frecuencia de Inferencia:** $20\text{ Hz}$ en GPU CUDA.
* **Tiempo de Cómputo por Cuadro:** $< 0.5\text{ milisegundos}$.
* **Publicación ROS 2:** Emite mensajes `geometry_msgs/msg/Twist` al tópico `/cmd_vel` para controlar los motores en Gazebo.

---

## 4. Guía de Ejecución de Comandos para Demostración

### 1️⃣ Terminal 1: Lanzar Simulación 3D (`headless:=false`)
```bash
cd /home/ubuntu/robot-vision-sim
git pull origin main --rebase
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch launch/robot_camera.launch.py world:=racetrack_decorated headless:=false
```

---

### 2️⃣ Terminal 2: Visor FPV de la Cámara Frontal (`rqt_image_view`)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash

ros2 run rqt_image_view rqt_image_view
```
*(Seleccionar `/camera/image_raw` arriba a la izquierda para la toma a color en vivo).*

---

### 3️⃣ Terminal 3: Lanzar el Piloto Neuronal Autónomo (IA PyTorch CUDA)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run sim_vision_test artudo_neural_pilot
```
