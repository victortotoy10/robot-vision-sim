# Informe Técnico Definitivo: Sistema Autónomo de Conducción por Clonación Neuronal

**Proyecto:** Vehículo Autónomo de Carreras  
**Plataforma:** ROS 2 Humble | Ignition Gazebo (Gazebo Sim) | PyTorch (GPU Tesla T4)  
**Fecha de Actualización:** 2026-07-30  

---

## 1. Resumen Ejecutivo y Metodología Unificada

Este proyecto implementa una **metodología avanzada de Conducción Autónoma por Clonación de Comportamiento (Behavioral Cloning)**. 

En lugar de requerir que un operador humano conduzca manualmente durante horas arriesgándose a cometer errores, el sistema utiliza un **flujo de trabajo automatizado de 4 fases**:

```mermaid
graph TD
    subgraph Fase1 ["Fase 1: Conducción Autónoma por Sensores"]
        LIDAR["Sensores LiDAR 2D"] --> WALL["Piloto Reactivo (artudo_wall_follower)"]
        WALL -->|Conduce Perfecto sin Choques| SIM["Gazebo Sim (Pista 3D Decorada)"]
    end

    subgraph Fase2 ["Fase 2: Grabación de Telemetría en Paralelo"]
        SIM -->|Captura Telemetría 20Hz| REC["Grabador Automático (artudo_data_recorder)"]
        REC -->|Acumula 200+ Vueltas Limpias| DATA["expert_dataset.npz (170,000+ muestras)"]
    end

    subgraph Fase3 ["Fase 3: Entrenamiento Supervisado en GPU"]
        DATA -->|Alimenta| GPU["Entrenador PyTorch CUDA (train_artudo_cloning)"]
        GPU -->|Minimiza Error (Loss ~0.001)| MODEL["artudo_expert_model.pth"]
    end

    subgraph Fase4 ["Fase 4: Despliegue del Piloto Neuronal"]
        MODEL -->|Carga Red Neuronal| PILOT["Piloto Neuronal (artudo_neural_pilot)"]
        PILOT -->|Inferencia en < 0.5ms| SIM
    end
```

---

## 2. Explicación Detallada del Flujo de Trabajo (Paso a Paso)

### 2.1. Fase 1: Conducción Experta por Sensores (`artudo_wall_follower`)
* **Propósito:** Generar una trayectoria experta y fluida en la pista decorada 3D (`racetrack_decorated.sdf`).
* **Algoritmo:** Utiliza lecturas del **LiDAR 2D** a $-45^\circ$ y $-90^\circ$ para calcular el ángulo de inclinación $\alpha$ con respecto a las paredes del circuito:
  $$\alpha = \arctan2(a \cdot \cos(45^\circ) - b, \; a \cdot \sin(45^\circ))$$
* **Proyección Futura:** Estima la posición del vehículo a $0.8\text{m}$ hacia adelante y un **Controlador PID** ajusta el volante para mantener el carro centrado en el carril.

---

### 2.2. Fase 2: Grabación Automática de Telemetría (`artudo_data_recorder`)
* **Propósito:** Capturar un dataset masivo sin intervención humana.
* **Frecuencia:** Muestrea a $20\text{ Hz}$ ($50\text{ms}$).
* **Estructura de Datos:**
  * **Vector de Estado ($X_t$):** 8 sectores de distancia LiDAR (normalizados de 0 a 1).
  * **Vector de Acción ($Y_t$):** $[\text{steer}, \; \text{speed}]$ ejecutados por el piloto experto.
* **Resultado:** Al cabo de 200+ vueltas, genera un dataset perfecto con más de **170,000 muestras puras** (`artudo_expert_dataset.npz`).

---

### 2.3. Fase 3: Entrenamiento Supervisado por Clonación (`train_artudo_cloning`)
* **Propósito:** Entrenar el "cerebro" de la Red Neuronal para que imite la conducción experta.
* **Arquitectura:** Perceptrón Multicapa (MLP) acelerado en la **GPU Tesla T4 (PyTorch CUDA)**:
  $$\text{Linear}(8 \to 64) \xrightarrow{\text{ReLU}} \text{Linear}(64 \to 64) \xrightarrow{\text{ReLU}} \text{Linear}(64 \to 32) \xrightarrow{\text{ReLU}} \text{Linear}(32 \to 2) \xrightarrow{\text{Tanh()}}$$
* **Normalización y Tanh():** La capa final `Tanh()` acota matemáticamente las predicciones a $[-1.0, 1.0]$, garantizando que los giros sean suaves y la marcha continuada.
* **Tiempo de Entrenamiento:** Menos de $30\text{ segundos}$ en GPU Tesla T4.

---

### 2.4. Fase 4: Despliegue del Piloto Neuronal Autónomo (`artudo_neural_pilot`)
* **Propósito:** Reemplazar los algoritmos deterministas y dejar que la **Red Neuronal Inteligente** conduzca el vehículo de forma autónoma.
* **Operación:** En cada fotograma del LiDAR, la Red Neuronal realiza la inferencia en **menos de $0.5\text{ milisegundos}$**, emitiendo los comandos `/cmd_vel` de aceleración y giro.

---

## 3. Comandos Principales para la Exposición / Demostración en Vivo

### 1️⃣ Terminal 1: Lanzar Simulación 3D Interactiva (`headless:=false`)
```bash
cd /home/ubuntu/robot-vision-sim
git pull origin main --rebase
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch launch/robot_camera.launch.py world:=racetrack_decorated headless:=false
```

---

### 2️⃣ Terminal 2: Visor de Cámara FPV en Vivo (Monitoreo Opcional)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash

ros2 run rqt_image_view rqt_image_view
```
*(Seleccionar `/camera/image_raw` arriba a la izquierda para ver la toma frontal FPV del auto).*

---

### 3️⃣ Terminal 3: Lanzar el Piloto Neuronal Autónomo (IA PyTorch)
```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run sim_vision_test artudo_neural_pilot
```

---

## 4. Guía Secundaria: Re-entrenamiento del Modelo en GPU (Terminal 4)

Si deseas demostrar el entrenamiento de la Red Neuronal en tiempo real durante la presentación:

```bash
cd /home/ubuntu/robot-vision-sim
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run sim_vision_test train_artudo_cloning
```
*(Procesará las 170,000+ muestras en 40 épocas y actualizará los pesos de la Red Neuronal en menos de 30 segundos).*
