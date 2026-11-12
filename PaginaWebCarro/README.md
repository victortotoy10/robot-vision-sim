# Estudio CAD 3D y Simulador 4WD Smart Car Robot

Aplicación web 3D interactiva de alta fidelidad construida con **Three.js** para la inspección CAD detallada y la simulación física de cinemática con **Tracción Diferencial (Skid-Steering)** del kit robótico **4WD Smart Car** (ESP32-WROOM + ESP32-CAM + L298N + LM2596S + Baterías 18650).

---

## Modos de Operación

### 1. Modo Estudio CAD 3D (Inspector Técnico)
- **Vista Explosionada (0% - 100%):** Slider interactivo que separa gradualmente las placas de chasis, motores, ruedas y componentes electrónicos a lo largo de sus ejes de montaje.
- **Inspector de Componentes:** Al hacer clic sobre cualquier pieza del modelo 3D o seleccionarla en el árbol, se despliega una ficha técnica con voltaje, consumo, dimensiones, conexiones y su rol en la arquitectura del robot.
- **Vistas Predefinidas de Cámara:**
  - `ISO`: Perspectiva isométrica 3/4.
  - `TOP`: Vista cenital técnica superior.
  - `FRNT`: Vista frontal hacia la lente OV2640.
  - `SIDE`: Vista lateral de perfil.
- **Modos de Sombreado:**
  - `PBR`: Renderizado realista basado en física.
  - `Malla`: Vista wireframe técnica de polígonos.
  - `Rayos X`: Vista semitransparente de inspección interna.

---

### 2. Modo Simulador de Tracción Diferencial
- **Cinemática Diferencial Real (Skid-Steering):**
  $$V = \frac{V_R + V_L}{2}$$
  $$\omega = \frac{V_R - V_L}{L}$$
  *Donde $L = 129\text{ mm}$ es la vía entre ruedas.*
- **Animación Individual de Ruedas:** Las ruedas izquierdas y derechas rotan independientemente según su velocidad angular respectiva ($\omega_i = v_i / R$).
- **Giro en el Sitio (Giro Tanque):** Al girar sobre su propio eje, las ruedas de un lado rotan hacia adelante y las del lado opuesto rotan hacia atrás.
- **Telemetría en Vivo:**
  - Velocidad lineal ($m/s$).
  - Velocidad angular de rotación ($\deg/s$).
  - Nivel de modulación PWM de cada canal del L298N ($0 - 255$).
  - Estado cinemático del robot.
- **Cámara FPV en Tiempo Real:** Ventana Picture-in-Picture (PiP) que transmite la vista subjetiva desde la lente de la cámara **ESP32-CAM (OV2640)** montada en el frontal.
- **Sensor Láser 2D LiDAR (Hokuyo UST-10LX):** Emulación de escáner láser de 25 haces con raycasting en tiempo real contra las paredes originales de Gazebo (`walls.dae`), renderizado de haces cian/esmeralda y nube de puntos de impacto.
- **Piloto Autónomo (Centrado y Seguidor de Pared PID):** Implementación 1:1 del nodo de navegación de Gazebo (`artudo_wall_follower_node.py`), con cálculo de ángulo de inclinación $\alpha$, predicción de distancia a la pared con lookahead $L$, error de centrado entre paredes y regulación dinámica de aceleración.
- **Telemetría Láser en Vivo:** HUD con lectura en tiempo real de distancia izquierda ($d_L$), distancia frontal ($d_F$), distancia derecha ($d_R$), desvío del centro de carril en centímetros y estado del piloto autónomo.
- **Sintetizador de Sonido de Motores (Web Audio API):** Generación procedural en tiempo real del sonido característico de los motores DC según la velocidad y el giro.

---

## Controles de Manejo

| Tecla / Control | Acción |
| :--- | :--- |
| **W** / **&uarr;** | Avanzar en línea recta |
| **S** / **&darr;** | Retroceder en línea recta |
| **A** / **&larr;** | Giro diferencial hacia la izquierda (giro tanque) |
| **D** / **&rarr;** | Giro diferencial hacia la derecha (giro tanque) |
| **Espacio** | Freno de emergencia inmediato |
| **P** | **Activar / Desactivar Piloto Autónomo (PID Centrado / Wall-Follower)** |
| **L** | **Mostrar / Ocultar Haces Láser 3D en la Pista** |
| **C** | Alternar modos de cámara (Persecución / Órbita / FPV) |
| **R** | Reiniciar robot al origen de la pista |
| **D-Pad en Pantalla** | Control táctil completo para ratón o dispositivos móviles |

---

## Cómo Ejecutar Localmente

### Opción 1: Con el archivo `.bat` (Recomendado en Windows)
Haz doble clic en `run_server.bat`. Esto iniciará automáticamente un servidor HTTP local en Python y abrirá tu navegador predeterminado en `http://localhost:8000`.

### Opción 2: Desde la terminal
Abre una terminal en esta carpeta y ejecuta:
```bash
python -m http.server 8000
```
Y abre `http://localhost:8000` en tu navegador.
