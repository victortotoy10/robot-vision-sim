# 🚗 Firmware ESP-IDF — Carro Físico 4WD Smart Car

Firmware modular desarrollado en C para **ESP-IDF v5.x** (Espressif IoT Development Framework) para la arquitectura de doble microcontrolador del robot físico autónomo:

1. **[`ESP32_Motores/`](ESP32_Motores/)** ([`Esp32.c`](Esp32.c)): Control motriz diferencial (skid-steering) de 4 motores DC mediante PWM (LEDC) sobre el controlador Puente H L298N, interfaz web HUD táctil y API REST con watchdog de seguridad.
2. **[`ESP32_Camara/`](ESP32_Camara/)** ([`Esp32Cam.c`](Esp32Cam.c)): Módulo **ESP32-CAM (AI-Thinker OV2640)** optimizado para streaming de video MJPEG continuo de baja latencia vía Wi-Fi en Core 1, capturas JPEG y telemetría de rendimiento.

---

## 📐 1. Esquema Eléctrico y Distribución de Pines

### ESP32 DevKit (Control de Motores) ➔ Puente H L298N

| Función | Pin L298N | Pin ESP32 | Tipo de Señal | Configuración / Rango |
| :--- | :--- | :--- | :--- | :--- |
| **PWM Lado Derecho** | **ENA** | **GPIO 32** | Salida PWM LEDC | Canal 0, 1 kHz, 8 bits (0 - 255) |
| Sentido Derecho 1 | **IN1** | **GPIO 33** | Salida Digital | `1` Adelante / `0` Atrás |
| Sentido Derecho 2 | **IN2** | **GPIO 22** | Salida Digital | `0` Adelante / `1` Atrás |
| Sentido Izquierdo 1 | **IN3** | **GPIO 27** | Salida Digital | `1` Adelante / `0` Atrás |
| Sentido Izquierdo 2 | **IN4** | **GPIO 26** | Salida Digital | `0` Adelante / `1` Atrás |
| **PWM Lado Izquierdo** | **ENB** | **GPIO 25** | Salida PWM LEDC | Canal 1, 1 kHz, 8 bits (0 - 255) |
| **Masa Común** | **GND** | **GND** | Referencia | **Unido al polo (-) de batería y L298N** |

> [!IMPORTANT]
> **Puentes Jumper ENA y ENB:** Para tener control proporcional de velocidad (PWM), se deben **retirar los dos jumpers de fábrica** colocados en los pines ENA y ENB del módulo L298N y conectar allí los cables a GPIO 32 y GPIO 25. Si se dejan los jumpers puestos, los motores solo funcionarán a velocidad 100% fija.
>
> **Tierra Común (Common Ground):** El pin GND del ESP32, el pin GND del ESP32-CAM, el pin GND del L298N y el borne negativo de la batería deben estar estrictamente interconectados. Sin tierra común compartida, los niveles lógicos flotan y los motores o la cámara se reinician aleatoriamente.

---

### ESP32-CAM (AI-Thinker + OV2640)

* **Alimentación:** Pin `5V` (alimentado desde la salida del convertidor DC-DC LM2596S regulado a 5.0 V estables).
* **Sensor de Imagen:** OV2640 de 2 Megapíxeles con soporte PSRAM (buffer DMA en memoria externa SPIRAM).
* **LED Flash Frontal:** `GPIO 4` (transistor de conmutación de alta intensidad).
* **LED Indicador de Estado:** `GPIO 33` (lógica invertida).
* **Frecuencia de Reloj XCLK:** 10 MHz (`CAMERA_GRAB_WHEN_EMPTY`) para evitar latencias acumuladas y saturación de buffers.

---

## 📶 2. Arquitectura de Red de Ultra Baja Latencia (Hotspot Directo)

Para evitar la latencia que introduce un router doméstico convencional (doble salto de señal, congestión y colisiones de paquetes), el sistema se conecta directamente a la **Zona con cobertura inalámbrica móvil de Windows (Laptop Hotspot)**:

```
[ ESP32-CAM ] (192.168.137.48:81)  ──┐
                                     ├── Wi-Fi 2.4 GHz ──► [ LAPTOP / PC DE CONTROL ]
[ ESP32 ]     (192.168.137.47:80)  ──┘                     (Hotspot: 192.168.137.1)
                                                                 │
                                                    ├── ROS 2 Vision Pilot
                                                    └── Interfaz Web HUD
```

### Configuración en Windows:
1. Ir a **Configuración ➔ Red e Internet ➔ Zona con cobertura inalámbrica móvil**.
2. **Nombre de red (SSID):** `LAPTOP_ROBOT`
3. **Contraseña:** `12345678`
4. **Banda de red:** `2.4 GHz` (obligatorio para ESP32).
5. **Ahorro de energía:** **Desactivado** (para evitar que Windows apague la red si hay pausas breves de tráfico).

### Asignación de Direcciones IP Estáticas:
* **Gateway / Laptop:** `192.168.137.1`
* **ESP32 Motores:** `192.168.137.47`
* **ESP32-CAM:** `192.168.137.48`
* **Máscara:** `255.255.255.0`

> [!TIP]
> **Gestión Segura de Credenciales:** Copia el archivo [`.env.example`](.env.example) a `.env` en la raíz del proyecto para personalizar tus credenciales locales sin subirlas a Git.

---

## 🌐 3. Endpoints de la API REST y Streaming

### ESP32-CAM (`http://192.168.137.48`)
* **Streaming de Video en Vivo (MJPEG):** `http://192.168.137.48:81/stream` (puerto dedicado 81, tarea en Core 1).
* **Captura de Fotograma Estático:** `http://192.168.137.48/cam.jpg` (ideal para pipelines de visión por computador y ROS 2).
* **Telemetría en Vivo (JSON):** `http://192.168.137.48/status`
* **Control de Flash:** `http://192.168.137.48/flash?val=1` (1 = encendido, 0 = apagado).

### ESP32 Motores (`http://192.168.137.47`)
* **Control Remoto HUD Web Táctil:** `http://192.168.137.47/` (interfaz D-Pad para celular y PC).
* **Avanzar Adelante:** `http://192.168.137.47/fwd`
* **Retroceder:** `http://192.168.137.47/rev`
* **Giro Tanque Izquierda:** `http://192.168.137.47/left`
* **Giro Tanque Derecha:** `http://192.168.137.47/right`
* **Frenar / Detener:** `http://192.168.137.47/stop`
* **Control Diferencial Continuo:** `http://192.168.137.47/drive?l=220&r=220` (rango -255 a +255).
* **Ajustar Velocidad Base:** `http://192.168.137.47/speed?val=220`
* **Telemetría y Estado:** `http://192.168.137.47/status`

> [!NOTE]
> **Fail-Safe Watchdog Activo:** El firmware del ESP32 incluye un temporizador de seguridad de **1500 ms**. Si el robot deja de recibir órdenes (por ejemplo por pérdida temporal de conexión o cierre de la ventana de control), los 4 motores se detienen automáticamente para evitar colisiones.

---

## 🛠️ 4. Guía de Compilación y Flasheo con ESP-IDF

### Requisitos:
* ESP-IDF v5.1 o superior instalado.
* Puerto serie identificado (ejemplo: `COM4` para ESP32 Motores, `COM5` para ESP32-CAM).

### Compilar y Grabar ESP32 Motores (`COM4`):
```bash
cd CarroFisico/ESP32_Motores
idf.py set-target esp32
idf.py build
idf.py -p COM4 flash monitor
```

### Compilar y Grabar ESP32-CAM (`COM5`):
> **Nota para ESP32-CAM con módulo programador FTDI / ESP32-CAM-MB:** 
> Asegurarse de que el pin `IO0` esté conectado a `GND` solo al momento de entrar en modo descarga si se usa programador externo sin auto-reset. Con el adaptador con botones, presionar `IO0` + `RST`, soltar `RST` y luego `IO0`.
```bash
cd CarroFisico/ESP32_Camara
idf.py set-target esp32
idf.py build
idf.py -p COM5 flash monitor
```
