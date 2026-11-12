import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pistaGazeboData } from './pista_gazebo.js';
import { paredesGazeboData } from './paredes_gazebo.js';
import { paredes2DData } from './paredes_2d.js';

/* ==========================================================================
   ESTUDIO CAD 3D Y SIMULADOR 4WD SMART CAR ROBOT
   Kinematics: Tracción Diferencial (Skid-Steering)
   Pista: Circuito de Carreras Auténtico de Gazebo
   Sensores: Emulador de Escáner Láser 2D LiDAR (Hokuyo UST-10LX)
   Navegación: Piloto Autónomo Centrado / Wall-Follower (PID ROS 2)
   ========================================================================== */

// --- Base State ---
const state = {
  mode: 'cad', // 'cad' | 'sim'
  exploded: 0.0,
  shading: 'pbr',
  selectedComponent: 'ESP32_CAM',
  soundEnabled: true,
  simCamMode: 'chase', // 'chase' | 'orbit' | 'fpv'
  
  // Autonomous Wall Centering & LiDAR State
  autopilot: false,
  laserVisible: true,
  laserDistances: {
    left: 0.0,
    front: 0.0,
    right: 0.0,
    frontLeft: 0.0,
    frontRight: 0.0,
    laneError: 0.0
  },

  // Physical parameters (in meters)
  trackWidth: 0.129,       // Distancia entre ruedas izq y der (129 mm)
  wheelbase: 0.108,        // Distancia entre ejes del/tras (108 mm)
  wheelRadius: 0.0325,     // Radio de la rueda (32.5 mm -> diam 65 mm)
  maxLinearSpeed: 0.85,    // m/s max velocidad lineal
  maxAngularSpeed: 4.5,    // rad/s max velocidad de giro (giro tanque)
  
  // Spawn position on Gazebo track straightaway (Gazebo launch: x=0, y=-0.25 -> Three.js x=0, z=+0.25, yaw=-PI/2)
  x: 0.0,
  z: 0.25,
  yaw: -Math.PI / 2,
  vLeft: 0,
  vRight: 0,
  targetVLeft: 0,
  targetVRight: 0,
  wheelRotLeft: 0,
  wheelRotRight: 0,
  
  // Control inputs
  input: { fwd: 0, turn: 0 }
};

// Component Specification Database
const COMPONENT_DB = {
  LiDAR_Hokuyo: {
    category: 'SENSORES LÁSER',
    name: 'Sensor Láser 2D LiDAR (Hokuyo UST-10LX)',
    desc: 'Escáner láser 2D para telemetría Time-of-Flight (ToF). Emite pulsos infrarrojos en abanico horizontal para medir la distancia exacta a las paredes y obstáculos.',
    voltage: '12.0 V DC (Alimentación dedicada)',
    current: '400 mA (Típico en barrido)',
    dimensions: '50.0 \u00d7 50.0 \u00d7 70.0 mm',
    role: 'Navegación autónoma, centrado en carril y seguimiento de paredes (Wall-Follower).',
    notes: 'Ubicado en el frontal superior (X = +0.21m, Z = 0.135m en Gazebo). Publica en ROS 2 en el tópico /scan (sensor_msgs/LaserScan).'
  },
  Placa_superior: {
    category: 'ESTRUCTURA',
    name: 'Placa Superior de Chasis (3.0 mm)',
    desc: 'Placa de soporte estructural en acrílico/PLA sólido de alta resistencia con calados estándar para herrajes y electrónica.',
    voltage: 'N/A (Estructural)',
    current: 'N/A',
    dimensions: '150.0 \u00d7 255.0 \u00d7 3.0 mm',
    role: 'Soporte rígido de protoboard, módulos de control y baterías.',
    notes: 'Cuenta con ranuras para cables y orificios de fijación estándar M3 separados por 30 mm de despeje.'
  },
  Placa_inferior: {
    category: 'ESTRUCTURA',
    name: 'Placa Inferior de Chasis (3.0 mm)',
    desc: 'Chasis base con muescas laterales para las 4 ruedas y ranuras en cruz para encoders de velocidad ópticos.',
    voltage: 'N/A (Estructural)',
    current: 'N/A',
    dimensions: '150.0 \u00d7 255.0 \u00d7 3.0 mm',
    role: 'Soporte mecánico de los 4 motores reductores TT y despeje al suelo.',
    notes: 'Despeje al suelo de 16.5 mm. Fija los soportes acrílicos en T de los motores.'
  },
  Ruedas: {
    category: 'TRACCIÓN',
    name: '4x Ruedas de Goma con Rin Amarillo',
    desc: 'Ruedas de tracción de 65 mm con dibujo de banda antideslizante y rin de plástico inyectado acoplable a eje doble D.',
    voltage: 'N/A (Mecánico)',
    current: 'N/A',
    dimensions: '\u2300 65.0 \u00d7 27.0 mm (cada una)',
    role: 'Transferencia de par motor al suelo mediante fricción lateral.',
    notes: 'En tracción diferencial, las ruedas derrapan levemente al girar (skid-steer).'
  },
  Motores: {
    category: 'ACTUADORES',
    name: '4x Motores DC con Reductora TT (1:48)',
    desc: 'Motores de corriente continua con caja reductora de engranajes de plástico para alto torque a baja velocidad.',
    voltage: '3.0 V a 6.0 V DC (Alimentados a ~7.4V vía L298N)',
    current: '150 mA sin carga / 1.1 A en bloqueo (stall)',
    dimensions: '70.0 \u00d7 22.5 \u00d7 19.0 mm',
    role: 'Accionamiento independiente de cada una de las 4 ruedas.',
    notes: 'Conectados en paralelo de a pares: 2 motores izquierdos y 2 motores derechos.'
  },
  Protoboard: {
    category: 'MONTAJE',
    name: 'Protoboard Solderless (400 Puntos)',
    desc: 'Placa de pruebas sin soldadura con bus de alimentación dual y matriz central para inserción de pines.',
    voltage: 'Línea de 5.0 V y GND común',
    current: 'Hasta 2.0 A máx en buses',
    dimensions: '54.0 \u00d7 82.0 \u00d7 9.0 mm',
    role: 'Plataforma de anclaje para ESP32-WROOM y bracket de ESP32-CAM.',
    notes: 'Facilita la interconexión rápida con cables jumper sin requerir soldaduras fijas.'
  },
  ESP32_CAM: {
    category: 'VISIÓN ARTIFICIAL',
    name: 'Módulo ESP32-CAM + Cámara OV2640',
    desc: 'Módulo SoC con procesador Xtensa de 32 bits, conectividad Wi-Fi/BLE y sensor de cámara de 2 Megapíxeles.',
    voltage: '5.0 V DC (Regulado por LM2596S)',
    current: '180 mA típico / Picos de 500 mA durante Wi-Fi',
    dimensions: '27.0 \u00d7 40.0 \u00d7 1.6 mm',
    role: 'Captura y streaming de video IP en tiempo real hacia red local o CNN.',
    notes: 'Montado en bracket vertical frontal. Conectado vía UART (U0T/U0R) al ESP32 para telemetría.'
  },
  ESP32_WROOM: {
    category: 'PROCESAMIENTO',
    name: 'Placa de Desarrollo ESP32-WROOM-32D',
    desc: 'Cerebro central del robot. Maneja la cinemática diferencial, PWM a motores y comunicación remota.',
    voltage: '5.0 V (Entrada VIN / micro-USB) -> 3.3V interno',
    current: '80 mA a 240 mA',
    dimensions: '28.0 \u00d7 51.5 \u00d7 4.8 mm',
    role: 'Genera 4 canales de PWM y señales de dirección IN1-IN4 para el L298N.',
    notes: 'Alivia a la ESP32-CAM del trabajo motriz pesado, evitando caídas de tensión (brownout).'
  },
  Cables: {
    category: 'INTERCONEXIÓN',
    name: 'Arnés de Cables Jumper Dupont',
    desc: 'Líneas flexibles de cobre multifilar con terminales Dupont hembra/macho de 2.54 mm.',
    voltage: '5.0 V (Rojo), GND (Negro), Señal (Amarillo/Azul)',
    current: 'Calibre 26 AWG (~1.0 A máx)',
    dimensions: 'Longitud ~100 mm',
    role: 'Conecta las líneas de alimentación y comunicación serie entre ambas placas.',
    notes: 'Forman un arco curvo organizado para evitar rozamientos con las ruedas delanteras.'
  },
  LM2596S: {
    category: 'POTENCIA',
    name: 'Regulador DC-DC Step-Down LM2596S',
    desc: 'Convertidor reductor conmutado de alta eficiencia (hasta 92%) con potenciómetro multivuelta de ajuste.',
    voltage: 'Entrada: 7.4V - 8.4V (Baterías) | Salida: 5.0V exactos',
    current: '2.0 A continuo / 3.0 A máximo con disipación',
    dimensions: '21.0 \u00d7 43.5 \u00d7 14.0 mm',
    role: 'Alimenta con voltaje ultra limpio y estable a ambos microcontroladores ESP32.',
    notes: 'Aísla la electrónica sensible del ruido eléctrico y caídas bruscas causadas por los motores.'
  },
  L298N: {
    category: 'POTENCIA / MOTORES',
    name: 'Módulo Controlador de Motores L298N',
    desc: 'Puente H dual de potencia con disipador térmico de aluminio y bornes a tornillo.',
    voltage: 'Bornera VMS: 7.4V - 8.4V | Lógica: 5.0V',
    current: 'Hasta 2.0 A por canal (4.0 A total)',
    dimensions: '43.5 \u00d7 43.5 \u00d7 26.0 mm',
    role: 'Invierte el giro y modula la velocidad (PWM) de los 4 motores reductores.',
    notes: 'Permite el giro en el sitio invirtiendo el sentido entre el canal A y el canal B.'
  },
  Chasis_Jeep: {
    category: 'CARROCERÍA / OFF-ROAD',
    name: 'Carrocería Mini-Jeep Off-Road',
    desc: 'Carrocería todoterreno estilo Mini-Jeep diseñada a medida en Blender para las dimensiones exactas del robot, protegiendo los componentes sin alterar la cinemática.',
    voltage: 'N/A (Carrocería)',
    current: 'N/A',
    dimensions: '160.0 \u00d7 265.0 \u00d7 95.0 mm',
    role: 'Protección estructural, rigidez y despeje aerodinámico para pruebas de pista.',
    notes: 'Diseñado respetando los arcos de rueda y anclaje a pilares de suspensión sin bloquear el LiDAR ni la cámara.'
  },
  Baterias: {
    category: 'ENERGÍA',
    name: 'Portabaterías 2\u00d7 18650 con Celdas Li-ion',
    desc: 'Portapilas cerrado para 2 celdas recargables de ion de litio 18650 conectadas en serie (2S).',
    voltage: '7.4 V nominal (8.4 V a plena carga, 6.0 V corte)',
    current: 'Capacidad ~2600 mAh por celda (descarga 5C)',
    dimensions: '77.0 \u00d7 42.0 \u00d7 20.0 mm (montaje transversal)',
    role: 'Fuente primaria de alimentación eléctrica de todo el sistema robótico.',
    notes: 'Ubicado en la popa para contrarrestar el peso de la cámara y equilibrar el centro de gravedad.'
  }
};

/* ==========================================================================
   CONTROLADOR PID (Algoritmo 1:1 de artudo_wall_follower_node.py)
   ========================================================================== */
class PIDController {
  constructor(kp = 1.8, ki = 0.001, kd = 0.5) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
    this.prevError = 0.0;
    this.integral = 0.0;
  }

  reset() {
    this.prevError = 0.0;
    this.integral = 0.0;
  }

  update(error, dt = 0.05) {
    this.integral += error * dt;
    this.integral = THREE.MathUtils.clamp(this.integral, -1.5, 1.5);
    const derivative = dt > 0 ? (error - this.prevError) / dt : 0.0;
    this.prevError = error;
    return (this.kp * error) + (this.ki * this.integral) + (this.kd * derivative);
  }
}

// --- Algoritmo de Intersección Rayo-Segmento 2D de Alto Rendimiento (500x más rápido) ---
function intersectRaySegments2D(ox, oz, dx, dz, maxRange, segments) {
  let closestDist = maxRange;
  let hitX = ox + dx * maxRange;
  let hitZ = oz + dz * maxRange;

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const x1 = seg[0][0];
    const z1 = seg[0][1];
    const x2 = seg[1][0];
    const z2 = seg[1][1];

    const v2x = x2 - x1;
    const v2z = z2 - z1;
    const det = v2x * dz - v2z * dx;
    if (Math.abs(det) < 1e-7) continue;

    const v1x = ox - x1;
    const v1z = oz - z1;
    const invDet = 1.0 / det;

    const t = (v2z * v1x - v2x * v1z) * invDet;
    if (t <= 0.005 || t >= closestDist) continue;

    const u = (dz * v1x - dx * v1z) * invDet;
    if (u < 0.0 || u > 1.0) continue;

    closestDist = t;
    hitX = ox + dx * t;
    hitZ = oz + dz * t;
  }

  return { dist: closestDist, x: hitX, z: hitZ };
}

// --- Resolución de Colisiones Físicas contra las Paredes (Impide atravesarlas) ---
function resolveWallCollisions(pos, radius, segments) {
  let hasCollided = false;
  // 3 iteraciones de relajación para esquinas agudas donde convergen 2 segmentos
  for (let iter = 0; iter < 3; iter++) {
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      const x1 = seg[0][0];
      const z1 = seg[0][1];
      const x2 = seg[1][0];
      const z2 = seg[1][1];

      const dx = x2 - x1;
      const dz = z2 - z1;
      const lenSq = dx * dx + dz * dz;
      if (lenSq < 1e-6) continue;

      let t = ((pos.x - x1) * dx + (pos.z - z1) * dz) / lenSq;
      if (t < 0) t = 0;
      else if (t > 1) t = 1;

      const closestX = x1 + t * dx;
      const closestZ = z1 + t * dz;

      const distX = pos.x - closestX;
      const distZ = pos.z - closestZ;
      const distSq = distX * distX + distZ * distZ;

      if (distSq < radius * radius) {
        hasCollided = true;
        const dist = Math.sqrt(distSq);
        let nx, nz;
        if (dist < 1e-4) {
          const segLen = Math.sqrt(lenSq);
          nx = -dz / segLen;
          nz = dx / segLen;
        } else {
          nx = distX / dist;
          nz = distZ / dist;
        }

        const penetration = radius - dist;
        pos.x += nx * penetration;
        pos.z += nz * penetration;
      }
    }
  }
  return hasCollided;
}

/* ==========================================================================
   SIMULADOR DE SENSOR LÁSER 2D (LiDAR Hokuyo Emulation)
   ========================================================================== */
class LaserScanner {
  constructor(scene, robotRoot) {
    this.scene = scene;
    this.robotRoot = robotRoot;
    this.raycaster = new THREE.Raycaster();
    this.raycaster.far = 10.0;
    this.targets = [];

    // Abanico horizontal de 25 haces distribuidos entre -120° y +120°
    this.numRays = 25;
    this.angles = [];
    const minAngle = -THREE.MathUtils.degToRad(120);
    const maxAngle = THREE.MathUtils.degToRad(120);
    for (let i = 0; i < this.numRays; i++) {
      this.angles.push(minAngle + (i / (this.numRays - 1)) * (maxAngle - minAngle));
    }

    this.distances = new Float32Array(this.numRays);
    this.hitPoints = [];
    for (let i = 0; i < this.numRays; i++) {
      this.hitPoints.push(new THREE.Vector3());
    }

    this.initVisualRays();
  }

  setTargets(meshList) {
    this.targets = meshList;
  }

  initVisualRays() {
    const vertexCount = this.numRays * 2;
    const positions = new Float32Array(vertexCount * 3);
    const colors = new Float32Array(vertexCount * 3);

    for (let i = 0; i < this.numRays; i++) {
      // Origen en el sensor: Cian neón brillante
      colors[i * 6 + 0] = 0.0;
      colors[i * 6 + 1] = 0.95;
      colors[i * 6 + 2] = 1.0;
      // Punto de impacto: Verde esmeralda con gradiente
      colors[i * 6 + 3] = 0.15;
      colors[i * 6 + 4] = 0.95;
      colors[i * 6 + 5] = 0.45;
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    this.material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      linewidth: 1.5
    });

    this.laserLines = new THREE.LineSegments(this.geometry, this.material);
    this.laserLines.renderOrder = 999;
    this.laserLines.frustumCulled = false; // Evita que los láseres desaparezcan por culling dinámico
    this.scene.add(this.laserLines);

    // Nube de puntos de impacto en las paredes
    const hitGeo = new THREE.BufferGeometry();
    const hitPos = new Float32Array(this.numRays * 3);
    hitGeo.setAttribute('position', new THREE.BufferAttribute(hitPos, 3));

    const hitMat = new THREE.PointsMaterial({
      color: 0x00f2fe,
      size: 0.045,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending
    });

    this.hitDots = new THREE.Points(hitGeo, hitMat);
    this.hitDots.renderOrder = 1000;
    this.hitDots.frustumCulled = false; // Puntos siempre visibles
    this.scene.add(this.hitDots);
  }

  setVisible(visible) {
    this.laserLines.visible = visible;
    this.hitDots.visible = visible;
  }

  update(carYaw) {
    const worldPos = new THREE.Vector3();
    this.robotRoot.getWorldPosition(worldPos);

    // Vector frontal del carro: -sin(yaw), -cos(yaw) en Three.js
    const forwardX = -Math.sin(carYaw);
    const forwardZ = -Math.cos(carYaw);

    // Posición del sensor Hokuyo en el paragolpes frontal
    const originX = worldPos.x + forwardX * 0.105;
    const originY = 0.045;
    const originZ = worldPos.z + forwardZ * 0.105;

    const positions = this.geometry.attributes.position.array;
    const hitPositions = this.hitDots.geometry.attributes.position.array;

    let minLeftDist = 10.0;
    let minRightDist = 10.0;
    let minFrontDist = 10.0;
    let frontLeftDist = 10.0;
    let frontRightDist = 10.0;

    for (let i = 0; i < this.numRays; i++) {
      const relAngle = this.angles[i];
      const absAngle = carYaw + relAngle;

      const rayDirX = -Math.sin(absAngle);
      const rayDirZ = -Math.cos(absAngle);

      // Intersección 2D ultrarrápida contra los segmentos de pared extraídos de Gazebo
      const hit = intersectRaySegments2D(originX, originZ, rayDirX, rayDirZ, 8.0, paredes2DData);
      const distance = hit.dist;

      this.distances[i] = distance;
      this.hitPoints[i].set(hit.x, originY, hit.z);

      // Línea de rayo láser (de origen al punto de impacto)
      positions[i * 6 + 0] = originX;
      positions[i * 6 + 1] = originY;
      positions[i * 6 + 2] = originZ;
      positions[i * 6 + 3] = hit.x;
      positions[i * 6 + 4] = originY;
      positions[i * 6 + 5] = hit.z;

      // Punto de contacto luminoso sobre la pared
      hitPositions[i * 3 + 0] = hit.x;
      hitPositions[i * 3 + 1] = originY;
      hitPositions[i * 3 + 2] = hit.z;

      // Clasificación de sectores para la lógica del Wall-Follower PID
      const deg = THREE.MathUtils.radToDeg(relAngle);
      if (deg <= -65 && deg >= -115) {
        minLeftDist = Math.min(minLeftDist, distance);
      } else if (deg >= 65 && deg <= 115) {
        minRightDist = Math.min(minRightDist, distance);
      } else if (Math.abs(deg) <= 15) {
        minFrontDist = Math.min(minFrontDist, distance);
      } else if (deg >= -55 && deg <= -35) {
        frontLeftDist = Math.min(frontLeftDist, distance);
      } else if (deg >= 35 && deg <= 55) {
        frontRightDist = Math.min(frontRightDist, distance);
      }
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.hitDots.geometry.attributes.position.needsUpdate = true;

    // Error de centrado entre paredes izquierda y derecha
    const laneError = (minLeftDist - minRightDist) * 0.5;

    return {
      left: minLeftDist,
      right: minRightDist,
      front: minFrontDist,
      frontLeft: frontLeftDist,
      frontRight: frontRightDist,
      laneError: laneError
    };
  }
}

/* ==========================================================================
   PILOTO AUTÓNOMO: SEGUIDOR DE PARED Y CENTRADO EN CARRIL (PID)
   Reproduce la lógica de navegación de Gazebo (artudo_wall_follower_node.py)
   ========================================================================== */
class WallFollowerAutopilot {
  constructor() {
    this.pid = new PIDController(1.8, 0.001, 0.5);
    this.lookaheadDist = 0.8; // L en artudo_wall_follower_node.py
  }

  reset() {
    this.pid.reset();
  }

  computeControl(laserData, dt) {
    if (!laserData) return { fwd: 0, turn: 0 };

    const { left, right, front, frontLeft, frontRight, laneError } = laserData;

    // Frenado y evasión si hay una pared u obstáculo frontal directo
    if (front < 0.65) {
      const evadeTurn = (left > right) ? 1.0 : -1.0;
      return {
        fwd: 0.15,
        turn: evadeTurn
      };
    }

    let steering = 0.0;

    // Cuando ambas paredes son visibles con precisión (< 4.5m)
    if (left < 4.5 && right < 4.5) {
      // Centrado perfecto entre las dos paredes:
      // error > 0: más cerca de la derecha -> girar a la izquierda (+turn)
      // error < 0: más cerca de la izquierda -> girar a la derecha (-turn)
      steering = this.pid.update(laneError, dt);
    } else if (right < 4.5) {
      // Seguidor de pared derecha (fórmula exacta de artudo_wall_follower_node.py)
      const a = frontRight;
      const b = right;
      const theta = Math.PI / 4; // 45°
      const alpha = Math.atan2(a * Math.cos(theta) - b, a * Math.sin(theta));
      const currentDist = b * Math.cos(alpha);
      const predictedDist = currentDist + this.lookaheadDist * Math.sin(alpha);
      const targetDist = 1.05; // Ancho medio de carril
      const error = targetDist - predictedDist;
      steering = this.pid.update(error, dt);
    } else if (left < 4.5) {
      // Seguidor de pared izquierda simétrico
      const a = frontLeft;
      const b = left;
      const theta = Math.PI / 4;
      const alpha = Math.atan2(a * Math.cos(theta) - b, a * Math.sin(theta));
      const currentDist = b * Math.cos(alpha);
      const predictedDist = currentDist + this.lookaheadDist * Math.sin(alpha);
      const targetDist = 1.05;
      const error = predictedDist - targetDist;
      steering = this.pid.update(error, dt);
    } else {
      this.pid.reset();
      steering = 0.0;
    }

    // Normalizar giro a rango [-1.0, 1.0]
    const turnInput = THREE.MathUtils.clamp(steering / 1.5, -1.0, 1.0);

    // Reducción proporcional de velocidad en curvas cerradas
    const speedFactor = 1.0 - 0.45 * Math.abs(turnInput);
    const fwdInput = THREE.MathUtils.clamp(speedFactor, 0.35, 0.85);

    return {
      fwd: fwdInput,
      turn: turnInput
    };
  }
}

// --- Web Audio Synthesizer ---
class SoundSynth {
  constructor() {
    this.ctx = null;
    this.osc1 = null;
    this.osc2 = null;
    this.gain = null;
    this.filter = null;
    this.initialized = false;
  }

  init() {
    if (this.initialized) return;
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioContext();

      this.osc1 = this.ctx.createOscillator();
      this.osc2 = this.ctx.createOscillator();
      this.gain = this.ctx.createGain();
      this.filter = this.ctx.createBiquadFilter();

      this.osc1.type = 'sawtooth';
      this.osc2.type = 'square';
      this.osc1.frequency.value = 60;
      this.osc2.frequency.value = 120;

      this.filter.type = 'lowpass';
      this.filter.frequency.value = 450;

      this.gain.gain.value = 0.0;

      this.osc1.connect(this.filter);
      this.osc2.connect(this.filter);
      this.filter.connect(this.gain);
      this.gain.connect(this.ctx.destination);

      this.osc1.start();
      this.osc2.start();
      this.initialized = true;
    } catch (e) {
      console.warn("Web Audio not supported or blocked:", e);
    }
  }

  update(speedMagnitude, isTurning) {
    if (!this.initialized || !state.soundEnabled) {
      if (this.gain) this.gain.gain.value = 0;
      return;
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
    const targetGain = Math.min(0.12, speedMagnitude * 0.12 + (isTurning ? 0.04 : 0));
    this.gain.gain.setTargetAtTime(targetGain, this.ctx.currentTime, 0.05);

    const baseFreq = 50 + speedMagnitude * 140 + (isTurning ? 40 : 0);
    this.osc1.frequency.setTargetAtTime(baseFreq, this.ctx.currentTime, 0.05);
    this.osc2.frequency.setTargetAtTime(baseFreq * 2.1, this.ctx.currentTime, 0.05);
  }

  click() {
    if (!this.initialized || !state.soundEnabled) return;
    const osc = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 800;
    g.gain.value = 0.08;
    g.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.04);
    osc.connect(g);
    g.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.05);
  }
}

const synth = new SoundSynth();

// --- Three.js Application Core ---
class App3D {
  constructor() {
    this.container = document.getElementById('canvas-container');
    this.fpvCanvas = document.getElementById('fpv-canvas');
    
    // Scene & Renderers
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1322);
    this.scene.fog = new THREE.FogExp2(0x0c1322, 0.025);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.container.appendChild(this.renderer.domElement);

    // FPV Secondary Renderer
    this.fpvRenderer = new THREE.WebGLRenderer({ canvas: this.fpvCanvas, antialias: true });
    this.fpvRenderer.setSize(320, 240);
    this.fpvRenderer.setPixelRatio(1);

    // Main Orbit / Free Camera
    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.02, 150);
    this.camera.position.set(0.35, 0.32, 0.40);

    // FPV Camera: ESP32-CAM OV2640 Lens Camera
    this.fpvCamera = new THREE.PerspectiveCamera(66, 320 / 240, 0.01, 80);

    // Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.02;
    this.controls.minDistance = 0.08;
    this.controls.maxDistance = 25;
    this.controls.target.set(0, 0.04, 0);

    // Robot Root Node
    this.robotRoot = new THREE.Group();
    this.scene.add(this.robotRoot);

    this.parts = {
      chassisBottom: null,
      chassisTop: null,
      wheels: { fl: null, fr: null, rl: null, rr: null },
      motors: { fl: null, fr: null, rl: null, rr: null }
    };

    this.explodedNodes = [];
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.clock = new THREE.Clock();

    // Laser Scanner & Autopilot Modules
    this.collisionMeshes = [];
    this.laserScanner = new LaserScanner(this.scene, this.robotRoot);
    this.autopilot = new WallFollowerAutopilot();

    this.initLights();
    this.buildGazeboTrack(); // Authentic Gazebo track
    this.loadRobotModel();
    this.setupUI();
    this.setupKeyboard();

    window.addEventListener('resize', () => this.onResize());
    this.renderer.domElement.addEventListener('pointerdown', (e) => this.onPointerClick(e));

    this.animate();
  }

  initLights() {
    const ambientLight = new THREE.AmbientLight(0xddeeff, 0.85);
    this.scene.add(ambientLight);

    const sunLight = new THREE.DirectionalLight(0xffffff, 1.4);
    sunLight.position.set(10, 25, 15);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.width = 2048;
    sunLight.shadow.mapSize.height = 2048;
    sunLight.shadow.camera.near = 0.5;
    sunLight.shadow.camera.far = 60;
    sunLight.shadow.camera.left = -20;
    sunLight.shadow.camera.right = 20;
    sunLight.shadow.camera.top = 20;
    sunLight.shadow.camera.bottom = -20;
    sunLight.shadow.bias = -0.0005;
    this.scene.add(sunLight);

    const rimCyan = new THREE.DirectionalLight(0x00f2fe, 0.5);
    rimCyan.position.set(-15, 10, -15);
    this.scene.add(rimCyan);
  }

  // --- Authentic Gazebo Track Construction ---
  buildGazeboTrack() {
    this.trackGroup = new THREE.Group();
    this.scene.add(this.trackGroup);

    console.log("Loading Gazebo racetrack geometry...");

    // Iterate through the 5 parts extracted from Gazebo's meshes/track.dae
    pistaGazeboData.forEach((part) => {
      const geo = new THREE.BufferGeometry();
      const rawVerts = part.vertices;
      
      // Remap coordinates from Collada Z_UP to Three.js Y_UP:
      // Three.js X = Collada X
      // Three.js Y = Collada Z (height, slightly above 0)
      // Three.js Z = -Collada Y (forward)
      const positions = new Float32Array(rawVerts.length);
      for (let i = 0; i < rawVerts.length; i += 3) {
        positions[i]     = rawVerts[i];         // X
        positions[i + 1] = rawVerts[i + 2] + 0.002; // Y (height)
        positions[i + 2] = -rawVerts[i + 1];    // Z
      }

      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setIndex(part.indices);
      geo.computeVertexNormals();

      // Materials matching Gazebo diffuse colors
      const col = new THREE.Color(part.color[0], part.color[1], part.color[2]);
      const isRoad = part.color[0] < 0.35 && part.color[1] < 0.35;
      const isCurb = part.color[0] > 0.8 || part.color[1] > 0.8;

      const mat = new THREE.MeshStandardMaterial({
        color: col,
        roughness: isRoad ? 0.9 : 0.6,
        metalness: isRoad ? 0.05 : 0.1,
        side: THREE.DoubleSide
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.receiveShadow = true;
      mesh.castShadow = isCurb;
      this.trackGroup.add(mesh);
    });

    // Outer Grass Ground Plane surrounding the track
    const groundGeo = new THREE.PlaneGeometry(120, 120);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x244a36, // Gazebo outer grass green
      roughness: 0.95,
      metalness: 0.0
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -0.005;
    ground.receiveShadow = true;
    this.trackGroup.add(ground);

    // Build Gazebo Track Boundary Walls (meshes/walls.dae converted)
    if (paredesGazeboData && paredesGazeboData.vertices) {
      const wallGeo = new THREE.BufferGeometry();
      const rawWallVerts = paredesGazeboData.vertices;
      const wallPositions = new Float32Array(rawWallVerts.length);
      for (let i = 0; i < rawWallVerts.length; i += 3) {
        wallPositions[i]     = rawWallVerts[i];              // X
        wallPositions[i + 1] = rawWallVerts[i + 2] + 0.005;  // Y (Height)
        wallPositions[i + 2] = -rawWallVerts[i + 1];         // Z
      }
      wallGeo.setAttribute('position', new THREE.BufferAttribute(wallPositions, 3));
      wallGeo.setIndex(paredesGazeboData.indices);
      wallGeo.computeVertexNormals();

      const wallMat = new THREE.MeshStandardMaterial({
        color: 0x8a929e, // Concreto gris sólido / barrera curb de Gazebo
        roughness: 0.8,
        metalness: 0.1,
        transparent: false,
        side: THREE.FrontSide
      });

      this.wallMesh = new THREE.Mesh(wallGeo, wallMat);
      this.wallMesh.receiveShadow = true;
      this.wallMesh.castShadow = true;
      this.wallMesh.name = "ParedesGazebo";
      this.trackGroup.add(this.wallMesh);

      this.collisionMeshes = [this.wallMesh];
      if (this.laserScanner) {
        this.laserScanner.setTargets(this.collisionMeshes);
      }
    }

    // Posición inicial en la recta de Gazebo (x=0, y=-0.25 en Gazebo -> x=0, z=+0.25, yaw=-PI/2 en Three.js)
    state.x = 0.0;
    state.z = 0.25;
    state.yaw = -Math.PI / 2; // Orientado hacia adelante a lo largo de la pista (+X)
    this.robotRoot.position.set(state.x, 0, state.z);
    this.robotRoot.rotation.y = state.yaw;

    console.log("Gazebo track successfully initialized!");
  }

  loadRobotModel() {
    const loader = new GLTFLoader();
    const glbUrl = 'carro_4wd_robot.glb';

    loader.load(
      glbUrl,
      (gltf) => {
        console.log("Loaded authentic GLB robot model!");
        gltf.scene.scale.set(0.001, 0.001, 0.001);
        gltf.scene.traverse((node) => {
          if (node.isMesh) {
            node.castShadow = true;
            node.receiveShadow = true;
            this.registerGlbPart(node);
          }
        });
        this.robotRoot.add(gltf.scene);
        this.buildHokuyoLidar();
        this.setupExplodedNodes();
        this.setupFPVCamera();
      },
      undefined,
      (error) => {
        console.warn("Generating fallback procedural model...", error);
        this.buildProceduralModel();
        this.buildHokuyoLidar();
        this.setupExplodedNodes();
        this.setupFPVCamera();
      }
    );
  }

  registerGlbPart(mesh) {
    const name = mesh.name.toLowerCase();
    
    if (name.includes('wheel_fl') || name.includes('rim') && name.includes('fl')) {
      mesh.userData.component = 'Ruedas';
      this.parts.wheels.fl = mesh;
    } else if (name.includes('wheel_fr') || name.includes('rim') && name.includes('fr')) {
      mesh.userData.component = 'Ruedas';
      this.parts.wheels.fr = mesh;
    } else if (name.includes('wheel_rl') || name.includes('rim') && name.includes('rl')) {
      mesh.userData.component = 'Ruedas';
      this.parts.wheels.rl = mesh;
    } else if (name.includes('wheel_rr') || name.includes('rim') && name.includes('rr')) {
      mesh.userData.component = 'Ruedas';
      this.parts.wheels.rr = mesh;
    } else if (name.includes('cam') || name.includes('ov2640')) {
      mesh.userData.component = 'ESP32_CAM';
    } else if (name.includes('esp32') && !name.includes('cam')) {
      mesh.userData.component = 'ESP32_WROOM';
    } else if (name.includes('proto') || name.includes('bread')) {
      mesh.userData.component = 'Protoboard';
    } else if (name.includes('lm2596')) {
      mesh.userData.component = 'LM2596S';
    } else if (name.includes('l298')) {
      mesh.userData.component = 'L298N';
    } else if (name.includes('bat') || name.includes('18650')) {
      mesh.userData.component = 'Baterias';
    } else if (name.includes('motor')) {
      mesh.userData.component = 'Motores';
    } else if (name.includes('jeep') || name.includes('carroceria') || name.includes('offroad') || (name.includes('chasis') && !name.includes('inferior') && !name.includes('superior'))) {
      mesh.userData.component = 'Chasis_Jeep';
    } else if (name.includes('superior') || name.includes('top')) {
      mesh.userData.component = 'Placa_superior';
    } else if (name.includes('inferior') || name.includes('bottom')) {
      mesh.userData.component = 'Placa_inferior';
    } else if (name.includes('wire') || name.includes('cable')) {
      mesh.userData.component = 'Cables';
    }
  }

  buildProceduralModel() {
    const mPlate = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.35, metalness: 0.1 });
    const mYellow = new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.3 });
    const mRubber = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const mBrass = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.25, metalness: 0.9 });
    const mSilver = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2, metalness: 0.95 });
    const mBlack = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.5 });
    const mWhite = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.3 });
    const mRed = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.3 });
    const mBlue = new THREE.MeshStandardMaterial({ color: 0x1d4ed8, roughness: 0.3 });

    // Chassis Plates
    const plateGeo = new THREE.BoxGeometry(0.150, 0.003, 0.255);
    const plateBot = new THREE.Mesh(plateGeo, mPlate);
    plateBot.position.set(0, 0.018, 0);
    plateBot.castShadow = true;
    plateBot.userData.component = 'Placa_inferior';
    this.robotRoot.add(plateBot);

    const plateTop = new THREE.Mesh(plateGeo, mPlate);
    plateTop.position.set(0, 0.051, 0);
    plateTop.castShadow = true;
    plateTop.userData.component = 'Placa_superior';
    this.robotRoot.add(plateTop);

    // 4 Wheels & Motors
    // Notice: in glTF/Three.js frame, FRONT is -Z and REAR is +Z
    const wheelGeo = new THREE.CylinderGeometry(0.0325, 0.0325, 0.027, 24);
    wheelGeo.rotateZ(Math.PI / 2);
    const rimGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.028, 24);
    rimGeo.rotateZ(Math.PI / 2);

    const wheelPos = [
      { id: 'fl', x: -0.0645, z: -0.054 }, // Front Left
      { id: 'fr', x: 0.0645, z: -0.054 },  // Front Right
      { id: 'rl', x: -0.0645, z: 0.054 },  // Rear Left
      { id: 'rr', x: 0.0645, z: 0.054 }   // Rear Right
    ];

    wheelPos.forEach((p) => {
      const wGroup = new THREE.Group();
      wGroup.position.set(p.x, 0.0325, p.z);
      
      const tire = new THREE.Mesh(wheelGeo, mRubber);
      tire.castShadow = true;
      tire.userData.component = 'Ruedas';
      wGroup.add(tire);

      const rim = new THREE.Mesh(rimGeo, mYellow);
      rim.userData.component = 'Ruedas';
      wGroup.add(rim);

      this.robotRoot.add(wGroup);
      this.parts.wheels[p.id] = wGroup;

      const motorGeo = new THREE.BoxGeometry(0.022, 0.022, 0.054);
      const motor = new THREE.Mesh(motorGeo, mYellow);
      motor.position.set(p.x * 0.72, 0.0325, p.z);
      motor.castShadow = true;
      motor.userData.component = 'Motores';
      this.robotRoot.add(motor);
    });

    // Standoffs (8x)
    const standoffGeo = new THREE.CylinderGeometry(0.003, 0.003, 0.030, 8);
    const sOffsets = [
      [-0.052, -0.088], [0.052, -0.088],
      [-0.062, 0.000], [0.062, 0.000],
      [-0.052, 0.088], [0.052, 0.088]
    ];
    sOffsets.forEach(([sx, sz]) => {
      const so = new THREE.Mesh(standoffGeo, mBrass);
      so.position.set(sx, 0.0345, sz);
      this.robotRoot.add(so);
    });

    // Electronics on Top Plate:
    // 1. Breadboard (at front: -Z)
    const bbGeo = new THREE.BoxGeometry(0.054, 0.009, 0.082);
    const bb = new THREE.Mesh(bbGeo, mWhite);
    bb.position.set(0, 0.057, -0.048);
    bb.castShadow = true;
    bb.userData.component = 'Protoboard';
    this.robotRoot.add(bb);

    // 2. Camera Bracket + ESP32-CAM (Front-most: Z = -0.076)
    const camGroup = new THREE.Group();
    camGroup.position.set(0, 0.0615, -0.076);
    camGroup.userData.component = 'ESP32_CAM';

    const bBase = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.004, 0.014), mBlack);
    camGroup.add(bBase);

    const bBack = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.032, 0.003), mBlack);
    bBack.position.set(0, 0.016, 0.002);
    camGroup.add(bBack);

    const camPcb = new THREE.Mesh(new THREE.BoxGeometry(0.027, 0.040, 0.0016), mBlack);
    camPcb.position.set(0, 0.020, -0.001);
    camGroup.add(camPcb);

    // Lens facing forward (-Z)
    const lensGeo = new THREE.CylinderGeometry(0.0032, 0.0032, 0.004, 16);
    lensGeo.rotateX(Math.PI / 2);
    const lens = new THREE.Mesh(lensGeo, mBlack);
    lens.position.set(0, 0.026, -0.004);
    camGroup.add(lens);

    const redDotGeo = new THREE.CylinderGeometry(0.0008, 0.0008, 0.0005, 12);
    redDotGeo.rotateX(Math.PI / 2);
    const redDot = new THREE.Mesh(redDotGeo, mRed);
    redDot.position.set(0, 0.026, -0.0062);
    camGroup.add(redDot);

    this.robotRoot.add(camGroup);

    // 3. ESP32-WROOM (Z = -0.034)
    const espGroup = new THREE.Group();
    espGroup.position.set(0, 0.0615, -0.034);
    espGroup.userData.component = 'ESP32_WROOM';

    const espPcb = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.0016, 0.0515), mBlack);
    espGroup.add(espPcb);
    const rfShield = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.003, 0.016), mSilver);
    rfShield.position.set(0, 0.002, -0.009);
    espGroup.add(rfShield);
    this.robotRoot.add(espGroup);

    // 4. LM2596S (Z = 0.008)
    const lm = new THREE.Mesh(new THREE.BoxGeometry(0.021, 0.010, 0.0435), mBlue);
    lm.position.set(0, 0.0575, 0.008);
    lm.userData.component = 'LM2596S';
    this.robotRoot.add(lm);

    // 5. L298N Motor Driver (Z = 0.048)
    const l298Group = new THREE.Group();
    l298Group.position.set(0, 0.0525, 0.048);
    l298Group.userData.component = 'L298N';

    const l298Pcb = new THREE.Mesh(new THREE.BoxGeometry(0.0435, 0.0016, 0.0435), mRed);
    l298Group.add(l298Pcb);

    const heatsink = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.024, 0.013), mBlack);
    heatsink.position.set(0, 0.013, -0.007);
    l298Group.add(heatsink);

    const termA = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.010, 0.015), mBlue);
    termA.position.set(-0.017, 0.006, 0);
    l298Group.add(termA);

    const termB = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.010, 0.015), mBlue);
    termB.position.set(0.017, 0.006, 0);
    l298Group.add(termB);

    this.robotRoot.add(l298Group);

    // 6. Transverse Battery Holder with 2x 18650s (At REAR: Z = +0.096)
    const batGroup = new THREE.Group();
    batGroup.position.set(0, 0.0525, 0.096);
    batGroup.userData.component = 'Baterias';

    const tray = new THREE.Mesh(new THREE.BoxGeometry(0.077, 0.018, 0.042), mBlack);
    tray.position.set(0, 0.009, 0);
    batGroup.add(tray);

    const cellGeo = new THREE.CylinderGeometry(0.0092, 0.0092, 0.065, 16);
    cellGeo.rotateZ(Math.PI / 2);
    const cell1 = new THREE.Mesh(cellGeo, mYellow);
    cell1.position.set(0, 0.011, -0.010);
    batGroup.add(cell1);

    const cell2 = new THREE.Mesh(cellGeo, mYellow);
    cell2.position.set(0, 0.011, 0.010);
    batGroup.add(cell2);

    this.robotRoot.add(batGroup);
  }

  setupExplodedNodes() {
    this.explodedNodes = [];
    this.robotRoot.traverse((child) => {
      if (child.isMesh || (child.isGroup && child.userData.component)) {
        const comp = child.userData.component;
        let offset = new THREE.Vector3(0, 0, 0);

        if (comp === 'Placa_superior') {
          offset.set(0, 0.04, 0);
        } else if (comp === 'LiDAR_Hokuyo') {
          offset.set(0, 0.09, -0.04);
        } else if (comp === 'Protoboard' || comp === 'ESP32_CAM' || comp === 'ESP32_WROOM') {
          offset.set(0, 0.08, -0.02);
        } else if (comp === 'LM2596S' || comp === 'L298N') {
          offset.set(0, 0.07, 0.02);
        } else if (comp === 'Baterias') {
          offset.set(0, 0.06, 0.04);
        } else if (comp === 'Ruedas') {
          const side = child.position.x >= 0 ? 1 : -1;
          offset.set(side * 0.06, 0, 0);
        } else if (comp === 'Motores') {
          const side = child.position.x >= 0 ? 1 : -1;
          offset.set(side * 0.03, -0.02, 0);
        } else if (comp === 'Placa_inferior') {
          offset.set(0, -0.03, 0);
        }

        if (offset.lengthSq() > 0) {
          this.explodedNodes.push({
            obj: child,
            origPos: child.position.clone(),
            offsetDir: offset
          });
        }
      }
    });
  }

  // --- Exact FPV Camera Setup on ESP32-CAM Lens ---
  setupFPVCamera() {
    this.fpvCamera.position.set(0.0, 0.0885, -0.085);
    this.fpvCamera.rotation.set(0, 0, 0); // Faces -Z by default in Three.js
    this.robotRoot.add(this.fpvCamera);
    console.log("FPV Camera mounted directly on ESP32-CAM OV2640 lens (facing -Z).");
  }

  // --- 3D Hokuyo UST-10LX 2D LiDAR Construction ---
  buildHokuyoLidar() {
    if (this.parts.lidar) return; // Prevent duplicate

    const lidarGroup = new THREE.Group();
    lidarGroup.name = "LiDAR_Hokuyo";
    lidarGroup.userData.component = 'LiDAR_Hokuyo';

    const darkMat = new THREE.MeshStandardMaterial({
      color: 0x1a202c,
      roughness: 0.35,
      metalness: 0.75
    });

    // Lower cylinder base
    const baseGeo = new THREE.CylinderGeometry(0.016, 0.018, 0.012, 20);
    const baseMesh = new THREE.Mesh(baseGeo, darkMat);
    baseMesh.position.y = 0.006;
    baseMesh.castShadow = true;
    baseMesh.userData.component = 'LiDAR_Hokuyo';
    lidarGroup.add(baseMesh);

    // Glowing cyan/amber optical scanning window
    const ringGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.009, 20);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x00f2fe,
      emissive: 0x00c4df,
      emissiveIntensity: 0.6,
      roughness: 0.15,
      metalness: 0.1,
      transparent: true,
      opacity: 0.95
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.position.y = 0.0165;
    ringMesh.userData.component = 'LiDAR_Hokuyo';
    lidarGroup.add(ringMesh);

    // Upper cover
    const capGeo = new THREE.CylinderGeometry(0.016, 0.016, 0.006, 20);
    const capMesh = new THREE.Mesh(capGeo, darkMat);
    capMesh.position.y = 0.024;
    capMesh.castShadow = true;
    capMesh.userData.component = 'LiDAR_Hokuyo';
    lidarGroup.add(capMesh);

    // Power / Laser Active Status LED
    const ledGeo = new THREE.SphereGeometry(0.002, 8, 8);
    const ledMat = new THREE.MeshBasicMaterial({ color: 0x00ff88 });
    const ledMesh = new THREE.Mesh(ledGeo, ledMat);
    ledMesh.position.set(0, 0.024, -0.015);
    lidarGroup.add(ledMesh);

    // Mounting at the front bumper top surface (Z = -0.105m, Y = 0.048m above ground)
    lidarGroup.position.set(0.0, 0.048, -0.105);
    this.robotRoot.add(lidarGroup);

    this.parts.lidar = lidarGroup;
  }

  updateExplodedView(factor) {
    this.explodedNodes.forEach((node) => {
      node.obj.position.copy(node.origPos).addScaledVector(node.offsetDir, factor);
    });
  }

  // --- Kinematics & Simulation Loop ---
  updateSimulation(delta) {
    if (state.mode !== 'sim') {
      // Keep laser visual hidden in CAD mode
      if (this.laserScanner) this.laserScanner.setVisible(false);
      return;
    } else {
      if (this.laserScanner) this.laserScanner.setVisible(state.laserVisible);
    }

    // 1. Live LiDAR 2D Raycast Scan
    const laserData = this.laserScanner.update(state.yaw);
    if (laserData) {
      state.laserDistances = laserData;
    }

    // 2. Autonomous Wall-Follower / Centering PID Controller
    if (state.autopilot) {
      const autoCmd = this.autopilot.computeControl(laserData, delta);
      state.input.fwd = autoCmd.fwd;
      state.input.turn = autoCmd.turn;
    }

    // Inputs: fwd = +1 (forward), -1 (reverse); turn = +1 (left), -1 (right)
    const fwdInput = state.input.fwd;
    const turnInput = state.input.turn;

    // Differential Skid-Steer Kinematics:
    // When turning left: left wheel reverses/slows, right wheel advances
    const baseLinear = fwdInput * state.maxLinearSpeed;
    const baseAngular = turnInput * state.maxAngularSpeed;

    state.targetVLeft = baseLinear - (baseAngular * state.trackWidth * 0.5);
    state.targetVRight = baseLinear + (baseAngular * state.trackWidth * 0.5);

    // Motor Inertia Smoothing
    const accelRate = 5.0 * delta;
    state.vLeft = THREE.MathUtils.lerp(state.vLeft, state.targetVLeft, accelRate);
    state.vRight = THREE.MathUtils.lerp(state.vRight, state.targetVRight, accelRate);

    // Resultant Linear Velocity (m/s) and Angular Velocity (rad/s)
    const v = (state.vRight + state.vLeft) * 0.5;
    const omega = (state.vRight - state.vLeft) / state.trackWidth;

    // In Three.js: Forward is -Z!
    state.yaw += omega * delta;
    state.x += -v * Math.sin(state.yaw) * delta;
    state.z += -v * Math.cos(state.yaw) * delta;

    // Resolución física de colisiones contra las paredes de la pista (Impide atravesar muros)
    const carCollisionRadius = 0.088; // Radio de colisión envolvente (88 mm, chasis + parachoques)
    const hasCollided = resolveWallCollisions(state, carCollisionRadius, paredes2DData);
    if (hasCollided) {
      // Amortiguación de impacto para deslizarse por la tangente sin penetrar
      state.vLeft *= 0.65;
      state.vRight *= 0.65;
    }

    // Boundary constraint on Gazebo track (-25m to +20m)
    state.x = THREE.MathUtils.clamp(state.x, -25, 20);
    state.z = THREE.MathUtils.clamp(state.z, -25, 20);

    // Apply to Robot
    this.robotRoot.position.set(state.x, 0, state.z);
    this.robotRoot.rotation.y = state.yaw;

    // Individual Wheel Rotation Animation
    const rotSpeedLeft = (state.vLeft / state.wheelRadius) * delta;
    const rotSpeedRight = (state.vRight / state.wheelRadius) * delta;

    state.wheelRotLeft += rotSpeedLeft;
    state.wheelRotRight += rotSpeedRight;

    if (this.parts.wheels.fl) this.parts.wheels.fl.rotation.x = -state.wheelRotLeft;
    if (this.parts.wheels.rl) this.parts.wheels.rl.rotation.x = -state.wheelRotLeft;
    if (this.parts.wheels.fr) this.parts.wheels.fr.rotation.x = -state.wheelRotRight;
    if (this.parts.wheels.rr) this.parts.wheels.rr.rotation.x = -state.wheelRotRight;

    // Audio
    synth.update(Math.abs(v), Math.abs(omega) > 0.1);

    // Telemetry & Laser HUD Updates
    this.updateTelemetryHUD(v, omega);
    this.updateLaserHUD(laserData);

    // Camera Modes
    if (state.simCamMode === 'chase') {
      const chaseOffset = new THREE.Vector3(
        Math.sin(state.yaw) * 0.65,
        0.30,
        Math.cos(state.yaw) * 0.65
      );
      const targetPos = this.robotRoot.position.clone().add(chaseOffset);
      this.camera.position.lerp(targetPos, 0.08);
      this.controls.target.lerp(this.robotRoot.position.clone().add(new THREE.Vector3(0, 0.06, 0)), 0.1);
    } else if (state.simCamMode === 'fpv') {
      const worldPos = new THREE.Vector3();
      const worldQuat = new THREE.Quaternion();
      this.fpvCamera.getWorldPosition(worldPos);
      this.fpvCamera.getWorldQuaternion(worldQuat);
      this.camera.position.copy(worldPos);
      this.camera.quaternion.copy(worldQuat);
    }
  }

  updateLaserHUD(laserData) {
    if (!laserData) return;

    const elL = document.getElementById('telem-dist-l');
    const elF = document.getElementById('telem-dist-f');
    const elR = document.getElementById('telem-dist-r');
    const elErr = document.getElementById('telem-lane-err');
    const elPilot = document.getElementById('telem-pilot-status');

    if (elL) elL.innerHTML = laserData.left < 9.0 ? `${laserData.left.toFixed(2)} <small>m</small>` : `-- <small>m</small>`;
    if (elF) elF.innerHTML = laserData.front < 9.0 ? `${laserData.front.toFixed(2)} <small>m</small>` : `-- <small>m</small>`;
    if (elR) elR.innerHTML = laserData.right < 9.0 ? `${laserData.right.toFixed(2)} <small>m</small>` : `-- <small>m</small>`;

    if (elErr) {
      const errCm = (laserData.laneError * 100).toFixed(1);
      const sign = laserData.laneError > 0 ? '+' : '';
      elErr.innerHTML = `${sign}${errCm} <small>cm</small>`;
      
      const cardErr = document.getElementById('card-dist-err');
      if (cardErr) {
        cardErr.classList.toggle('centered', Math.abs(laserData.laneError) < 0.10);
        cardErr.classList.toggle('warning', Math.abs(laserData.laneError) >= 0.10 && Math.abs(laserData.laneError) < 0.35);
        cardErr.classList.toggle('danger', Math.abs(laserData.laneError) >= 0.35);
      }
    }

    if (elPilot) {
      if (state.autopilot) {
        elPilot.innerText = 'AUTÓNOMO PID';
        elPilot.style.color = 'var(--accent-emerald)';
      } else {
        elPilot.innerText = 'MANUAL';
        elPilot.style.color = 'var(--text-dim)';
      }
    }
  }

  updateTelemetryHUD(v, omega) {
    document.getElementById('telem-v').innerHTML = `${v.toFixed(2)} <small>m/s</small>`;
    document.getElementById('telem-w').innerHTML = `${(omega * 180 / Math.PI).toFixed(1)} <small>&deg;/s</small>`;

    const pwmL = Math.round(Math.abs(state.vLeft / state.maxLinearSpeed) * 255);
    const pwmR = Math.round(Math.abs(state.vRight / state.maxLinearSpeed) * 255);

    document.getElementById('telem-pwm-l').innerHTML = `${pwmL} <small>PWM</small>`;
    document.getElementById('telem-pwm-r').innerHTML = `${pwmR} <small>PWM</small>`;

    const statusBadge = document.getElementById('telem-status');
    if (Math.abs(v) < 0.01 && Math.abs(omega) < 0.05) {
      statusBadge.innerText = 'DETENIDO';
      statusBadge.style.color = 'var(--text-dim)';
    } else if (Math.abs(v) < 0.05 && Math.abs(omega) >= 0.1) {
      statusBadge.innerText = omega > 0 ? 'GIRO TANQUE IZQ' : 'GIRO TANQUE DER';
      statusBadge.style.color = 'var(--accent-yellow)';
    } else {
      statusBadge.innerText = v > 0 ? 'AVANCE RECTO' : 'RETROCESO';
      statusBadge.style.color = 'var(--accent-emerald)';
    }
  }

  // --- UI & Controls ---
  setupUI() {
    const btnCad = document.getElementById('btn-mode-cad');
    const btnSim = document.getElementById('btn-mode-sim');
    const panelCad = document.getElementById('panel-cad');
    const panelInspector = document.getElementById('panel-inspector');
    const panelSim = document.getElementById('panel-sim');
    const fpvPip = document.getElementById('fpv-pip');

    btnCad.addEventListener('click', () => {
      state.mode = 'cad';
      btnCad.classList.add('active');
      btnSim.classList.remove('active');
      panelCad.classList.remove('hidden');
      panelInspector.classList.remove('hidden');
      panelSim.classList.add('hidden');
      fpvPip.classList.add('hidden');
      this.resetRobotPosition();
      this.setCameraPreset('iso');
      synth.click();
    });

    btnSim.addEventListener('click', () => {
      state.mode = 'sim';
      btnSim.classList.add('active');
      btnCad.classList.remove('active');
      panelCad.classList.add('hidden');
      panelInspector.classList.add('hidden');
      panelSim.classList.remove('hidden');
      fpvPip.classList.remove('hidden');
      state.exploded = 0;
      document.getElementById('slider-exploded').value = 0;
      document.getElementById('exploded-val').innerText = '0%';
      this.updateExplodedView(0);

      // Posicionar cámara inmediatamente detrás del carro en la pista
      if (state.simCamMode === 'chase') {
        const chaseOffset = new THREE.Vector3(
          Math.sin(state.yaw) * 0.65,
          0.30,
          Math.cos(state.yaw) * 0.65
        );
        this.camera.position.copy(this.robotRoot.position.clone().add(chaseOffset));
        this.controls.target.copy(this.robotRoot.position.clone().add(new THREE.Vector3(0, 0.06, 0)));
      }

      synth.init();
      synth.click();
    });

    // Camera View Presets
    document.getElementById('cam-iso').addEventListener('click', () => this.setCameraPreset('iso'));
    document.getElementById('cam-top').addEventListener('click', () => this.setCameraPreset('top'));
    document.getElementById('cam-front').addEventListener('click', () => this.setCameraPreset('front'));
    document.getElementById('cam-side').addEventListener('click', () => this.setCameraPreset('side'));

    // Exploded View Slider
    const sliderExploded = document.getElementById('slider-exploded');
    const explodedVal = document.getElementById('exploded-val');
    sliderExploded.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value) / 100.0;
      state.exploded = val;
      explodedVal.innerText = `${e.target.value}%`;
      this.updateExplodedView(val);
    });

    // Shading Mode Buttons
    const shadePbr = document.getElementById('shading-pbr');
    const shadeWire = document.getElementById('shading-wire');
    const shadeXray = document.getElementById('shading-xray');

    const updateShadingUI = (activeBtn) => {
      [shadePbr, shadeWire, shadeXray].forEach(b => b.classList.remove('active'));
      activeBtn.classList.add('active');
    };

    shadePbr.addEventListener('click', () => {
      this.setShadingMode('pbr');
      updateShadingUI(shadePbr);
      synth.click();
    });
    shadeWire.addEventListener('click', () => {
      this.setShadingMode('wire');
      updateShadingUI(shadeWire);
      synth.click();
    });
    shadeXray.addEventListener('click', () => {
      this.setShadingMode('xray');
      updateShadingUI(shadeXray);
      synth.click();
    });

    // Component Tree Clicking
    document.querySelectorAll('.comp-item').forEach(item => {
      item.addEventListener('click', () => {
        const comp = item.dataset.comp;
        this.selectComponent(comp);
        synth.click();
      });
    });

    // Sound Toggle
    const btnSound = document.getElementById('btn-sound-toggle');
    const iconSoundOn = document.getElementById('icon-sound-on');
    const iconSoundOff = document.getElementById('icon-sound-off');
    btnSound.addEventListener('click', () => {
      state.soundEnabled = !state.soundEnabled;
      iconSoundOn.classList.toggle('hidden', !state.soundEnabled);
      iconSoundOff.classList.toggle('hidden', state.soundEnabled);
      synth.click();
    });

    // Reset Button
    document.getElementById('btn-reset-pos').addEventListener('click', () => {
      this.resetRobotPosition();
      this.setCameraPreset('iso');
      synth.click();
    });

    // FPV Close Button
    document.getElementById('btn-close-fpv').addEventListener('click', () => {
      document.getElementById('fpv-pip').classList.add('hidden');
    });

    // Real ESP32-CAM Stream Connect & Controls
    const btnConnectCam = document.getElementById('btn-connect-cam');
    const btnReconnectCam = document.getElementById('btn-reconnect-cam');
    const btnToggleFlash = document.getElementById('btn-toggle-flash');
    const flashStatusText = document.getElementById('flash-status-text');
    const selectCamMode = document.getElementById('select-cam-mode');
    const inputCamIp = document.getElementById('input-cam-ip');
    const fpvRealImg = document.getElementById('fpv-real-img');
    const fpvCanvas = document.getElementById('fpv-canvas');
    const fpvStatus = document.getElementById('fpv-hud-status');
    const fpvTitle = document.getElementById('fpv-title');

    let isRealCamConnected = false;
    let isFlashActive = false;
    let snapshotPollingTimer = null;

    const getCleanCamIp = () => {
      let raw = (inputCamIp ? inputCamIp.value.trim() : '') || '192.168.100.25';
      return raw.replace(/^https?:\/\//i, '').split(':')[0].split('/')[0];
    };

    const stopSnapshotPolling = () => {
      if (snapshotPollingTimer) {
        clearTimeout(snapshotPollingTimer);
        snapshotPollingTimer = null;
      }
    };

    const runSnapshotPolling = () => {
      if (!isRealCamConnected || (selectCamMode && selectCamMode.value !== 'snapshot')) return;
      const ip = getCleanCamIp();
      const tempImg = new Image();
      tempImg.onload = () => {
        if (!isRealCamConnected) return;
        fpvRealImg.src = tempImg.src;
        if (fpvStatus) fpvStatus.innerHTML = 'FPS: ~20 &bull; ANTI-FREEZE';
        snapshotPollingTimer = setTimeout(runSnapshotPolling, 50); // ~20 FPS fluido
      };
      tempImg.onerror = () => {
        // En caso de ruido por motores o micro-corte, reintentar tras 250ms automáticamente
        if (fpvStatus) fpvStatus.innerHTML = '<span style="color:#f59e0b">REINTENTANDO...</span>';
        snapshotPollingTimer = setTimeout(runSnapshotPolling, 250);
      };
      tempImg.src = `http://${ip}/cam.jpg?t=${Date.now()}`;
    };

    const activateCamFeed = () => {
      const ip = getCleanCamIp();
      if (!ip) return;
      stopSnapshotPolling();

      const mode = selectCamMode ? selectCamMode.value : 'stream';
      if (mode === 'snapshot') {
        runSnapshotPolling();
      } else {
        // Modo MJPEG en puerto 81
        fpvRealImg.src = `http://${ip}:81/stream?t=${Date.now()}`;
        if (fpvStatus) fpvStatus.innerHTML = 'FPS: LIVE &bull; MJPEG';
      }

      fpvRealImg.classList.remove('hidden');
      if (fpvCanvas) fpvCanvas.classList.add('hidden');
      if (fpvTitle) fpvTitle.innerText = `ESP32-CAM (${ip})`;

      btnConnectCam.innerText = 'Desconectar';
      btnConnectCam.style.background = '#dc2626';
      btnConnectCam.style.color = '#fff';
      isRealCamConnected = true;
    };

    const disconnectCamFeed = () => {
      stopSnapshotPolling();
      isRealCamConnected = false;
      fpvRealImg.classList.add('hidden');
      fpvRealImg.src = '';
      if (fpvCanvas) fpvCanvas.classList.remove('hidden');
      if (fpvStatus) fpvStatus.innerHTML = 'FPS: 30.0 &bull; 3D SIM';
      if (fpvTitle) fpvTitle.innerText = 'ESP32-CAM STREAM • VIRTUAL';
      btnConnectCam.innerText = 'Conectar';
      btnConnectCam.style.background = '#00f2fe';
      btnConnectCam.style.color = '#000';
    };

    if (btnConnectCam && inputCamIp && fpvRealImg) {
      btnConnectCam.addEventListener('click', () => {
        if (isRealCamConnected) {
          disconnectCamFeed();
        } else {
          activateCamFeed();
        }
        synth.click();
      });

      if (btnReconnectCam) {
        btnReconnectCam.addEventListener('click', () => {
          if (isRealCamConnected) {
            activateCamFeed();
            synth.click();
          }
        });
      }

      if (selectCamMode) {
        selectCamMode.addEventListener('change', () => {
          if (isRealCamConnected) {
            activateCamFeed();
          }
        });
      }

      // Linterna Flash (GPIO 4)
      if (btnToggleFlash) {
        btnToggleFlash.addEventListener('click', () => {
          const ip = getCleanCamIp();
          if (!ip) return;

          isFlashActive = !isFlashActive;
          const val = isFlashActive ? 1 : 0;

          // Petición al endpoint /flash del ESP32-CAM
          fetch(`http://${ip}/flash?val=${val}`, { method: 'GET', mode: 'no-cors' })
            .catch(err => console.warn('Flash request warning:', err));

          if (isFlashActive) {
            btnToggleFlash.style.background = '#eab308';
            btnToggleFlash.style.color = '#000';
            btnToggleFlash.style.borderColor = '#fde047';
            if (flashStatusText) flashStatusText.innerText = 'ON';
          } else {
            btnToggleFlash.style.background = '#1e293b';
            btnToggleFlash.style.color = '#94a3b8';
            btnToggleFlash.style.borderColor = '#475569';
            if (flashStatusText) flashStatusText.innerText = 'OFF';
          }
          synth.click();
        });
      }

      // Auto-recuperación si la imagen MJPEG falla
      fpvRealImg.onerror = () => {
        if (isRealCamConnected && selectCamMode && selectCamMode.value === 'stream') {
          if (fpvStatus) fpvStatus.innerHTML = '<span style="color:#ef4444">PAUSA WI-FI...</span>';
          setTimeout(() => {
            if (isRealCamConnected) activateCamFeed();
          }, 1500);
        }
      };
    }

    // Autopilot Toggle Button
    const btnAutopilot = document.getElementById('btn-autopilot');
    if (btnAutopilot) {
      btnAutopilot.addEventListener('click', () => {
        this.toggleAutopilot();
        synth.click();
      });
    }

    // Laser Rays Toggle Button
    const btnToggleLaser = document.getElementById('btn-toggle-laser');
    if (btnToggleLaser) {
      btnToggleLaser.addEventListener('click', () => {
        this.toggleLaserVisibility();
        synth.click();
      });
    }

    this.setupDpad();
  }

  setupDpad() {
    const bindBtn = (id, fwd, turn) => {
      const btn = document.getElementById(id);
      if (!btn) return;
      const start = (e) => {
        e.preventDefault();
        synth.init();
        state.input.fwd = fwd;
        state.input.turn = turn;
        btn.classList.add('active');
      };
      const end = (e) => {
        e.preventDefault();
        state.input.fwd = 0;
        state.input.turn = 0;
        btn.classList.remove('active');
      };
      btn.addEventListener('pointerdown', start);
      btn.addEventListener('pointerup', end);
      btn.addEventListener('pointerleave', end);
    };

    bindBtn('btn-drive-fwd', 1, 0);
    bindBtn('btn-drive-rev', -1, 0);
    bindBtn('btn-drive-left', 0, 1);   // Giro Tanque Izquierda
    bindBtn('btn-drive-right', 0, -1); // Giro Tanque Derecha

    const stopBtn = document.getElementById('btn-drive-stop');
    if (stopBtn) {
      stopBtn.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        state.input.fwd = 0;
        state.input.turn = 0;
        state.vLeft = 0;
        state.vRight = 0;
        synth.click();
      });
    }

    const chaseBtn = document.getElementById('sim-cam-chase');
    const orbitBtn = document.getElementById('sim-cam-orbit');
    const fpvBtn = document.getElementById('sim-cam-fpv');

    const setSimCam = (mode, btn) => {
      state.simCamMode = mode;
      [chaseBtn, orbitBtn, fpvBtn].forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      synth.click();
    };

    chaseBtn.addEventListener('click', () => setSimCam('chase', chaseBtn));
    orbitBtn.addEventListener('click', () => setSimCam('orbit', orbitBtn));
    fpvBtn.addEventListener('click', () => {
      setSimCam('fpv', fpvBtn);
      document.getElementById('fpv-pip').classList.remove('hidden');
    });
  }

  setupKeyboard() {
    const activeKeys = {};
    window.addEventListener('keydown', (e) => {
      synth.init();
      activeKeys[e.key.toLowerCase()] = true;
      this.handleKeyInputs(activeKeys);

      if (e.key === ' ' || e.code === 'Space') {
        state.input.fwd = 0;
        state.input.turn = 0;
        state.vLeft = 0;
        state.vRight = 0;
      }
      if (e.key.toLowerCase() === 'r') {
        this.resetRobotPosition();
      }
      if (e.key.toLowerCase() === 'p') {
        this.toggleAutopilot();
        synth.click();
      }
      if (e.key.toLowerCase() === 'l') {
        this.toggleLaserVisibility();
        synth.click();
      }
      if (e.key.toLowerCase() === 'c') {
        const modes = ['chase', 'orbit', 'fpv'];
        const nextIdx = (modes.indexOf(state.simCamMode) + 1) % modes.length;
        state.simCamMode = modes[nextIdx];
        document.querySelectorAll('.cam-follow-group .view-btn').forEach((b, i) => {
          b.classList.toggle('active', i === nextIdx);
        });
      }
    });

    window.addEventListener('keyup', (e) => {
      delete activeKeys[e.key.toLowerCase()];
      this.handleKeyInputs(activeKeys);
    });
  }

  handleKeyInputs(keys) {
    let fwd = 0;
    let turn = 0;

    if (keys['w'] || keys['arrowup']) fwd += 1;
    if (keys['s'] || keys['arrowdown']) fwd -= 1;
    if (keys['a'] || keys['arrowleft']) turn += 1;  // Tracción Diferencial Izquierda
    if (keys['d'] || keys['arrowright']) turn -= 1; // Tracción Diferencial Derecha

    // Intervención manual desactiva suavemente el piloto automático para dar control inmediato
    if ((fwd !== 0 || turn !== 0) && state.autopilot) {
      this.setAutopilot(false);
    }

    if (!state.autopilot) {
      state.input.fwd = fwd;
      state.input.turn = turn;
    }
  }

  toggleAutopilot() {
    this.setAutopilot(!state.autopilot);
  }

  setAutopilot(enabled) {
    state.autopilot = enabled;
    if (enabled) {
      this.autopilot.reset();
    } else {
      state.input.fwd = 0;
      state.input.turn = 0;
    }
    const btn = document.getElementById('btn-autopilot');
    const label = document.getElementById('autopilot-label');
    const status = document.getElementById('telem-pilot-status');
    if (btn) btn.classList.toggle('active', enabled);
    if (label) label.innerText = enabled ? 'PILOTO AUTÓNOMO: ACTIVO' : 'PILOTO AUTÓNOMO: APAGADO';
    if (status) {
      status.innerText = enabled ? 'AUTÓNOMO PID' : 'MANUAL';
      status.style.color = enabled ? 'var(--accent-emerald)' : 'var(--text-dim)';
    }
  }

  toggleLaserVisibility() {
    state.laserVisible = !state.laserVisible;
    if (this.laserScanner) this.laserScanner.setVisible(state.laserVisible);
    const btn = document.getElementById('btn-toggle-laser');
    if (btn) {
      btn.classList.toggle('active', state.laserVisible);
      btn.innerText = state.laserVisible ? 'Haces Láser: VISIBLES' : 'Haces Láser: OCULTOS';
    }
  }

  resetRobotPosition() {
    state.x = 0.0;
    state.z = 0.25;
    state.yaw = -Math.PI / 2;
    state.vLeft = 0;
    state.vRight = 0;
    state.input.fwd = 0;
    state.input.turn = 0;
    this.autopilot.reset();
    this.robotRoot.position.set(state.x, 0, state.z);
    this.robotRoot.rotation.y = state.yaw;
  }

  setCameraPreset(preset) {
    const target = this.robotRoot.position.clone().add(new THREE.Vector3(0, 0.04, 0));
    this.controls.target.copy(target);

    document.querySelectorAll('.btn-grid-4 .view-btn').forEach(b => b.classList.remove('active'));
    const btn = document.getElementById(`cam-${preset}`);
    if (btn) btn.classList.add('active');

    if (preset === 'iso') {
      this.camera.position.set(target.x + 0.35, target.y + 0.28, target.z + 0.38);
    } else if (preset === 'top') {
      this.camera.position.set(target.x + 0.001, target.y + 0.55, target.z + 0.001);
    } else if (preset === 'front') {
      this.camera.position.set(target.x + 0.001, target.y + 0.08, target.z - 0.42);
    } else if (preset === 'side') {
      this.camera.position.set(target.x + 0.42, target.y + 0.08, target.z + 0.001);
    }
    this.controls.update();
  }

  setShadingMode(mode) {
    state.shading = mode;
    this.robotRoot.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        if (mode === 'wire') {
          obj.material.wireframe = true;
          obj.material.transparent = false;
          obj.material.opacity = 1.0;
        } else if (mode === 'xray') {
          obj.material.wireframe = false;
          obj.material.transparent = true;
          obj.material.opacity = 0.35;
        } else {
          obj.material.wireframe = false;
          obj.material.transparent = false;
          obj.material.opacity = 1.0;
        }
      }
    });
  }

  selectComponent(compKey) {
    state.selectedComponent = compKey;
    const data = COMPONENT_DB[compKey];
    if (!data) return;

    document.getElementById('insp-category').innerText = data.category;
    document.getElementById('insp-name').innerText = data.name;
    document.getElementById('insp-desc').innerText = data.desc;
    document.getElementById('spec-volt').innerText = data.voltage;
    document.getElementById('spec-curr').innerText = data.current;
    document.getElementById('spec-dim').innerText = data.dimensions;
    document.getElementById('spec-role').innerText = data.role;
    document.getElementById('insp-notes').innerText = data.notes;

    document.querySelectorAll('.comp-item').forEach(item => {
      item.classList.toggle('selected', item.dataset.comp === compKey);
    });

    this.highlight3dPart(compKey);
  }

  highlight3dPart(compKey) {
    this.robotRoot.traverse((obj) => {
      if (obj.isMesh && obj.userData.component === compKey) {
        obj.material.emissive = new THREE.Color(0x00f2fe);
        obj.material.emissiveIntensity = 0.6;
        setTimeout(() => {
          if (obj.material) {
            obj.material.emissive = new THREE.Color(0x000000);
            obj.material.emissiveIntensity = 0.0;
          }
        }, 350);
      }
    });
  }

  onPointerClick(event) {
    if (state.mode !== 'cad') return;
    this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.robotRoot.children, true);

    if (intersects.length > 0) {
      let hit = intersects[0].object;
      while (hit && !hit.userData.component && hit.parent && hit.parent !== this.robotRoot) {
        hit = hit.parent;
      }
      if (hit && hit.userData.component) {
        this.selectComponent(hit.userData.component);
        synth.click();
      }
    }
  }

  onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const delta = Math.min(this.clock.getDelta(), 0.05);

    this.updateSimulation(delta);

    if (state.simCamMode !== 'fpv') {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);

    const fpvPip = document.getElementById('fpv-pip');
    if (!fpvPip.classList.contains('hidden')) {
      this.fpvRenderer.render(this.scene, this.fpvCamera);
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new App3D();
});
