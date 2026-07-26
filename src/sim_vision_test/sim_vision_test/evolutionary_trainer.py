#!/usr/bin/env python3
"""
Entrenador Evolutivo (Algoritmo Genético) para conducción autónoma.
Adaptado de ar-tu-do-master/autonomous/evolutionary → ROS 2 Humble + Gazebo Fortress.

El carro aprende SOLO por prueba y error, sin intervención humana.
Usa el sensor LiDAR para navegar y una red neuronal pequeña como cerebro.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import torch
import torch.nn as nn
import numpy as np
import math
import os
import random
import subprocess
import time
import json

# ============================================================
# HIPERPARÁMETROS DEL ALGORITMO GENÉTICO
# ============================================================
LIDAR_SAMPLES = 10          # Cuántos rayos LiDAR usa el cerebro
POPULATION_SIZE = 20        # Individuos por generación (rápido en GPU)
SURVIVOR_COUNT = 6          # Cuántos sobreviven cada generación
INITIAL_POPULATION = 60     # Población inicial (más diversidad)
MAX_EPISODE_STEPS = 3000    # Pasos máximos antes de terminar un episodio
CRASH_DISTANCE = 0.20       # Distancia en metros para considerar "choque"
MUTATION_RATE = 0.15        # Intensidad de las mutaciones genéticas

MIN_SPEED = 0.3
MAX_SPEED = 1.2

# Directorio donde se guardan los mejores cerebros
MODEL_DIR = os.path.expanduser('~/evolutionary_models')

# Nombre del modelo y mundo en Gazebo (del launch file)
GAZEBO_MODEL_NAME = 'my_robot'
GAZEBO_WORLD_NAME = 'racetrack'

# Posición de inicio del carro en la pista
SPAWN_X = 0.0
SPAWN_Y = 0.0
SPAWN_Z = 0.15


# ============================================================
# RED NEURONAL (CEREBRO DEL CARRO)
# ============================================================
class NeuralDriver(nn.Module):
    """Red neuronal pequeña: LiDAR → decisiones de conducción.
    Entrada: N rayos LiDAR normalizados
    Salida: [ángulo de dirección, velocidad] en rango [-1, 1] (Tanh)
    """
    def __init__(self):
        super(NeuralDriver, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(LIDAR_SAMPLES, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Tanh()
        )
        self.fitness = 0.0
        self.total_velocity = 0.0

    def decide(self, lidar_data):
        """Dada una lectura de LiDAR, decide ángulo y velocidad."""
        with torch.no_grad():
            state = torch.tensor(lidar_data, dtype=torch.float32)
            output = self.layers(state)
        angle = output[0].item()       # [-1, 1] → se mapea a angular_z
        speed_norm = output[1].item()  # [-1, 1] → se mapea a linear_x
        speed = MIN_SPEED + (speed_norm + 1.0) / 2.0 * (MAX_SPEED - MIN_SPEED)
        return angle * 2.5, speed      # angular_z máximo ±2.5 rad/s

    def to_vector(self):
        """Serializa todos los pesos de la red a un vector plano."""
        state_dict = self.layers.state_dict()
        tensors = [state_dict[key] for key in sorted(state_dict.keys())]
        tensors = [torch.flatten(t) for t in tensors]
        return torch.cat(tensors)

    def load_vector(self, vector):
        """Carga pesos desde un vector plano."""
        state_dict = self.layers.state_dict()
        pos = 0
        for key in sorted(state_dict.keys()):
            old = state_dict[key]
            size = np.prod(old.shape)
            state_dict[key] = vector[pos:pos + size].reshape(old.shape)
            pos += size
        self.layers.load_state_dict(state_dict)

    def mutate(self):
        """Crea un hijo con mutaciones aleatorias en los pesos."""
        params = self.to_vector()
        noise = torch.randn_like(params) * MUTATION_RATE
        child = NeuralDriver()
        child.load_vector(params + noise)
        return child

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location='cpu'))


# ============================================================
# NODO ROS 2: ENTRENADOR EVOLUTIVO
# ============================================================
class EvolutionaryTrainerNode(Node):
    def __init__(self):
        super().__init__('evolutionary_trainer')

        os.makedirs(MODEL_DIR, exist_ok=True)

        # Publicador de velocidad
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Suscripción al LiDAR
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_lidar, 10)

        # Suscripción a odometría para medir distancia recorrida
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.on_odom, 10)

        # Estado de la simulación
        self.scan_indices = None
        self.latest_lidar = None
        self.car_x = 0.0
        self.car_y = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.distance_traveled = 0.0

        # Estado del algoritmo genético
        self.generation = 0
        self.individual_idx = 0
        self.episode_steps = 0
        self.is_crashed = False

        # Crear población inicial
        self.get_logger().info(f'Creando poblacion inicial de {INITIAL_POPULATION} cerebros aleatorios...')
        self.untested = [NeuralDriver() for _ in range(INITIAL_POPULATION)]
        self.tested = []
        self.current_driver = self.untested[0]
        self.best_fitness_ever = 0.0

        # Timer principal del entrenamiento (cada 50ms = 20 Hz)
        self.timer = self.create_timer(0.05, self.training_step)

        # Esperar a que el LiDAR empiece a publicar
        self.get_logger().info('Esperando datos del LiDAR...')
        self.warmup_done = False
        self.warmup_count = 0

        self.get_logger().info('='*60)
        self.get_logger().info('   ENTRENADOR EVOLUTIVO INICIADO')
        self.get_logger().info(f'   Poblacion inicial: {INITIAL_POPULATION}')
        self.get_logger().info(f'   Por generacion: {POPULATION_SIZE}')
        self.get_logger().info(f'   Sobrevivientes: {SURVIVOR_COUNT}')
        self.get_logger().info('='*60)

    def on_lidar(self, msg):
        """Callback del LiDAR: extrae N muestras equidistantes."""
        if self.scan_indices is None:
            n = len(msg.ranges)
            self.scan_indices = [int(i * (n - 1) / (LIDAR_SAMPLES - 1))
                                 for i in range(LIDAR_SAMPLES)]

        values = []
        for i in self.scan_indices:
            v = msg.ranges[i]
            if math.isinf(v) or math.isnan(v):
                v = 10.0
            values.append(min(v, 10.0))  # Clamp a 10m máximo

        # Normalizar al rango [0, 1]
        self.latest_lidar = [v / 10.0 for v in values]

        # Detectar choque: si muchos rayos frontales están muy cerca
        front_start = len(msg.ranges) // 3
        front_end = 2 * len(msg.ranges) // 3
        front_ranges = msg.ranges[front_start:front_end]
        close_count = sum(1 for r in front_ranges
                         if not math.isinf(r) and not math.isnan(r) and r < CRASH_DISTANCE)
        if close_count > len(front_ranges) * 0.3:
            self.is_crashed = True

        if not self.warmup_done:
            self.warmup_count += 1
            if self.warmup_count >= 20:
                self.warmup_done = True
                self.get_logger().info('LiDAR activo. Iniciando entrenamiento evolutivo...')

    def on_odom(self, msg):
        """Callback de odometría: actualiza la posición del carro."""
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y

    def training_step(self):
        """Bucle principal de entrenamiento (se ejecuta a 20 Hz)."""
        if not self.warmup_done or self.latest_lidar is None:
            return

        # ¿Terminó el episodio?
        if self.is_crashed or self.episode_steps >= MAX_EPISODE_STEPS:
            self.end_episode()
            return

        # El cerebro actual decide qué hacer
        angular_z, linear_x = self.current_driver.decide(self.latest_lidar)

        # Publicar comando de velocidad
        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

        # Actualizar métricas del individuo
        self.current_driver.total_velocity += abs(linear_x)

        # Calcular distancia recorrida
        dx = self.car_x - self.last_x
        dy = self.car_y - self.last_y
        self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self.last_x = self.car_x
        self.last_y = self.car_y

        self.episode_steps += 1

    def end_episode(self):
        """Finaliza el episodio actual y pasa al siguiente individuo."""
        # Calcular fitness: distancia recorrida × velocidad promedio
        avg_velocity = (self.current_driver.total_velocity / max(self.episode_steps, 1))
        self.current_driver.fitness = self.distance_traveled * avg_velocity

        reason = "CHOQUE" if self.is_crashed else "MAX PASOS"
        self.get_logger().info(
            f'  Gen {self.generation:03d} | Individuo {self.individual_idx+1:02d}/{len(self.untested):02d} '
            f'| Fitness: {self.current_driver.fitness:.1f} '
            f'| Dist: {self.distance_traveled:.1f}m '
            f'| Pasos: {self.episode_steps} '
            f'| Fin: {reason}')

        # Mover a la lista de probados
        self.tested.append(self.current_driver)
        self.individual_idx += 1

        # ¿Quedan individuos por probar?
        if self.individual_idx < len(self.untested):
            self.current_driver = self.untested[self.individual_idx]
            self.reset_episode()
        else:
            # Generación completa → selección natural
            self.complete_generation()

    def complete_generation(self):
        """Selección natural: los mejores sobreviven y se reproducen."""
        # Ordenar por fitness (mejor primero)
        self.tested.sort(key=lambda d: d.fitness, reverse=True)

        best = self.tested[0]
        worst = self.tested[-1]

        self.get_logger().info('='*60)
        self.get_logger().info(
            f'  GENERACIÓN {self.generation:03d} COMPLETADA | '
            f'Mejor: {best.fitness:.1f} | '
            f'Peor: {worst.fitness:.1f}')

        # Guardar los mejores modelos al disco
        for i in range(min(SURVIVOR_COUNT, len(self.tested))):
            path = os.path.join(MODEL_DIR, f'driver_{i}.pth')
            self.tested[i].save(path)

        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness
            best_path = os.path.join(MODEL_DIR, 'best_driver.pth')
            best.save(best_path)
            self.get_logger().info(
                f'  ★ NUEVO RECORD! Fitness: {best.fitness:.1f} → Guardado en best_driver.pth')

        # Guardar estadísticas
        stats_path = os.path.join(MODEL_DIR, 'training_stats.json')
        stats = []
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        stats.append({
            'generation': self.generation,
            'best_fitness': best.fitness,
            'worst_fitness': worst.fitness,
            'avg_fitness': sum(d.fitness for d in self.tested) / len(self.tested),
            'best_ever': self.best_fitness_ever,
        })
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        self.get_logger().info('='*60)

        # Selección: los mejores sobreviven
        survivors = self.tested[:SURVIVOR_COUNT]

        # Reproducción: crear hijos mutados
        children = []
        num_children = POPULATION_SIZE - SURVIVOR_COUNT
        for _ in range(num_children):
            parent = random.choice(survivors)
            child = parent.mutate()
            children.append(child)

        # Nueva generación = sobrevivientes + hijos
        # Los sobrevivientes se "reinician" para ser probados de nuevo
        for s in survivors:
            s.fitness = 0.0
            s.total_velocity = 0.0
        self.untested = survivors + children
        self.tested = []
        self.individual_idx = 0
        self.current_driver = self.untested[0]
        self.generation += 1

        self.reset_episode()

    def reset_episode(self):
        """Reinicia el carro a la posición inicial en la pista."""
        # Detener el carro
        stop = Twist()
        self.cmd_pub.publish(stop)
        time.sleep(0.1)

        # Resetear estado
        self.episode_steps = 0
        self.is_crashed = False
        self.distance_traveled = 0.0
        self.current_driver.total_velocity = 0.0

        # Teletransportar el carro a la posición inicial usando Gazebo service
        self.reset_car_position()

        # Dar tiempo a Gazebo para procesar el reset
        time.sleep(0.3)
        self.last_x = self.car_x
        self.last_y = self.car_y

    def reset_car_position(self):
        """Teletransporta el carro al inicio de la pista via servicio de Gazebo."""
        # Añadir variación aleatoria para diversidad
        offset_x = random.uniform(-0.3, 0.3)
        offset_y = random.uniform(-0.1, 0.1)
        # Ángulo aleatorio pequeño para que no siempre arranque igual
        yaw = random.uniform(-0.15, 0.15)
        sz = math.sin(yaw / 2.0)
        cz = math.cos(yaw / 2.0)

        req = (
            f'name: "{GAZEBO_MODEL_NAME}" '
            f'position: {{x: {SPAWN_X + offset_x}, y: {SPAWN_Y + offset_y}, z: {SPAWN_Z}}} '
            f'orientation: {{x: 0, y: 0, z: {sz:.4f}, w: {cz:.4f}}}'
        )

        try:
            subprocess.run(
                ['ign', 'service',
                 '-s', f'/world/{GAZEBO_WORLD_NAME}/set_pose',
                 '--reqtype', 'ignition.msgs.Pose',
                 '--reptype', 'ignition.msgs.Boolean',
                 '--timeout', '2000',
                 '--req', req],
                capture_output=True, timeout=5
            )
        except Exception as e:
            self.get_logger().warn(f'Error al resetear posicion: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EvolutionaryTrainerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Entrenamiento detenido por el usuario.')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
