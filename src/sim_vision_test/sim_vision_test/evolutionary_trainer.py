#!/usr/bin/env python3
"""
Entrenador Evolutivo Avanzado (Método de Hijos / ar-tu-do-master).
Fixes:
1. Eliminado set_pose (que daba error id:[0]). Usa WorldControl reset oficial.
2. Eliminada auto-colisión LiDAR de 14cm (misma corrección que en racetrack_env).
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

# Waypoints de la pista Racetrack
PATH_POINTS = np.array([
    [2.64, -0.36], [6.08, -0.33], [7.64, -0.15], [9.10, 0.41], [10.31, 1.39],
    [11.13, 2.72], [11.56, 4.23], [11.67, 5.79], [11.65, 12.07], [11.14, 13.52],
    [10.05, 14.65], [8.73, 15.48], [7.18, 15.66], [-0.67, 15.70], [-2.06, 15.14],
    [-2.39, 13.68], [-1.97, 12.11], [-1.48, 11.79], [-0.73, 11.67], [2.41, 11.68],
    [3.88, 11.34], [5.05, 10.33], [5.60, 8.88], [5.44, 7.37], [4.78, 5.96],
    [3.76, 4.83], [2.39, 4.06], [0.86, 3.76], [-0.67, 3.98], [-2.08, 4.66],
    [-3.31, 5.63], [-9.86, 12.40], [-12.18, 14.53], [-13.41, 15.26], [-14.93, 15.60],
    [-16.46, 15.35], [-17.66, 14.38], [-18.21, 12.93], [-18.30, 11.36], [-18.11, 9.81],
    [-17.35, 6.88], [-17.28, 5.31], [-16.91, 3.80], [-16.07, 2.48], [-14.89, 1.53],
    [-13.44, 0.92], [-11.90, 0.65], [-9.02, 0.66], [-6.00, -0.17], [2.64, -0.36]
])

class Track:
    def __init__(self, points):
        self.points = points[:-1, :]
        self.size = len(self.points)
        next_points = points[1:, :]
        relative = next_points - self.points
        self.segment_length = np.linalg.norm(relative, axis=1)
        self.length = np.sum(self.segment_length)
        self.forward = relative / self.segment_length[:, np.newaxis]
        self.right = np.array([self.forward[:, 1], -self.forward[:, 0]]).transpose()

    def localize(self, px, py):
        local = np.array([px, py]) - self.points
        x = local[:, 0] * self.right[:, 0] + local[:, 1] * self.right[:, 1]
        y = local[:, 0] * self.forward[:, 0] + local[:, 1] * self.forward[:, 1]
        
        distances = np.abs(x)
        distances[(y < 0) | (y > self.segment_length)] = float("Inf")
        
        segment = np.argmin(distances)
        if distances[segment] == float("Inf"):
            return 999.0, 0.0
            
        return x[segment], math.atan2(self.forward[segment, 1], self.forward[segment, 0])


def euler_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


STATE_SIZE = 8
POPULATION_SIZE = 25
SURVIVOR_COUNT = 5
MAX_EPISODE_STEPS = 2000
MUTATION_RATE = 0.15

MIN_SPEED = 0.15
MAX_SPEED = 0.45
MAX_STEER = 0.50

MODEL_DIR = os.path.expanduser('~/evolutionary_models')


class NeuralDriver(nn.Module):
    def __init__(self):
        super(NeuralDriver, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Tanh()
        )
        self.fitness = 0.0
        self.total_velocity = 0.0

    def decide(self, lidar_data):
        with torch.no_grad():
            state = torch.tensor(lidar_data, dtype=torch.float32)
            output = self.layers(state)
        angle = output[0].item() * MAX_STEER
        speed_norm = output[1].item()
        speed = (MIN_SPEED + MAX_SPEED) / 2.0 + speed_norm * (MAX_SPEED - MIN_SPEED) / 2.0
        return angle, speed

    def to_vector(self):
        state_dict = self.layers.state_dict()
        tensors = [state_dict[key] for key in sorted(state_dict.keys())]
        tensors = [torch.flatten(t) for t in tensors]
        return torch.cat(tensors)

    def load_vector(self, vector):
        state_dict = self.layers.state_dict()
        pos = 0
        for key in sorted(state_dict.keys()):
            old = state_dict[key]
            size = np.prod(old.shape)
            state_dict[key] = vector[pos:pos + size].reshape(old.shape)
            pos += size
        self.layers.load_state_dict(state_dict)

    def mutate(self):
        params = self.to_vector()
        noise = torch.randn_like(params) * MUTATION_RATE
        child = NeuralDriver()
        child.load_vector(params + noise)
        return child

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)


class EvolutionaryTrainerNode(Node):
    def __init__(self):
        super().__init__('evolutionary_trainer')

        os.makedirs(MODEL_DIR, exist_ok=True)
        self.track = Track(PATH_POINTS)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_lidar, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.on_odom, 10)

        self.scan_indices = None
        self.latest_lidar = None
        self.car_x = 0.0
        self.car_y = -0.25
        self.car_yaw = 0.0
        self.last_x = 0.0
        self.last_y = -0.25
        self.distance_traveled = 0.0

        self.just_reset = False
        self.reset_time = time.time()

        self.generation = 0
        self.individual_idx = 0
        self.episode_steps = 0
        self.is_crashed = False
        self.crash_reason = ""

        self.untested = [NeuralDriver() for _ in range(POPULATION_SIZE)]
        self.tested = []
        self.current_driver = self.untested[0]
        self.best_fitness_ever = 0.0

        self.timer = self.create_timer(0.05, self.training_step)
        self.warmup_done = False
        self.warmup_count = 0

        self.get_logger().info('='*60)
        self.get_logger().info('   ENTRENADOR EVOLUTIVO GENÉTICO (25 INDIVIDUOS / MÉTODOS DE HIJOS)')
        self.get_logger().info('='*60)

    def on_lidar(self, msg):
        if self.scan_indices is None:
            n = len(msg.ranges)
            self.scan_indices = [int(i * (n - 1) / (STATE_SIZE - 1))
                                 for i in range(STATE_SIZE)]

        values = []
        for i in self.scan_indices:
            v = msg.ranges[i]
            if math.isinf(v) or math.isnan(v) or v <= 0.18:
                v = 10.0
            values.append(min(v, 10.0))

        self.latest_lidar = [v / 10.0 for v in values]

        if not self.warmup_done:
            self.warmup_count += 1
            if self.warmup_count >= 20:
                self.warmup_done = True

    def on_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.just_reset:
            if time.time() - self.reset_time < 0.4:
                return
            else:
                self.just_reset = False

        self.car_x = x
        self.car_y = y
        self.car_yaw = euler_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

    def training_step(self):
        if not self.warmup_done or self.latest_lidar is None:
            return

        if time.time() - self.reset_time < 0.4:
            stop = Twist()
            self.cmd_pub.publish(stop)
            return

        # 2. Control de Posicion
        dist_to_center, seg_angle = self.track.localize(self.car_x, self.car_y)
        
        if abs(dist_to_center) > 0.85:
            self.is_crashed = True
            self.crash_reason = f"SALIDA DE PISTA ({abs(dist_to_center):.2f}m)"
            self.end_episode()
            return

        # 3. Control de Orientacion
        raw_diff = seg_angle - self.car_yaw
        angle_diff = math.atan2(math.sin(raw_diff), math.cos(raw_diff))
        if abs(angle_diff) > 1.2:
            self.is_crashed = True
            self.crash_reason = f"SENTIDO INCORRECTO ({abs(angle_diff)*180/math.pi:.0f}°)"
            self.end_episode()
            return

        if self.is_crashed or self.episode_steps >= MAX_EPISODE_STEPS:
            self.end_episode()
            return

        angular_z, linear_x = self.current_driver.decide(self.latest_lidar)

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

        self.current_driver.total_velocity += abs(linear_x)

        dx = self.car_x - self.last_x
        dy = self.car_y - self.last_y
        self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self.last_x = self.car_x
        self.last_y = self.car_y

        self.episode_steps += 1

    def end_episode(self):
        avg_velocity = (self.current_driver.total_velocity / max(self.episode_steps, 1))
        
        if self.episode_steps < 8:
            self.current_driver.fitness = 0.0
        else:
            self.current_driver.fitness = self.distance_traveled * avg_velocity

        reason = self.crash_reason if self.is_crashed else "MAX PASOS"
        self.get_logger().info(
            f'  Gen {self.generation:03d} | Ind {self.individual_idx+1:02d}/{len(self.untested):02d} '
            f'| Fit: {self.current_driver.fitness:.1f} '
            f'| Dist: {self.distance_traveled:.1f}m '
            f'| Pasos: {self.episode_steps} '
            f'| Fin: {reason}')

        self.tested.append(self.current_driver)
        self.individual_idx += 1

        if self.individual_idx < len(self.untested):
            self.current_driver = self.untested[self.individual_idx]
            self.reset_episode()
        else:
            self.complete_generation()

    def complete_generation(self):
        self.tested.sort(key=lambda d: d.fitness, reverse=True)
        best = self.tested[0]
        worst = self.tested[-1]

        self.get_logger().info('='*60)
        self.get_logger().info(
            f'  GENERACIÓN {self.generation:03d} COMPLETADA | '
            f'Mejor: {best.fitness:.1f} | '
            f'Peor: {worst.fitness:.1f}')

        for i in range(min(SURVIVOR_COUNT, len(self.tested))):
            path = os.path.join(MODEL_DIR, f'driver_{i}.pth')
            self.tested[i].save(path)

        if best.fitness > self.best_fitness_ever and best.fitness > 5.0:
            self.best_fitness_ever = best.fitness
            best_path = os.path.join(MODEL_DIR, 'best_driver.pth')
            best.save(best_path)
            self.get_logger().info(
                f'  ★ NUEVO RECORD DE VERDAD! Fitness: {best.fitness:.1f} → Guardado en best_driver.pth')

        self.get_logger().info('='*60)

        survivors = self.tested[:SURVIVOR_COUNT]

        children = []
        num_children = POPULATION_SIZE - SURVIVOR_COUNT
        for _ in range(num_children):
            parent = random.choice(survivors)
            child = parent.mutate()
            children.append(child)

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
        stop = Twist()
        self.cmd_pub.publish(stop)

        self.episode_steps = 0
        self.is_crashed = False
        self.crash_reason = ""
        self.distance_traveled = 0.0
        self.current_driver.total_velocity = 0.0

        self.reset_car_position()
        self.last_x = self.car_x
        self.last_y = self.car_y

    def reset_car_position(self):
        self.car_x = 0.0
        self.car_y = -0.25
        self.car_yaw = 0.0
        self.just_reset = True
        self.reset_time = time.time()

        req_reset = 'pause: false reset: { model_only: true }'

        try:
            subprocess.run(['ign', 'service', '-s', '/world/racetrack/control', '--reqtype', 'ignition.msgs.WorldControl', '--reptype', 'ignition.msgs.Boolean', '--timeout', '500', '--req', req_reset], capture_output=True, timeout=1.0)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EvolutionaryTrainerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
