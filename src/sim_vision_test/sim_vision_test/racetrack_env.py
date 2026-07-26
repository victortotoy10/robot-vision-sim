#!/usr/bin/env python3
"""
Entorno Gymnasium Avanzado para ROS 2 + Gazebo Sim (Racetrack).
Corrección Integral de MDP Markoviano, Espacio de Estados 12-dim,
LiDAR de 32 Rayos (sin puntos ciegos), Rango de Giro Ampliado (+-0.7 rad)
y Recompensa Basada en Progreso Longitudinal Real (F1TENTH / AWS DeepRacer Pro).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time
import subprocess
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

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
        self.cumulative_length = np.zeros(self.size + 1)
        self.cumulative_length[1:] = np.cumsum(self.segment_length)
        
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
            return 999.0, 0.0, 0.0
            
        progress = self.cumulative_length[segment] + y[segment]
        return x[segment], math.atan2(self.forward[segment, 1], self.forward[segment, 0]), progress

def euler_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class RacetrackEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, random_spawn=False, max_steps=1500):
        super().__init__()
        self.random_spawn = random_spawn
        self.max_steps = max_steps

        # 1. Acciones ampliadas: [angulo (-0.7 a 0.7 rad = +-40.1°), velocidad (0.0 a 0.50 m/s)]
        self.action_space = spaces.Box(
            low=np.array([-0.70, 0.00], dtype=np.float32),
            high=np.array([0.70, 0.50], dtype=np.float32),
            dtype=np.float32
        )

        # 2. Observaciones completas Markovianas (12 dimensiones):
        # 8 rayos LiDAR normalizados + 4 estados cinematicos/temporales
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )

        # ROS 2 Setup
        if not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node('gym_racetrack_env')
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self._on_odom, 10)

        # Executor Thread
        self.executor_thread = threading.Thread(target=self._spin_node, daemon=True)
        self.executor_thread.start()

        self.track = Track(PATH_POINTS)
        self.latest_scan = None
        self.car_x = 0.0
        self.car_y = -0.25
        self.car_yaw = 0.0
        self.car_vx = 0.0
        self.car_wz = 0.0
        
        self.last_steer = 0.0
        self.last_speed = 0.0
        self.last_progress = 0.0
        self.current_step = 0

        self.just_reset = False
        self.reset_time = time.time()

        time.sleep(1.0)

    def _spin_node(self):
        rclpy.spin(self.node)

    def _on_scan(self, msg):
        n = len(msg.ranges)
        # Extraer 8 sectores angulares con min-pooling (sin puntos ciegos)
        obs = []
        num_sectors = 8
        sector_size = n // num_sectors
        for i in range(num_sectors):
            sector_ranges = msg.ranges[i*sector_size : (i+1)*sector_size]
            valid_ranges = [r for r in sector_ranges if not math.isinf(r) and not math.isnan(r) and r > 0.18]
            min_r = min(valid_ranges) if valid_ranges else 10.0
            obs.append(min(min_r, 10.0) / 10.0)
            
        self.latest_scan = np.array(obs, dtype=np.float32)

    def _on_odom(self, msg):
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
        self.car_vx = msg.twist.twist.linear.x
        self.car_wz = msg.twist.twist.angular.z

    def _get_obs(self):
        lidar = self.latest_scan if self.latest_scan is not None else np.ones(8, dtype=np.float32)
        kinematics = np.array([
            self.car_vx / 0.50,            # Velocidad lineal normalizada
            self.car_wz / 2.0,             # Velocidad angular normalizada
            self.last_steer / 0.70,        # Accion de giro anterior
            self.last_speed / 0.50         # Accion de velocidad anterior
        ], dtype=np.float32)
        return np.concatenate([lidar, kinematics])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.just_reset = True
        self.reset_time = time.time()

        # Detener vehiculo
        stop = Twist()
        self.cmd_pub.publish(stop)

        self.car_x = 0.0
        self.car_y = -0.25
        self.car_yaw = 0.0
        self.car_vx = 0.0
        self.car_wz = 0.0
        self.last_steer = 0.0
        self.last_speed = 0.0
        
        _, _, self.last_progress = self.track.localize(0.0, -0.25)

        # Reset Oficial Gazebo Sim
        req_reset = 'pause: false reset: { model_only: true }'
        try:
            subprocess.run(['ign', 'service', '-s', '/world/racetrack/control', '--reqtype', 'ignition.msgs.WorldControl', '--reptype', 'ignition.msgs.Boolean', '--timeout', '500', '--req', req_reset], capture_output=True, timeout=1.0)
        except Exception:
            pass

        time.sleep(0.15)
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        steer, speed = float(action[0]), float(action[1])

        # Publicar accion
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = steer
        self.cmd_pub.publish(twist)

        self.last_steer = steer
        self.last_speed = speed

        # Ciclo de control 50 ms
        time.sleep(0.05)

        # Ventana de amortiguacion post-reset
        if time.time() - self.reset_time < 0.4:
            return self._get_obs(), 0.0, False, False, {}

        # Evaluar estado en el circuito
        dist_to_center, seg_angle, current_progress = self.track.localize(self.car_x, self.car_y)
        raw_diff = seg_angle - self.car_yaw
        angle_diff = math.atan2(math.sin(raw_diff), math.cos(raw_diff))

        # Calcular avance longitudinal real en metros
        delta_progress = current_progress - self.last_progress
        if delta_progress < -self.track.length / 2.0:
            delta_progress += self.track.length
        elif delta_progress > self.track.length / 2.0:
            delta_progress -= self.track.length
            
        self.last_progress = current_progress

        terminated = False
        truncated = self.current_step >= self.max_steps

        # 3. RECOMPENSA DE PROGRESO LONGITUDINAL (F1TENTH / AWS DeepRacer Pro)
        # Recompensa por metros avanzados en el eje central
        progress_reward = 10.0 * delta_progress
        center_penalty = 0.3 * abs(dist_to_center)
        step_penalty = -0.005  # Evita que se quede quieto a 0 m/s

        reward = progress_reward - center_penalty + step_penalty

        # Bonus por completar una vuelta completa (no termina el episodio para permitir vueltas continuas)
        if current_progress >= self.track.length - 0.5:
            reward += 50.0

        # Choque o sentido contrario
        if abs(dist_to_center) > 0.85:
            terminated = True
            reward = -10.0
        elif abs(angle_diff) > 1.2:
            terminated = True
            reward = -5.0

        if self.current_step % 50 == 0:
            obs_check = self._get_obs()
            print(f"[OBS DEBUG] LiDAR min/max: {obs_check[:8].min():.2f}/{obs_check[:8].max():.2f} | "
                  f"v_norm: {obs_check[8]:.2f} | w_norm: {obs_check[9]:.2f} | "
                  f"steer_prev: {obs_check[10]:.2f} | speed_prev: {obs_check[11]:.2f}")

        info = {
            "dist_to_center": dist_to_center,
            "delta_progress": delta_progress,
            "speed": speed,
            "steer": steer
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.node.destroy_node()
