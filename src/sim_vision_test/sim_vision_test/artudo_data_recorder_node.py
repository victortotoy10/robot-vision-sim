#!/usr/bin/env python3
"""
Grabador Automático de Telemetría LiDAR (1:1 con artudo_wall_follower).
Graba muestras de Estado (LiDAR + Cinemática) y Acción (Giro + Velocidad)
mientras el vehículo conduce de forma autónoma.
"""

import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ARTUDODataRecorder(Node):
    def __init__(self):
        super().__init__('artudo_data_recorder')

        self.save_dir = os.path.expanduser('~/dataset_artudo')
        os.makedirs(self.save_dir, exist_ok=True)
        self.output_file = os.path.join(self.save_dir, 'artudo_expert_dataset.npz')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan, 10)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd, 10)

        self.latest_scan_sectors = None
        self.latest_twist = Twist()

        self.observations = []
        self.actions = []

        self.timer = self.create_timer(0.05, self.record_step) # 20 Hz (50ms)

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'   GRABADOR DE TELEMETRÍA AUTOMÁTICO INICIADO')
        self.get_logger().info(f'   Guardando en: {self.output_file}')
        self.get_logger().info('=' * 60)

    def on_scan(self, msg):
        n = len(msg.ranges)
        num_sectors = 8
        sector_size = n // num_sectors
        obs = []
        for i in range(num_sectors):
            sector_ranges = msg.ranges[i*sector_size : (i+1)*sector_size]
            valid_ranges = [r for r in sector_ranges if not math.isinf(r) and not math.isnan(r) and r > 0.12]
            min_r = min(valid_ranges) if valid_ranges else 10.0
            obs.append(min(min_r, 10.0) / 10.0)
        self.latest_scan_sectors = np.array(obs, dtype=np.float32)

    def on_cmd(self, msg):
        self.latest_twist = msg

    def record_step(self):
        if self.latest_scan_sectors is None:
            return

        steer = self.latest_twist.angular.z
        speed = self.latest_twist.linear.x

        # Solo grabar si el auto se está moviendo activamente
        if abs(speed) > 0.05 or abs(steer) > 0.05:
            # Estado X: 8 sectores LiDAR + speed + steer
            state = np.concatenate([
                self.latest_scan_sectors,
                np.array([speed / 0.50, steer / 0.70], dtype=np.float32)
            ])
            action = np.array([steer, speed], dtype=np.float32)

            self.observations.append(state)
            self.actions.append(action)

            count = len(self.observations)
            if count % 200 == 0:
                laps_estimate = count / 750.0
                self.get_logger().info(f'[GRABANDO] Muestras: {count:05d} (~{laps_estimate:.1f} vueltas)')

    def save_dataset(self):
        if len(self.observations) > 0:
            obs_array = np.array(self.observations, dtype=np.float32)
            act_array = np.array(self.actions, dtype=np.float32)
            np.savez_compressed(self.output_file, obs=obs_array, actions=act_array)
            self.get_logger().info('=' * 60)
            self.get_logger().info(f'[ÉXITO] Dataset guardado correctamente:')
            self.get_logger().info(f'  Total muestras: {len(obs_array)}')
            self.get_logger().info(f'  Formato Obs   : {obs_array.shape}')
            self.get_logger().info(f'  Formato Act   : {act_array.shape}')
            self.get_logger().info(f'  Ruta          : {self.output_file}')
            self.get_logger().info('=' * 60)

def main(args=None):
    rclpy.init(args=args)
    node = ARTUDODataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Deteniendo grabación...")
    finally:
        node.save_dataset()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
