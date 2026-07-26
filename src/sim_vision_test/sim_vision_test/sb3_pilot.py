#!/usr/bin/env python3
"""
Nodo Piloto Autónomo que ejecuta la política entrenada por Stable-Baselines3 (PPO).
"""

import os
import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SB3PilotNode(Node):
    def __init__(self, model_path):
        super().__init__('sb3_pilot')
        
        try:
            from stable_baselines3 import PPO
        except ImportError:
            self.get_logger().error("Stable-Baselines3 no instalado. Ejecuta: pip install stable-baselines3")
            sys.exit(1)

        self.get_logger().info(f"Cargando modelo PPO desde: {model_path}")
        self.model = PPO.load(model_path)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.on_lidar, 10)

        self.latest_scan = None
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("=" * 60)
        self.get_logger().info("   PILOTO AUTÓNOMO STABLE-BASELINES3 (PPO) ACTIVO")
        self.get_logger().info("=" * 60)

    def on_lidar(self, msg):
        n = len(msg.ranges)
        indices = [int(i * (n - 1) / 7) for i in range(8)]
        obs = []
        for i in indices:
            r = msg.ranges[i]
            if math.isinf(r) or math.isnan(r) or r <= 0.10:
                r = 10.0
            obs.append(min(r, 10.0) / 10.0)
        self.latest_scan = np.array(obs, dtype=np.float32)

    def control_loop(self):
        if self.latest_scan is None:
            return

        action, _ = self.model.predict(self.latest_scan, deterministic=True)
        steer, speed = float(action[0]), float(action[1])

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = steer
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)

    model_dir = os.path.expanduser("~/sb3_models")
    model_path = os.path.join(model_dir, "ppo_racetrack_final.zip")

    if not os.path.exists(model_path):
        # Buscar el ultimo modelo guardado
        files = [f for f in os.listdir(model_dir) if f.endswith('.zip')] if os.path.exists(model_dir) else []
        if files:
            files.sort()
            model_path = os.path.join(model_dir, files[-1])
        else:
            print(f"[ERROR] No se encontró ningún modelo en {model_dir}")
            sys.exit(1)

    try:
        node = SB3PilotNode(model_path)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
