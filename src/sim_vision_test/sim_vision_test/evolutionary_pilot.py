#!/usr/bin/env python3
"""
Piloto Autónomo Evolutivo: Usa el mejor cerebro evolucionado para conducir.
Carga el modelo 'best_driver.pth' entrenado por el algoritmo genético.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import torch
import torch.nn as nn
import numpy as np
import math
import os

# Mismos parámetros que en el entrenador
LIDAR_SAMPLES = 10
MIN_SPEED = 0.3
MAX_SPEED = 1.2
MODEL_DIR = os.path.expanduser('~/evolutionary_models')


class NeuralDriver(nn.Module):
    """Misma arquitectura que el entrenador evolutivo."""
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

    def decide(self, lidar_data):
        with torch.no_grad():
            state = torch.tensor(lidar_data, dtype=torch.float32)
            output = self.layers(state)
        angle = output[0].item() * 2.5
        speed_norm = output[1].item()
        speed = MIN_SPEED + (speed_norm + 1.0) / 2.0 * (MAX_SPEED - MIN_SPEED)
        return angle, speed


class EvolutionaryPilotNode(Node):
    def __init__(self):
        super().__init__('evolutionary_pilot_node')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Cargar el mejor modelo evolucionado
        model_path = os.path.join(MODEL_DIR, 'best_driver.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(
                f'No se encontro el modelo en {model_path}. '
                'Primero ejecuta el entrenamiento evolutivo.')
            raise FileNotFoundError(f'Modelo no encontrado: {model_path}')

        self.driver = NeuralDriver()
        self.driver.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.driver.eval()
        self.get_logger().info(f'Modelo evolucionado cargado desde {model_path}')

        self.scan_indices = None

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_lidar, 10)

        self.get_logger().info('Piloto Autónomo Evolutivo (LiDAR) iniciado.')

    def on_lidar(self, msg):
        if self.scan_indices is None:
            n = len(msg.ranges)
            self.scan_indices = [int(i * (n - 1) / (LIDAR_SAMPLES - 1))
                                 for i in range(LIDAR_SAMPLES)]

        values = []
        for i in self.scan_indices:
            v = msg.ranges[i]
            if math.isinf(v) or math.isnan(v):
                v = 10.0
            values.append(min(v, 10.0))

        lidar_norm = [v / 10.0 for v in values]

        angular_z, linear_x = self.driver.decide(lidar_norm)

        twist = Twist()
        twist.linear.x = float(linear_x)
        twist.angular.z = float(angular_z)
        self.cmd_pub.publish(twist)

        print(f'[EVOL] Vel: {linear_x:+.2f} m/s | Giro: {angular_z:+.2f} rad/s', flush=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EvolutionaryPilotNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
