#!/usr/bin/env python3
"""
Piloto Neuronal Clonado (Inferencia PyTorch en Tiempo Real).
Carga la Red Neuronal entrenada con el dataset de artudo y conduce
el vehículo de forma autónoma basándose únicamente en la red neuronal.
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ArtudoNeuralDriver(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super(ArtudoNeuralDriver, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ARTUDONeuralPilot(Node):
    def __init__(self):
        super().__init__('artudo_neural_pilot')

        model_path = os.path.expanduser('~/dataset_artudo/artudo_expert_model.pth')
        if not os.path.exists(model_path):
            self.get_logger().error(f"No se encontró el modelo entrenado en: {model_path}")
            self.get_logger().error("Ejecuta primero: ros2 run sim_vision_test train_artudo_cloning")
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ArtudoNeuralDriver(input_dim=10, output_dim=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan, 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.last_steer = 0.0
        self.last_speed = 0.0

        self.get_logger().info('=' * 60)
        self.get_logger().info(f'   PILOTO NEURONAL CLONADO INICIADO ({self.device})')
        self.get_logger().info(f'   Modelo cargado: {model_path}')
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

        lidar_sectors = np.array(obs, dtype=np.float32)
        state_vec = np.concatenate([
            lidar_sectors,
            np.array([self.last_speed / 0.50, self.last_steer / 0.70], dtype=np.float32)
        ])

        with torch.no_grad():
            tensor_in = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self.model(tensor_in).cpu().numpy()[0]

        steer = float(output[0])
        speed = float(output[1])

        steer = max(min(steer, 0.70), -0.70)
        speed = max(min(speed, 0.50), 0.0)

        self.last_steer = steer
        self.last_speed = speed

        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = steer
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = ARTUDONeuralPilot()
        rclpy.spin(node)
    except FileNotFoundError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
