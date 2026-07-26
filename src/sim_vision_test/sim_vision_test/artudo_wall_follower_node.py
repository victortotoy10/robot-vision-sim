#!/usr/bin/env python3
"""
Nodo Autónomo Wall-Following de ar-tu-do-master (ROS 2 Humble / Gazebo Sim).
Adaptación 1:1 de la navegación por seguimiento de paredes con controlador PID
de F1TENTH / ar-tu-do-master.
"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class PIDController:
    def __init__(self, kp=1.8, ki=0.001, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt=0.05):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class ARTUDOWallFollower(Node):
    def __init__(self):
        super().__init__('artudo_wall_follower_node')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.on_scan, 10)
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.pid = PIDController(kp=1.8, ki=0.001, kd=0.5)
        self.target_distance = 1.0          # Distancia objetivo al centro del carril / pared
        self.prediction_distance = 0.8      # Proyección hacia adelante en metros
        self.max_speed = 0.45               # Velocidad máxima en rectas
        self.min_speed = 0.18               # Velocidad mínima en curvas cerradas
        self.last_time = self.get_clock().now()

        self.get_logger().info('=' * 60)
        self.get_logger().info('   PILOTO AUTÓNOMO WALL-FOLLOWING (1:1 de ar-tu-do-master)')
        self.get_logger().info('=' * 60)

    def get_range_at_angle(self, scan_msg, angle_deg):
        angle_rad = math.radians(angle_deg)
        if angle_rad < scan_msg.angle_min or angle_rad > scan_msg.angle_max:
            return 5.0

        index = int((angle_rad - scan_msg.angle_min) / scan_msg.angle_increment)
        if 0 <= index < len(scan_msg.ranges):
            r = scan_msg.ranges[index]
            if not math.isinf(r) and not math.isnan(r) and r > 0.12:
                return min(r, 10.0)
        return 5.0

    def on_scan(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0.0:
            dt = 0.05
        self.last_time = now

        # Medición a dos ángulos (-45° y -90° a la derecha) para predecir la trayectoria
        a = self.get_range_at_angle(msg, -45.0)
        b = self.get_range_at_angle(msg, -90.0)
        theta = math.radians(45.0)

        # Ángulo alpha de inclinación del vehículo con respecto a la pared
        alpha = math.atan2(a * math.cos(theta) - b, a * math.sin(theta))
        current_dist = b * math.cos(alpha)
        predicted_dist = current_dist + self.prediction_distance * math.sin(alpha)

        # Error entre la posición predicha y la distancia objetivo
        error = self.target_distance - predicted_dist

        # Corrección PID de giro
        steering = self.pid.update(error, dt=dt)
        steering = max(min(steering, 0.70), -0.70)

        # Ajuste de velocidad: frena proporcionalmente en curvas cerradas
        speed = self.max_speed * (1.0 - 0.45 * (abs(steering) / 0.70))
        speed = max(min(speed, self.max_speed), self.min_speed)

        # Publicar orden de control a /cmd_vel
        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = float(steering)
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ARTUDOWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
