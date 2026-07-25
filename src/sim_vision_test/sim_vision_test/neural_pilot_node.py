import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import torch
import torch.nn as nn

# Modelo CNN con 1 sola salida (steering)
class RacerCNN(nn.Module):
    def __init__(self):
        super(RacerCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 13, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

class NeuralPilotNode(Node):
    def __init__(self):
        super().__init__('neural_pilot_node')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # Cargar parametros configurables de velocidad
        self.declare_parameter('base_speed', 1.2)
        self.declare_parameter('max_angular_speed', 2.5)

        # Cargar modelo entrenado
        model_path = os.path.expanduser('~/training_data/racer_model.pth')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Dispositivo de ejecucion de IA: {self.device}")

        if not os.path.exists(model_path):
            self.get_logger().error(f"ERROR: No se encontro el archivo del modelo en {model_path}")
            self.get_logger().error("Primero debes grabar datos y entrenar el modelo con train_cnn.py")
            raise FileNotFoundError("Modelo racer_model.pth no encontrado.")

        self.model = RacerCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.get_logger().info("Autopiloto de Red Neuronal (Ackermann) iniciado.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            frame_resized = cv2.resize(frame, (160, 120))

            # Transformar imagen para PyTorch
            img_tensor = frame_resized.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))
            img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(self.device)

            # Inferencia en la GPU
            with torch.no_grad():
                prediction = self.model(img_tensor)
                outputs = prediction.cpu().numpy()[0]

            # El modelo predice unicamente la direccion (angular_z)
            angular_z = float(outputs[0])

            # Cargar parametros
            base_speed = self.get_parameter('base_speed').value
            max_ang = self.get_parameter('max_angular_speed').value

            # Limitar el giro predicho por la red neuronal
            angular_z = float(np.clip(angular_z, -max_ang, max_ang))

            # --- CONTROL VELOCIDAD INTELIGENTE (F1) ---
            # Reducir velocidad lineal proporcionalmente segun la fuerza del giro
            turn_ratio = abs(angular_z) / max_ang
            linear_x = base_speed * max(0.2, 1.0 - 0.7 * turn_ratio)

            # Publicar velocidades
            twist = Twist()
            twist.linear.x = float(linear_x)
            twist.angular.z = float(angular_z)
            self.cmd_pub.publish(twist)

            print(f"IA Piloto -> Lin: {twist.linear.x:.2f} m/s | Ang (Giro): {twist.angular.z:+.2f} rad/s", flush=True)

        except Exception as e:
            self.get_logger().error(f"Error en piloto de IA: {e}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = NeuralPilotNode()
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
