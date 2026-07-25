import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import csv
import os
import time

class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('data_recorder_node')

        # Directorio de guardado de datos
        self.data_dir = os.path.expanduser('~/training_data')
        self.images_dir = os.path.join(self.data_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

        self.csv_path = os.path.join(self.data_dir, 'data.csv')
        self.csv_fileExists = os.path.exists(self.csv_path)

        # Crear archivo CSV e incluir cabecera si es nuevo
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if not self.csv_fileExists:
            self.csv_writer.writerow(['image_path', 'linear_x', 'angular_z'])
            self.csv_file.flush()

        self.bridge = CvBridge()
        self.latest_twist = Twist()

        # Suscripción a la cámara
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Suscripción a cmd_vel (comandos de teclado/teleop)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10)

        self.frame_count = 0
        self.get_logger().info(f"Grabador de datos iniciado. Guardando en {self.data_dir}...")
        self.get_logger().info("Conduce el carro usando teleop para iniciar la grabacion.")

    def cmd_callback(self, msg):
        self.latest_twist = msg

    def image_callback(self, msg):
        # Solo grabamos datos si el carro se esta moviendo (conduccion activa)
        linear = self.latest_twist.linear.x
        angular = self.latest_twist.angular.z

        if abs(linear) > 0.01 or abs(angular) > 0.01:
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                
                # Redimensionar la imagen para entrenamiento rapido y bajo consumo (como el proyecto ESP32)
                # Reducimos a 160x120 para optimizar memoria en la GPU
                frame_resized = cv2.resize(frame, (160, 120))
                
                timestamp = int(time.time() * 1000)
                img_filename = f"frame_{timestamp}_{self.frame_count:05d}.png"
                img_path = os.path.join(self.images_dir, img_filename)

                # Guardar imagen en disco
                cv2.imwrite(img_path, frame_resized)

                # Escribir en CSV (ruta relativa para portabilidad)
                rel_img_path = os.path.join('images', img_filename)
                self.csv_writer.writerow([rel_img_path, linear, angular])
                self.csv_file.flush()

                self.frame_count += 1
                if self.frame_count % 100 == 0:
                    self.get_logger().info(f"Frames grabados: {self.frame_count}")

            except Exception as e:
                self.get_logger().error(f"Error al grabar frame: {e}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Apagando grabador de datos...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
