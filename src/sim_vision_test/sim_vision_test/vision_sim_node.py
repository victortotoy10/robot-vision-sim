import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os

class VisionSimNode(Node):
    def __init__(self):
        super().__init__('vision_sim_node')
        
        # Suscripción al tema de imagen de la cámara
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publicador de comandos de velocidad para control autónomo
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Puente de OpenCV a ROS
        self.bridge = CvBridge()
        
        # Parámetros HSV configurables para segmentación de líneas
        # Por defecto, rango para detectar amarillo/naranja en HSV
        self.declare_parameter('hsv_lower', [15, 100, 100])
        self.declare_parameter('hsv_upper', [45, 255, 255])
        
        # Parámetro para mostrar la ventana de depuración (cv2.imshow)
        self.declare_parameter('show_image', False)
        
        # Parámetros del Controlador PID para seguimiento autónomo de línea
        self.declare_parameter('follow_line', False)
        self.declare_parameter('kp', 0.005)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.001)
        self.declare_parameter('base_speed', 0.3)
        self.declare_parameter('max_angular_speed', 1.0)
        
        # Variables de estado del PID
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_control_time = self.get_clock().now()
        
        # Variables para calcular FPS
        self.last_time = time.time()
        self.fps_rolling = 0.0
        
        self.get_logger().info("Nodo vision_sim_node iniciado. Esperando imágenes...")

    def image_callback(self, msg):
        # Medir tiempo para cálculo de FPS
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        fps = 1.0 / dt if dt > 0 else 0.0
        # Suavizar FPS para evitar lecturas ruidosas
        self.fps_rolling = 0.9 * self.fps_rolling + 0.1 * fps if self.fps_rolling > 0 else fps
        
        try:
            # Conversión de mensaje ROS a OpenCV usando CvBridge
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Redimensionamiento para bajo consumo
            frame_resized = cv2.resize(frame, (320, 240))
            h, w, _ = frame_resized.shape
            
            # Procesamiento de Región de Interés (ROI): bottom 40%
            roi_start_y = int(0.6 * h)
            roi = frame_resized[roi_start_y:, :]
            
            # Conversión a HSV y máscara
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Obtener parámetros de límites HSV
            lower_hsv = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
            upper_hsv = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
            
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            
            # Cálculo de centroide mediante momentos
            moments = cv2.moments(mask)
            
            line_detected = False
            cx = w // 2 # Valor por defecto si no hay píxeles segmentados
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                line_detected = True
            
            # Cálculo del error
            error = cx - (w // 2)
            
            # Calcular tiempo delta para el PID usando tiempo de simulación de ROS
            now = self.get_clock().now()
            dt_pid = (now - self.last_control_time).nanoseconds / 1e9
            self.last_control_time = now
            
            # Algoritmo PID
            if dt_pid > 0.0:
                derivative = (error - self.prev_error) / dt_pid
                self.integral += error * dt_pid
            else:
                derivative = 0.0
            
            # Limitar la parte integral para evitar windup
            self.integral = max(min(self.integral, 50.0), -50.0)
            self.prev_error = error
            
            kp = self.get_parameter('kp').value
            ki = self.get_parameter('ki').value
            kd = self.get_parameter('kd').value
            
            steering_value = kp * error + ki * self.integral + kd * derivative
            
            # Generar comandos de velocidad si está en modo autónomo
            follow_line = self.get_parameter('follow_line').value
            lin_speed = 0.0
            ang_speed = 0.0
            
            if follow_line:
                twist = Twist()
                if not line_detected:
                    # Si perdemos la línea por completo, nos detenemos y rotamos suavemente 
                    # hacia el último lado conocido para intentar recuperarla
                    twist.linear.x = 0.0
                    if self.prev_error > 0:
                        twist.angular.z = -0.3  # rotar a la derecha
                    else:
                        twist.angular.z = 0.3   # rotar a la izquierda
                else:
                    base_speed = self.get_parameter('base_speed').value
                    max_angular_speed = self.get_parameter('max_angular_speed').value
                    
                    # Giro del robot (si el error es positivo -> línea a la derecha -> girar a la derecha)
                    angular_z = -steering_value
                    twist.angular.z = max(min(angular_z, max_angular_speed), -max_angular_speed)
                    
                    # Velocidad lineal: se reduce proporcionalmente a la desviación (error)
                    # En recta va a velocidad máxima, en curvas reduce la velocidad.
                    speed_scale = max(0.0, 1.0 - (abs(error) / (w // 2)))
                    twist.linear.x = base_speed * speed_scale
                
                self.cmd_vel_pub.publish(twist)
                lin_speed = twist.linear.x
                ang_speed = twist.angular.z
            
            # Impresión por consola de FPS, error y comandos de velocidad si está en modo autónomo
            if follow_line:
                print(f"FPS: {self.fps_rolling:.1f} | Error: {error:4d} px | Velocidad -> Lin: {lin_speed:.2f} m/s, Ang: {ang_speed:.2f} rad/s", flush=True)
            else:
                print(f"FPS: {self.fps_rolling:.1f} | Error: {error:4d} px | Autonomo: INACTIVO (teleop/teclado)", flush=True)
            
            # Ventana opcional de depuración con cv2.imshow
            show_image = self.get_parameter('show_image').value
            if show_image and 'DISPLAY' in os.environ:
                # Dibujar centroide y error en la imagen para depuración visual
                debug_frame = roi.copy()
                if line_detected:
                    cv2.circle(debug_frame, (cx, int(0.2 * h)), 5, (0, 255, 0), -1)
                # Dibujar línea central
                cv2.line(debug_frame, (w // 2, 0), (w // 2, int(0.4 * h)), (255, 0, 0), 1)
                
                cv2.imshow("ROI original", debug_frame)
                cv2.imshow("Mascara HSV", mask)
                cv2.waitKey(1)
                
        except Exception as e:
            self.get_logger().error(f"Error al procesar la imagen: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = VisionSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Apagando nodo de visión...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
