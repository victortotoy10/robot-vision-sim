import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
        
        # Puente de OpenCV a ROS
        self.bridge = CvBridge()
        
        # Parámetros HSV configurables para segmentación de líneas
        # Por defecto, rango para detectar amarillo/naranja en HSV
        self.declare_parameter('hsv_lower', [15, 100, 100])
        self.declare_parameter('hsv_upper', [45, 255, 255])
        
        # Parámetro para mostrar la ventana de depuración (cv2.imshow)
        self.declare_parameter('show_image', False)
        
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
            
            cx = w // 2 # Valor por defecto si no hay píxeles segmentados
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
            
            # Cálculo del error
            error = cx - (w // 2)
            
            # Impresión por consola de FPS y error
            print(f"FPS: {self.fps_rolling:.1f} | Error: {error} px", flush=True)
            
            # Ventana opcional de depuración con cv2.imshow
            show_image = self.get_parameter('show_image').value
            if show_image and 'DISPLAY' in os.environ:
                # Dibujar centroide y error en la imagen para depuración visual
                debug_frame = roi.copy()
                if moments["m00"] > 0:
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
