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

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # HSV (opcional)
        self.declare_parameter('hsv_lower', [0, 0, 180])
        self.declare_parameter('hsv_upper', [180, 60, 255])
        self.declare_parameter('use_hsv', False)
        # Umbral de brillo para detectar las lineas blancas
        self.declare_parameter('brightness_threshold', 160)

        self.declare_parameter('show_image', False)
        self.declare_parameter('follow_line', False)

        # PID ganancias por defecto
        self.declare_parameter('kp', 0.008)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.002)
        self.declare_parameter('base_speed', 0.4)
        self.declare_parameter('max_angular_speed', 1.5)

        # Estado PID
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_control_time = self.get_clock().now()

        # FPS
        self.last_time = time.time()
        self.fps_rolling = 0.0

        # Estado de perdida de linea y recuperacion
        self.frames_no_line = 0
        self.last_known_side = 1  # +1=derecha, -1=izquierda

        self.get_logger().info("VisionSimNode iniciado con direccion dinamica amortiguada...")

    def image_callback(self, msg):
        t = time.time()
        dt = t - self.last_time
        self.last_time = t
        fps = 1.0 / dt if dt > 0 else 0.0
        self.fps_rolling = 0.9 * self.fps_rolling + 0.1 * fps if self.fps_rolling > 0 else fps

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            frame = cv2.resize(frame, (320, 240))
            h, w = frame.shape[:2]

            # ROI: 60% inferior donde estan las lineas
            roi = frame[int(0.4 * h):, :]
            rh, rw = roi.shape[:2]

            # Deteccion de lineas
            if self.get_parameter('use_hsv').value:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lo = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
                hi = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
                mask = cv2.inRange(hsv, lo, hi)
            else:
                gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
                thr = self.get_parameter('brightness_threshold').value
                _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

            # Limpieza
            k = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

            # Centroide
            M = cv2.moments(mask)
            cx = rw // 2
            line_detected = False
            if M['m00'] > 500:
                cx = int(M['m10'] / M['m00'])
                line_detected = True
                self.frames_no_line = 0
                self.last_known_side = 1 if cx > rw // 2 else -1
            else:
                self.frames_no_line += 1

            error = cx - rw // 2

            # PID
            now = self.get_clock().now()
            dt_pid = (now - self.last_control_time).nanoseconds / 1e9
            self.last_control_time = now
            deriv = (error - self.prev_error) / dt_pid if dt_pid > 0 else 0.0
            self.integral = float(np.clip(self.integral + error * dt_pid, -100.0, 100.0))
            self.prev_error = error

            kp = self.get_parameter('kp').value
            ki = self.get_parameter('ki').value
            kd = self.get_parameter('kd').value
            steering = kp * error + ki * self.integral + kd * deriv

            # Velocidad y maniobra
            follow = self.get_parameter('follow_line').value
            lin, ang = 0.0, 0.0

            if follow:
                twist = Twist()
                base = self.get_parameter('base_speed').value
                max_ang = self.get_parameter('max_angular_speed').value

                if not line_detected:
                    # FASE DE RECUPERACION DE COLISIONES
                    if self.frames_no_line <= 10:
                        # Fase 1: Rotar suavemente en el lugar buscando recuperar
                        twist.linear.x = 0.0
                        twist.angular.z = 0.5 * float(self.last_known_side)
                    elif self.frames_no_line <= 35:
                        # Fase 2: Asumir choque contra pared. ¡Marcha atras (reversa)!
                        twist.linear.x = -0.25
                        twist.angular.z = -0.6 * float(self.last_known_side)
                    else:
                        # Fase 3: Detenerse y rotar buscando la linea de nuevo
                        twist.linear.x = 0.0
                        twist.angular.z = 0.7 * float(self.last_known_side)
                        if self.frames_no_line > 80:
                            self.frames_no_line = 11
                else:
                    # Conduccion normal
                    ratio = abs(error) / (rw // 2)
                    # Velocidad lineal decae proporcionalmente en curvas cerradas
                    twist.linear.x = base * max(0.2, 1.0 - 0.8 * ratio)

                    # --- DIRECCION DINAMICA AMORTIGUADA (Estabilidad F1) ---
                    # A mayor velocidad lineal, reducimos proporcionalmente el limite angular maximo
                    # para evitar que el carro de traccion diferencial derrape bruscamente (spinning)
                    current_max_ang = max_ang
                    if twist.linear.x > 0.3:
                        speed_factor = twist.linear.x / base
                        # Amortiguamos hasta un 50% el rango de giro a maxima velocidad
                        current_max_ang = max_ang * (1.0 - 0.5 * speed_factor)

                    twist.angular.z = float(np.clip(-steering, -current_max_ang, current_max_ang))

                self.cmd_vel_pub.publish(twist)
                lin, ang = twist.linear.x, twist.angular.z

            # Consola
            st = "LINEA OK" if line_detected else f"SIN LINEA ({self.frames_no_line}f)"
            if follow:
                print(f"FPS:{self.fps_rolling:4.1f} | {st} | cx={cx:3d} err={error:+4d}px | Lin:{lin:.2f} Ang:{ang:+.2f}", flush=True)
            else:
                print(f"FPS:{self.fps_rolling:4.1f} | {st} | cx={cx:3d} err={error:+4d}px | AUTONOMO: INACTIVO", flush=True)

            # Debug visual
            if self.get_parameter('show_image').value and 'DISPLAY' in os.environ:
                dbg = roi.copy()
                overlay = np.zeros_like(roi)
                overlay[mask > 0] = (0, 220, 0)
                dbg = cv2.addWeighted(dbg, 0.7, overlay, 0.3, 0)
                if line_detected:
                    cv2.circle(dbg, (cx, rh // 2), 8, (0, 0, 255), -1)
                cv2.line(dbg, (rw // 2, 0), (rw // 2, rh), (255, 0, 0), 2)
                cv2.arrowedLine(dbg, (rw // 2, rh // 2), (cx, rh // 2), (0, 255, 255), 2)
                color = (0, 200, 0) if line_detected else (0, 0, 255)
                cv2.putText(dbg, st, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imshow("ROI original", dbg)
                cv2.imshow("Mascara deteccion", mask)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
