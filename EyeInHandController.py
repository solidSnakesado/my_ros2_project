import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class EyeInHandController(Node):
    def __init__(self):
        super().__init__("eye_in_hand_controller")

        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.publisher = self.create_publisher(JointTrajectory, "set_joint_trajectory", 10)

        self.bridge = CvBridge()
        self.state = "SEARCH"
        
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.current_joints = None

        # --- 정밀 설정값 ---
        self.TARGET_RATIO = 0.085
        self.CENTER_TOLERANCE = 20
        self.STABLE_COUNT_TARGET = 15 
        self.stable_counter = 0

        self.loss_patience = 0
        self.MAX_PATIENCE = 50

    def joint_callback(self, msg):
        joint_dict = dict(zip(msg.name, msg.position))
        try:
            self.current_joints = [joint_dict[name] for name in self.joint_names]
        except KeyError:
            pass

    def image_callback(self, data):
        if self.current_joints is None: return

        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # [수정] 로봇 몸체(하늘색)를 피하기 위해 색상 범위를 더 엄격하게 제한
        # 보라색: 135-160 (기존보다 좁힘)
        mask_purple = cv2.inRange(hsv, np.array([135, 50, 50]), np.array([160, 255, 255]))
        # 초록색: 45-75 (기존보다 좁힘)
        mask_green = cv2.inRange(hsv, np.array([45, 50, 50]), np.array([75, 255, 255]))
        # 노랑색: 22-33
        mask_yellow = cv2.inRange(hsv, np.array([22, 50, 50]), np.array([33, 255, 255]))

        mask = cv2.bitwise_or(mask_purple, mask_green)
        mask = cv2.bitwise_or(mask, mask_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_found = False
        cx, cy, area, angle_rad = 0, 0, 0, 0.0

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            # [수정] 몸체 일부가 인식되는 것을 방지하기 위해 면적 기준 상향 (400 -> 800)
            if area > 800:
                target_found = True
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                rect = cv2.minAreaRect(c)
                angle = rect[2]
                w, h = rect[1]
                if w < h: angle += 90
                if angle > 45: angle -= 90
                elif angle < -45: angle += 90
                angle_rad = angle * (math.pi / 180.0)

                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        if target_found:
            self.loss_patience = 0
            self.process_logic(cx, cy, area, angle_rad, frame.shape)
        else:
            self.loss_patience += 1
            if self.loss_patience > self.MAX_PATIENCE:
                if self.state != "SEARCH":
                    self.get_logger().info("Target Lost! -> Adjusting View for Search")
                    self.state = "SEARCH"
                    self.stable_counter = 0
                self.execute_search()

        ratio = area / (frame.shape[0]*frame.shape[1]) if frame.shape[0] > 0 else 0
        cv2.putText(frame, f"STATE: {self.state} | RATIO: {ratio:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Robot View", frame)
        cv2.waitKey(1)

    def process_logic(self, cx, cy, area, target_angle, shape):
        h, w, _ = shape
        err_x, err_y = (w // 2) - cx, (h // 2) - cy
        ratio = area / (w * h)

        if self.state == "SEARCH":
            self.state = "CENTERING"
            self.stable_counter = 0

        if self.state == "CENTERING":
            self.control_arm(err_x, err_y, 0, "CENTER")
            if abs(err_x) < self.CENTER_TOLERANCE and abs(err_y) < self.CENTER_TOLERANCE:
                self.stable_counter += 1
                if self.stable_counter >= self.STABLE_COUNT_TARGET:
                    self.state = "ALIGNING"
                    self.stable_counter = 0
            else:
                self.stable_counter = max(0, self.stable_counter - 1)

        elif self.state == "ALIGNING":
            self.control_arm(err_x, err_y, target_angle, "ALIGN")
            if abs(target_angle) < 0.05:
                self.stable_counter += 1
                if self.stable_counter >= 10:
                    self.state = "APPROACHING"
            else:
                self.stable_counter = 0

        elif self.state == "APPROACHING":
            self.control_arm(err_x, err_y, target_angle, "DIVE")
            if ratio > self.TARGET_RATIO:
                self.state = "PICK"
                self.get_logger().info("Target Locked-on Ready!")
            
            if abs(err_x) > 60 or abs(err_y) > 60: 
                self.state = "CENTERING"
                self.stable_counter = 0

    def control_arm(self, err_x, err_y, target_angle, mode):
        if self.current_joints is None: return
        next_q = self.current_joints[:]

        k_p = 0.0016; k_r = 0.0008; k_d = 0.0035; k_rot = 0.12

        next_q[0] += err_x * k_p

        if mode == "CENTER":
            next_q[1] -= err_y * k_r
        elif mode == "ALIGN":
            next_q[1] -= err_y * k_r
            next_q[5] += target_angle * k_rot
        elif mode == "DIVE":
            next_q[1] -= k_d * 0.7
            next_q[2] -= k_d * 0.4
            next_q[2] -= err_y * 0.0004
            next_q[5] += target_angle * k_rot

        next_q[4] = -1.57 - (next_q[1] + next_q[2])

        next_q[1] = float(np.clip(next_q[1], -3.1, 1.5))
        next_q[2] = float(np.clip(next_q[2], -3.5, 1.5))
        next_q[4] = float(np.clip(next_q[4], -3.5, 1.5))
        next_q[5] = float(np.clip(next_q[5], -3.14, 3.14))

        self.publish_traj(next_q)

    def execute_search(self):
        if self.current_joints is None: return
        next_q = self.current_joints[:]
        
        # [중요 수정] 자기 몸체를 보지 않도록 시야를 더 바깥쪽으로 고정
        # 어깨(J2)는 더 앞으로 내밀고(0.6), 팔꿈치(J3)는 약간 펴서(-0.7) 바닥 중심부를 멀리서 보게 함
        target_shoulder = 0.6
        target_elbow = -0.7
        next_q[1] = next_q[1] * 0.95 + target_shoulder * 0.05
        next_q[2] = next_q[2] * 0.95 + target_elbow * 0.05
        
        next_q[0] += 0.035
        if next_q[0] > math.pi: next_q[0] -= 2 * math.pi
        
        next_q[4] = -1.57 - (next_q[1] + next_q[2])
        
        self.publish_traj(next_q)

    def publish_traj(self, positions):
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        p = JointTrajectoryPoint()
        p.positions = positions
        p.time_from_start.nanosec = 100000000
        msg.points.append(p)
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = EyeInHandController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()