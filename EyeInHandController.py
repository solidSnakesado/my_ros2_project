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

        # --- 설정값 ---
        self.PICK_RATIO = 0.085       
        self.loss_patience = 0
        self.MAX_PATIENCE = 40        
        self.search_angle = 0.0
        self.prev_target_center = None 

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

        # 큐브 색상 마스크
        mask_purple = cv2.inRange(hsv, np.array([130, 60, 50]), np.array([165, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([35, 60, 50]), np.array([85, 255, 255])) 
        mask_yellow = cv2.inRange(hsv, np.array([15, 60, 50]), np.array([40, 255, 255]))
        mask = cv2.bitwise_or(mask_purple, mask_green)
        mask = cv2.bitwise_or(mask, mask_yellow)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_target = None
        if contours:
            valid_candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # [수정] 인식 면적 기준 하향 (300 -> 150)
                if 150 < area < (frame.shape[0] * frame.shape[1] * 0.3):
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        valid_candidates.append((cnt, cx, cy, area))

            if valid_candidates:
                if self.prev_target_center is None:
                    best_target = max(valid_candidates, key=lambda x: x[3])
                else:
                    best_target = min(valid_candidates, key=lambda x: math.hypot(x[1]-self.prev_target_center[0], x[2]-self.prev_target_center[1]))

        if best_target:
            cnt, cx, cy, area = best_target
            self.prev_target_center = (cx, cy)
            self.loss_patience = 0
            
            rect = cv2.minAreaRect(cnt)
            angle = rect[2]
            w, h = rect[1]
            if w < h: angle += 90
            if angle > 45: angle -= 90
            elif angle < -45: angle += 90
            angle_rad = angle * (math.pi / 180.0)

            cv2.drawContours(frame, [np.int0(cv2.boxPoints(rect))], 0, (0, 255, 0), 2)
            self.run_visual_servoing(cx, cy, area, angle_rad, frame.shape)
        else:
            self.loss_patience += 1
            if self.loss_patience > self.MAX_PATIENCE:
                self.state = "SEARCH"
                self.prev_target_center = None
                self.run_search_mode()

        cv2.putText(frame, f"MODE: {self.state} | LOCKED: {self.prev_target_center is not None}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Robot View", frame)
        cv2.waitKey(1)

    def run_visual_servoing(self, cx, cy, area, target_angle, shape):
        self.state = "SERVOING"
        h, w, _ = shape
        err_x, err_y = (w // 2) - cx, (h // 2) - cy
        ratio = area / (w * h)

        if ratio > self.PICK_RATIO:
            self.get_logger().info("Target Reached! Ready to Pick.")
            self.state = "PICK"
            return

        next_q = self.current_joints[:]
        k_pan = 0.0022; k_reach = 0.0028; k_dive = 0.0095; k_rot = 0.15

        next_q[0] += err_x * k_pan          
        next_q[5] += target_angle * k_rot   

        # [검증된 방향] Joint 2를 - 방향으로 접으며 하강
        next_q[1] -= k_dive 
        # Joint 3는 위로 접어 올려(+) 몸체 거리 확보
        next_q[2] += k_dive * 0.6  
        next_q[2] -= err_y * k_reach

        next_q[4] = -1.57 - (next_q[1] + next_q[2])

        # 관절 제한
        next_q[1] = float(np.clip(next_q[1], -2.5, 0.5)) 
        next_q[2] = float(np.clip(next_q[2], -1.5, 3.0)) 
        next_q[4] = float(np.clip(next_q[4], -3.14, 3.14))
        next_q[5] = float(np.clip(next_q[5], -3.14, 3.14))

        self.publish_traj(next_q)

    def run_search_mode(self):
        if self.current_joints is None: return
        next_q = self.current_joints[:]
        
        # [수정] 테이블과 더 가까운 탐색 자세 (Low Crane Pose)
        # J2를 -0.8까지 숙여서 테이블 중심을 가까이서 비춤
        target_shoulder = -0.8 
        target_elbow = 0.8   
        
        next_q[1] = next_q[1] * 0.95 + target_shoulder * 0.05
        next_q[2] = next_q[2] * 0.95 + target_elbow * 0.05
        
        self.search_angle += 0.040
        if self.search_angle > math.pi: self.search_angle -= 2*math.pi
        next_q[0] = self.search_angle
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