import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2
import numpy as np

class EyeInHandController(Node):
    def __init__(self):
        super().__init__("eye_in_hand_controller")

        # 1. 유니티 카메라 이미지 구독
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        
        # 2. 로봇 관절 상태 구독
        self.joint_sub = self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)

        # 3. 로봇 관절 명령 발행
        self.publisher = self.create_publisher(JointTrajectory, "set_joint_trajectory", 10)

        self.bridge = CvBridge()
        
        self.state = "SEARCH"   # SEARCH, TRACKING, APPROACH, PICK
        self.search_step = 0
        self.current_joints = None 
        
        # --- 설정값 ---
        self.TARGET_RATIO = 0.06
        
        # [수정 포인트 1] 허용 오차를 20 -> 50으로 완화 (더 쉽게 APPROACH로 진입)
        self.CENTER_TOLERANCE = 50 

    def joint_callback(self, msg):
        # [수정 포인트 2] 관절 매핑 (Joint Mapping) - 순서 섞임 방지
        # 들어오는 메시지의 이름과 각도를 딕셔너리로 만듦
        joint_dict = {name: pos for name, pos in zip(msg.name, msg.position)}
        
        # 우리가 원하는 순서대로 리스트 재조립 (Niryo One 기준)
        # 만약 에러가 난다면 유니티의 관절 이름을 확인해야 함 (보통 joint_1, joint_2... 또는 joint1, joint2...)
        try:
            self.current_joints = [
                joint_dict['joint1'],
                joint_dict['joint2'],
                joint_dict['joint3'],
                joint_dict['joint4'],
                joint_dict['joint5'],
                joint_dict['joint6']
            ]
        except KeyError:
            # 만약 이름이 joint1이 아니라 joint_1 등으로 되어있을 경우를 대비해 예외처리
            # 현재 유니티 설정에 따라 다를 수 있음. 에러 발생 시 로그 확인 필요.
            # 일단 순서대로 넣어봄 (차선책)
            self.current_joints = list(msg.position)

    def image_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 색상 필터 (보라/초록/노랑) - 그림자 고려하여 V값 50으로 낮춤
        mask = cv2.inRange(hsv, np.array([130, 50, 50]), np.array([160, 255, 255])) # 보라
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))) # 초록
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array([20, 50, 50]), np.array([35, 255, 255]))) # 노랑

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_found = False
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            # 노이즈 필터 (크기 300 이상)
            if area > 300:
                target_found = True
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    self.process_state_machine(cx, cy, area, frame.shape)
            else:
                target_found = False

        if not target_found:
            if self.state != "SEARCH":
                self.get_logger().info("Target Lost! -> SEARCH")
                self.state = "SEARCH"
            self.execute_search_pattern()

        # UI 표시
        cv2.putText(frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.imshow("Mask View", mask) # 마스크 뷰 끄고 싶으면 주석 처리
        cv2.imshow("Robot View", frame)
        cv2.waitKey(1)

    def process_state_machine(self, cx, cy, area, shape):
        height, width, _ = shape
        center_x, center_y = width // 2, height // 2
        
        image_total_area = width * height
        current_ratio = area / image_total_area
        
        error_x = center_x - cx
        error_y = center_y - cy

        if self.state == "SEARCH":
            self.state = "TRACKING"
            self.get_logger().info("Target Found! -> TRACKING")

        elif self.state == "TRACKING":
            # 1. 중심 맞추기
            self.control_robot(error_x, error_y, approach_z=False)

            # [수정 포인트 3] 디버깅 로그 추가 (왜 안 넘어가는지 확인용)
            # self.get_logger().info(f"Tracking.. ErrX:{error_x}, ErrY:{error_y}")

            # 2. 오차가 50 미만이면 APPROACH 진입
            if abs(error_x) < self.CENTER_TOLERANCE and abs(error_y) < self.CENTER_TOLERANCE:
                self.state = "APPROACH"
                self.get_logger().info(f"Centered! -> APPROACH (Ratio: {current_ratio:.4f})")

        elif self.state == "APPROACH":
            self.control_robot(error_x, error_y, approach_z=True)
            if current_ratio > self.TARGET_RATIO:
                self.state = "PICK"
                self.get_logger().info(f"Target Reached! -> READY TO PICK")

        elif self.state == "PICK":
            pass

    def control_robot(self, error_x, error_y, approach_z):
        if self.current_joints is None: return

        next_joints = self.current_joints[:]

        k_pan = 0.0003
        k_reach = 0.0003
        k_descend = 0.001

        # --- 제어 방향 (부호) ---
        # 로봇이 반대로 움직이면 여기 부호를 바꾸세요 (+= / -=)
        next_joints[0] += error_x * k_pan  # X축 (좌우)
        next_joints[1] -= error_y * k_reach # Y축 (앞뒤)

        if approach_z:
            next_joints[2] -= k_descend # Z축 (하강)
            next_joints[4] += k_descend # 손목 보정

        # 관절 제한
        next_joints[1] = np.clip(next_joints[1], -1.5, 1.5)
        next_joints[2] = np.clip(next_joints[2], -1.5, 0.5)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        point.positions = next_joints
        point.time_from_start.nanosec = 100000000 
        
        traj_msg.points.append(point)
        self.publisher.publish(traj_msg)
        
    def execute_search_pattern(self):
        self.search_step += 1
        traj_msg = JointTrajectory()
        traj_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        
        angle = 3.14 * np.sin(self.search_step * 0.05)
        point.positions = [angle, 0.0, -0.7, 0.0, -0.7, 0.2]
        point.time_from_start.nanosec = 500000000
        
        traj_msg.points.append(point)
        self.publisher.publish(traj_msg)

def main():
    rclpy.init()
    node = EyeInHandController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()