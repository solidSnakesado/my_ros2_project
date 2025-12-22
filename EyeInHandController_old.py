import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import cv2 
import numpy as np
from sensor_msgs.msg import Image, JointState   # JointState 메시지 타입을 임포트

class EyeInHandController(Node):
    def __init__(self):
        super().__init__("eye_in_hand_controller")

        # 1. 유니티 카메라 이미지 구독
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        # 2. 현재 로봇의 관절 상태(각도)를 실시간으로 파악하기 위한 구독
        self.joint_sub = self.create_subscription(JointState, "joint_states", self.joint_callback, 10)

        # 3. 로봇 관절 명령 발행
        self.publisher = self.create_publisher(JointTrajectory, "set_joint_trajectory", 10)

        self.bridge = CvBridge()

        # --- 상태 정의 ---
        self.state = "SEARCH"   # SEARCH, TRACKING, APPROACH, PICK
        self.search_step = 0
        
        # 현재 관절 각도를 저장할 변수 (초기값 None)
        self.current_joints = None
        # --- 상태 정의 ---

        # --- 비율(Ratio) 기반 설정 ---
        # 화면 전체 면적 대비 물체가 차지하는 비율 (0.0 ~ 1.0)
        # ex> 0.06은 화면의 6% 크기 (640x480 해상도라먄 약 18000 픽셀에 해당)
        self.TARGET_RATIO = 0.06
        self.CENTER_TOLERANCE = 20  # 중심 오차 허용 범위 (픽셀)

    # 관절 상태 콜백 함수
    def joint_callback(self, msg):
        # 로봇의 현재 관절 각도 (position)를 저장
        # msg.name과 msg.position 매핑이 필요
        # 보톤 순서대로 joint1~joint6 인 경우가 많음 (확인 필요)
        self.current_joints = list(msg.position)

    def image_callback(self, data):
        # ROS 이미지를 OpenCV로 변환
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 보라
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])

        # 1. 색상 별 범위 정의
        # 초록
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        # 노랑
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([35, 255, 255])

        # 2. 각 색상 마스크 생성
        mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 3. 모든 마스크 합치기 (or 연산)
        full_mask = cv2.bitwise_or(mask_purple, mask_green)
        full_mask = cv2.bitwise_or(full_mask, mask_yellow)

        # full_mask 로 r/g/y 색상 검출 가능
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 노이즈 필터링 로직 추가
        target_found = False

        if contours:
            # 큐브 발견! -> 추적(Tracking) 모드
            self.state = "TRACKING"
            self.search_step = 0  # 탐색 단계 초기화

            # 가장 큰 물체의 중심점 찾기
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            # M = cv2.moments(c)

            # 면적이 300보다 클 때만 큐브로 인식 하고 추적 (노이즈 무시)
            if area > 300:
                target_found = True
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # # 화면 중심과의 오차 계산 및 로봇 제어 명령 생성 로직 호출
                    # self.control_robot_to_target(cx, cy, frame.shape)

                    # 상태 머신 처리
                    self.process_state_machine(cx, cy, area, frame.shape)
            else:
                # 너무 작음 (노이즈)
                target_found = False
        
        if not target_found:
            # 큐브 상실! -> 탐색(Search) 모드 진입
            if self.state != "SEARCH":
                self.get_logger().info("Target Lost! Switching to SEARCH mode.")
                self.state = "SEARCH"

            self.execute_search_pattern()

        # 상태 표시 UI
        cv2.putText(frame, f"State: {self.state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Robot View", frame)
        cv2.imshow("Mask View", full_mask)  # 이 창에서는 큐브만 흰색으로
        cv2.waitKey(1)

    def process_state_machine(self, cx, cy, area, shape):
        height, width, _ = shape
        center_x, center_y = width // 2, height // 2

        # --- 비율 계산 ---
        image_total_area = width * height
        current_ratio = area / image_total_area

        error_x = center_x - cx
        error_y = center_y - cy

        if self.state == "SEARCH":
            self.state = "TRACKING"
            self.get_logger().info("Target Found! -> TRACKING")

        elif self.state == "TRACKING":
            # 중심 맞추기 (z축 이동은 아직 없음)
            self.control_robot(error_x, error_y, approach_z=False)

            if abs(error_x) < self.CENTER_TOLERANCE and abs(error_y) < self.CENTER_TOLERANCE:
                self.state = "APPROACH"
                self.get_logger().info(f"Centered! -> APPROACH (Current Ratio: {current_ratio: .4f})")

        elif self.state == "APPROACH":
            # 중심 유지하며 하간 (Z축 이동)
            self.control_robot(error_x, error_y, approach_z=True)

            # 비율 조건 체크
            if current_ratio > self.TARGET_RATIO:
                self.state = "PICK"
                self.get_logger().info(f"Target Reached (Ratio: {current_ratio: .4f}) -> READY TO PICK")

        elif self.state == "PICK":
            # 집기 동작 대기
            pass

    def control_robot(self, error_x, error_y, approach_z):
        if self.current_joints is None:
            return
        
        next_joints = self.current_joints[:]

        # Gain 값 (반응 속도)
        k_pan = 0.002
        k_reach = 0.003
        k_descend = 0.001

        # x, joint 1 (base)
        next_joints[0] -= error_x * k_pan

        # y, joint 2 (shoulder)
        next_joints[1] -= error_y * k_reach

        # 하강 로직
        if approach_z == True:
            # joint 3 (elbow)를 조절하여 하강
            # (로봇 모델에 따라 부호 확인 필요, 내려가는 방향으로 설정)
            next_joints[2] -= k_descend

            # 카메라 각도 유지 (joint 5)
            next_joints[4] += k_descend

        # 관절 제한 (safety)
        next_joints[1] = np.clip(next_joints[1], -1.5, 1.5)
        next_joints[2] = np.clip(next_joints[2], -1.5, 0.5)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()
        point.positions = next_joints
        point.time_from_start.nanosec = 100000000

        traj_msg.points.append(point)
        self.publisher.publish(traj_msg)


    # def control_robot_to_target(self, cx, cy, shape):
    #     # 큐브가 보일 때 중심에 맞추도록 로봇 이동
        
    #     # 현재 관절 각도를 모르면 제어 불가
    #     if self.current_joints is None:
    #         return
        
    #     # 1. 화면 중심 좌표 계산
    #     height, width, _ = shape
    #     center_x = width // 2
    #     center_y = height // 2

    #     # 2. 오차(Error) 계산 (픽셀 단위)
    #     error_x = center_x - cx     # 가로 오차
    #     error_y = center_y - cy     # 세로 오차

    #     # 3. 비례 상수(Gain) 설정 (반응 속도 조절)
    #     # 값이 너무 크면 로봇이 확확 움직여서 발생(오버슈팅)하고, 너무 작으면 느림
    #     k_pan = 0.001       # 좌우 회전 (Joint) 민감도
    #     k_reach = 0.001      # 상하/전후 (Joint 2) 민감도

    #     # 4. 다음 관절 각도 계산
    #     # 리스트 복사 (원본 보존)
    #     next_joints = self.current_joints[:]

    #     # x축 오차 -> Joint 1 (Base) 회전
    #     # 화면의 물체가 중심보다 왼쪽에 있으면 (error_x > 0), 로봇을 왼쪽(+)으로 회전
    #     # (반대라면 부호를 -로 변경 필요)
    #     next_joints[0] += error_x * k_pan

    #     # y축 오차 -> Joint 2 (Shoulder) 조절 
    #     # 물체가 화면 위쪽(멀리)에 있으면 (error_y > 0), 어깨를 앞으로 뻗어야 함
    #     # Niryo One에서 Joint 2는 보통 값이 줄어드면 앞으로 숙여짐 (방향 확인 필요)
    #     # (반대라면 부로를 변경)
    #     next_joints[1] -= error_y * k_reach

    #     # 안전 장치, 과도하게 꺽이지 않도록 관절 한계 설정
    #     # Joint 2가 바닥에 박거나 너무 뒤로 넘어가지 않게 제한
    #     next_joints[1] = np.clip(next_joints[1], -1.0, 1.0)

    #     # 4. 명령 발행
    #     traj_msg = JointTrajectory()
    #     traj_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    #     point = JointTrajectoryPoint()
    #     point.positions = next_joints
    #     point.time_from_start.nanosec = 100000000 # 0.1초

    #     traj_msg.points.append(point)
    #     self.publisher.publish(traj_msg)

    #     # Niryo One의 관절 리스트는 보통 아래 순서입니다.
    #     # [0] Joint 1 (Base): 좌우 회전
    #     # [1] Joint 2 (Shoulder): 팔 전체를 앞/뒤로 보냄 (거리 조절 핵심)
    #     # [2] Joint 3 (Elbow): 높이 조절 보조
    #     # [3] Joint 4 (Forearm): 회전
    #     # [4] Joint 5 (Wrist): 카메라 각도 (이걸 고정해야 집기 편함)
    #     # [5] Joint 6 (Hand): 회전

    def execute_search_pattern(self):
        # 큐브가 보이지 않을 때 로봇 팔을 지그재그나 나선형으로 움직임
        self.search_step += 1

        traj_msg = JointTrajectory()
        traj_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        point = JointTrajectoryPoint()

        # joint1(Base)을 크게 회전시켜 뒤편까지 확인 (예: -3.14 ~ 3.14 라디안)
        # 사인 함수를 이용하여 좌우 180도 전 영역을 스캔
        angle = 3.14 * np.sin(self.search_step * 0.05)
        point.positions = [
            angle, 
            -0.0,  # Shoulder: 기존 0.7에서 위로 꺾였다면, -0.8 정도로 앞으로 숙여야 합니다.
            -0.7,   # Elbow: 마이너스일 때 위로 꺾였다면, 플러스 1.5(약 85도)로 안쪽으로 완전히 굽힙니다.
            0.0, 
            -0.7,   # Wrist: 카메라가 바닥을 정면으로 보게 하기 위해 양수 값을 크게 줍니다.
            0.2        
        ]
        point.time_from_start.nanosec = 500000000

        traj_msg.points.append(point)
        self.publisher.publish(traj_msg)

def main():
    rclpy.init()
    node = EyeInHandController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

    