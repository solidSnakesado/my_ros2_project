import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration # 시간 관련 메시지

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher_ = self.create_publisher(JointTrajectory, 'set_joint_trajectory', 10)

        # 2초마다 timer_callback 함수 실행
        self.timer = self.create_timer(2.0, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = JointTrajectory()

        # 1. 관절 이름 명시
        msg.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

        point = JointTrajectoryPoint()

        # ex> 로봇팡을 조금씩 움직이게 설정 (관절 개수 6개)
        if self.i % 2 == 0:
            # 짝수 초: 0도
            point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            # 홀수 초: 약 28도 (0.5 라디안) 움직임
            point.positions = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]

        # 2. 도달 시간 설정 (ex> 1초 동안 부드럽게 이동)
        point.time_from_start = Duration(sec=1, nanosec=0)

        msg.points = [point]

        self.publisher_.publish(msg)
        self.get_logger().info(f"메시지 전송함: {point.positions}")
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 1. 노드가 만들어졌는지 확인 후 파괴
        if node is not None:
            node.destroy_node()
        
        # 2. ROS 시스템이 아직 켜져 있는지 확인 후 종료
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == '__main__':
    main()
