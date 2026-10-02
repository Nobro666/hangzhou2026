#!/usr/bin/env python
# coding: UTF-8 

import rospy
from geometry_msgs.msg import Twist,Pose
from nav_msgs.msg import Odometry
import math


class Base:
    def __init__(self):
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot_pose',Pose, self.now_pose)
        self.twist = Twist()
        self.px = 0.0
        self.py = 0.0
        self.pz = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.oz = 0.0
        self.ow = 0.0
        self.position = []  # 坐标
        self.orientation = []  # 四元数
        self.angle = 0  # 角度
        self.pose_received = False
        rospy.sleep(1)  # 等待获取现在位置的回调函数开始工作

    def now_pose(self, pose):
        """实时更新现在的坐标和四元数和角度"""
        self.px = pose.position.x
        self.py = pose.position.y
        self.pz = pose.position.z
        self.ox = pose.orientation.x
        self.oy = pose.orientation.y
        self.oz = pose.orientation.z
        self.ow = pose.orientation.w
        self.position = [self.px, self.py, self.pz]
        self.orientation = [self.ox, self.oy, self.oz, self.ow]
        self.get_angle()
        self.pose_received = True

    def get_pose(self):
        """可调用获取现在的坐标和四元数"""
        print('[[{},{},{}],[{},{},{},{}]]'.format(self.px, self.py, self.pz, self.ox, self.oy, self.oz, self.ow))
        return [[self.px, self.py, self.pz], [self.ox, self.oy, self.oz, self.ow]]

    def get_angle(self):
        """可调用获取现在的角度"""
        eular = self.quad2euler(self.ox, self.oy, self.oz, self.ow)
        self.angle = eular
        return eular

    @staticmethod
    def normalize_angle(angle):
        """将弧度角归一化到 [-pi, pi)。"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def turn(self, angle, kp=1.5, kd=0.05, timeout=10.0,
             max_speed=0.6, min_speed=0.10):
        """
        原地转动指定角度。

        参数 angle 的单位为度，正值左转、负值右转，范围为 [-180, 180]。
        返回 True 表示到达目标角度，False 表示位姿不可用、超时或 ROS 关闭。
        """
        angle = float(angle)
        if angle < -180.0 or angle > 180.0:
            raise ValueError("turn angle must be in [-180, 180] degrees")

        if not self.pose_received:
            try:
                pose = rospy.wait_for_message('/robot_pose', Pose, timeout=2.0)
                self.now_pose(pose)
            except rospy.ROSException:
                rospy.logwarn("底盘转向失败：未收到 /robot_pose")
                self.stop()
                return False

        angle_radian = math.radians(angle)
        if abs(angle_radian) <= 0.02:
            self.stop()
            return True

        start_angle = self.get_angle()
        end_angle = self.normalize_angle(start_angle + angle_radian)
        start_time = rospy.get_time()
        last_time = start_time
        last_error = self.normalize_angle(end_angle - start_angle)
        rate = rospy.Rate(30)

        print(
            "开始底盘转向：目标={:.1f}°，起始角={:.3f} rad，"
            "目标角={:.3f} rad".format(angle, start_angle, end_angle)
        )

        try:
            while not rospy.is_shutdown():
                now = rospy.get_time()
                error = self.normalize_angle(end_angle - self.get_angle())

                if abs(error) <= 0.02:
                    print("底盘转向完成")
                    return True

                if now - start_time >= timeout:
                    rospy.logwarn(
                        "底盘转向超时：剩余误差 %.3f rad", error
                    )
                    return False

                dt = max(now - last_time, 1e-3)
                derivative = (error - last_error) / dt
                speed = kp * error + kd * derivative

                if abs(speed) < min_speed:
                    speed = math.copysign(min_speed, error)
                speed = max(-max_speed, min(max_speed, speed))

                self.rotate(speed)
                last_error = error
                last_time = now
                rate.sleep()
        finally:
            self.stop()

        return False

    def rotate(self, speed):
        """旋转"""
        print('rotating')
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = speed
        self.pub.publish(self.twist)

    def stop(self):
        print('stop')
        self.twist.linear.x = 0
        self.twist.linear.y = 0
        self.twist.linear.z = 0
        self.twist.angular.x = 0
        self.twist.angular.y = 0
        self.twist.angular.z = 0
        self.pub.publish(self.twist)

    def quad2euler(self, x, y, z, w):
        X = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        Y = math.asin(2 * (w * y - x * z))
        Z = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
        return Z


if __name__ == '__main__':
    try:
        rospy.init_node('base', anonymous=True)
        base = Base()
        base.get_pose()
        # while not rospy.is_shutdown():
        #     base.get_angle()
        #     base.rotate(1.0)

        for i in range(10):
            degree = input('Input degree:')
            base.turn(float(degree))
            print(i)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
