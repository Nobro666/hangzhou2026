#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.srv import GetPlan, GetPlanRequest
import math
import numpy as np
import time

"""
该方法在人周围一个安全的环形区域内 找到一个最佳的可达导航目标点
传入人的三维坐标 返回一个可导航点

created by zx 2025-10-03
"""

class SmartGoalFinder:
    """
    一个智能寻找并验证导航目标的ROS节点。
    它会在人的周围一个安全的环形区域内，找到一个最佳的可达导航目标点。
    """
    def __init__(self):
        # rospy.init_node('smart_goal_finder', anonymous=True)

        self.robot_base_frame = 'base_link'
        self.map_frame = 'map'

        self.MAX_SEARCH_RADIUS = 0.6
        self.MIN_SEARCH_RADIUS = 0.3
        
        self.RADIUS_STEP = 0.05
        self.ANGULAR_STEP = 10

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.make_plan_service_name = "/move_base/make_plan"
        rospy.loginfo(f"等待服务 '{self.make_plan_service_name}'...")
        try:
            rospy.wait_for_service(self.make_plan_service_name, timeout=5.0)
            self.make_plan_client = rospy.ServiceProxy(self.make_plan_service_name, GetPlan)
            rospy.loginfo("服务连接成功!")
        except rospy.ROSException as e:
            rospy.logerr(f"连接服务失败: {e}")
            rospy.signal_shutdown("无法连接到 make_plan 服务")
            return

    def get_robot_pose(self):
        """获取机器人当前在map坐标系下的位姿"""
        try:
            # 等待并获取从 map 到 base_link 的变换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.robot_base_frame, 
                rospy.Time(0), 
                rospy.Duration(1.0)
            )
            
            pose = PoseStamped()
            pose.header.frame_id = self.map_frame
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            return pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"获取机器人位姿失败: {e}")
            return None

    def validate_goal(self, start_pose, goal_pose):
        """
        调用 move_base/make_plan 服务来验证一个目标点是否可达。
        如果服务返回的路径不为空, 则认为该点可达。
        """
        req = GetPlanRequest()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = 0.1  # 容忍度设小一点，确保规划到目标点附近
        try:
            res = self.make_plan_client(req)
            if res.plan.poses:  # 如果路径点列表不为空
                return True
            else:
                return False
        except rospy.ServiceException as e:
            rospy.logwarn(f"调用 make_plan 服务异常: {e}")
            return False

    def find_best_goal(self, person_pose_stamped):
        """
        主逻辑函数：围绕人的位置搜索最佳导航目标点。
        
        Args:
            person_pose_stamped(list): 人在map坐标系下的位姿。
        
        Returns:
            PoseStamped: 找到的最佳导航目标位姿, 如果找不到则返回 None。
        """
        # 1. 获取机器人当前位姿作为路径规划的起点
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return None

        person_point_x = person_pose_stamped[0]
        person_point_y = person_pose_stamped[1]
        person_point_z = person_pose_stamped[2]

        robot_point = robot_pose.pose.position

        # 2. 计算机器人到人的初始角度，作为搜索的0度方向
        initial_angle = math.atan2(person_point_y - robot_point.y, person_point_x - robot_point.x)

        # 3. 从外环向内环进行迭代搜索
        current_radius = self.MAX_SEARCH_RADIUS
        while current_radius >= self.MIN_SEARCH_RADIUS:
            rospy.loginfo(f"正在半径 {current_radius:.2f}m 处搜索...")

            # 4. 在当前半径的圆上，从0度开始向两侧扩展搜索角度
            # 角度偏移顺序: 0, +15, -15, +30, -30, ...
            for angle_offset_multiplier in range(0, int(math.pi / self.ANGULAR_STEP) + 1):
                for sign in ([1, -1] if angle_offset_multiplier > 0 else [1]):
                    
                    angle_offset = angle_offset_multiplier * self.ANGULAR_STEP * sign
                    current_angle = initial_angle + angle_offset

                    # 计算候选点的坐标
                    goal_x = person_point_x - current_radius * math.cos(current_angle)
                    goal_y = person_point_y - current_radius * math.sin(current_angle)

                    # 计算朝向人的姿态 (yaw角)
                    face_person_yaw = math.atan2(person_point_y - goal_y, person_point_x - goal_x)
                    q = tf_conversions.transformations.quaternion_from_euler(0, 0, face_person_yaw)
                    
                    # 构建候选目标位姿
                    candidate_goal = PoseStamped()
                    candidate_goal.header.frame_id = self.map_frame
                    candidate_goal.header.stamp = rospy.Time.now()
                    candidate_goal.pose.position.x = goal_x
                    candidate_goal.pose.position.y = goal_y
                    candidate_goal.pose.position.z = 0.138 # Z轴高度与人保持一致   0.138
                    candidate_goal.pose.orientation.x = q[0]
                    candidate_goal.pose.orientation.y = q[1]
                    candidate_goal.pose.orientation.z = q[2]
                    candidate_goal.pose.orientation.w = q[3]

                    # 5. 验证该候选点是否可达
                    rospy.loginfo(f"  -> 验证角度: {math.degrees(current_angle):.1f}°, "
                                f"坐标: ({goal_x:.2f}, {goal_y:.2f})")
                    time.sleep(0.1)
                    if self.validate_goal(robot_pose, candidate_goal):
                        rospy.loginfo(f"成功找到有效目标点！半径: {current_radius:.2f}m, "
                                      f"坐标: ({goal_x:.2f}, {goal_y:.2f})")
                        goodgoal = [[goal_x,goal_y,0.138],[q[0],q[1],q[2],q[3]]]
                        return goodgoal

            # 如果当前半径的所有点都无效，则向内缩小半径
            current_radius -= self.RADIUS_STEP
        
        rospy.logwarn("在整个定义的环形区域内都未找到有效的导航目标点。")
        return None


if __name__ == '__main__':
    try:
        finder = SmartGoalFinder()

        rospy.sleep(1.0) 

        person_pose = PoseStamped()
        person_pose.header.frame_id = "map"
        person_pose.header.stamp = rospy.Time.now()
        person_pose.pose.position.x = 2.0
        person_pose.pose.position.y = 2.0
        person_pose.pose.position.z = 0.0 
        person_pose.pose.orientation.w = 1.0 # 朝向无所谓

        rospy.loginfo("开始为虚拟的人的位置搜索目标点...")
        
        # 调用核心函数
        best_goal = finder.find_best_goal(person_pose)

        if best_goal:
            rospy.loginfo("\n--- 最终找到的最佳目标点 ---")
            rospy.loginfo(f"坐标 (x, y, z): ({best_goal.pose.position.x:.3f}, "
                          f"{best_goal.pose.position.y:.3f}, {best_goal.pose.position.z:.3f})")
            rospy.loginfo(f"朝向四元数 (x, y, z, w): ({best_goal.pose.orientation.x:.3f}, "
                          f"{best_goal.pose.orientation.y:.3f}, {best_goal.pose.orientation.z:.3f}, "
                          f"{best_goal.pose.orientation.w:.3f})")
            # 在实际应用中，您可以在这里将 `best_goal` 发送给 move_base
            # goal_publisher.publish(best_goal)
        else:
            rospy.logerr("搜索失败，没有找到可达的目标点。")

    except rospy.ROSInterruptException:
        pass
