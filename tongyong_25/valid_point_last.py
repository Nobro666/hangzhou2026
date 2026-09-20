#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.srv import GetPlan, GetPlanRequest
import math
import numpy as np

"""
此方案作为备用方案 
传入人的坐标 返回一个机器人可导航的目标点 如果没有在理想区域找到目标点 那就尽量靠近人
如果代码position_last_second.py能用就用position_last_second.py

created by zx 2025-10-11

"""

class SmartGoalFinder:
    """
    一个智能寻找并验证导航目标的ROS节点。
    它会优先在人的周围一个安全的环形区域内寻找最佳目标点。
    如果找不到，则会启动后备模式，寻找离人最近的一个可达点。
    """
    def __init__(self):
        # rospy.init_node('smart_goal_finder', anonymous=True)

        self.robot_base_frame = 'base_link'
        self.map_frame = 'map'
        
        # 阶段一：理想区域参数
        self.MAX_SEARCH_RADIUS = 0.8
        self.MIN_SEARCH_RADIUS = 0.5
        self.RADIUS_STEP = 0.05
        
        # 阶段二：后备搜索参数
        self.FALLBACK_MAX_RADIUS = 1.2  #如果没有找到目标点 也尽量移动到人附近 
        self.FALLBACK_RADIUS_STEP = 0.1
        
        # 通用角度步长
        self.ANGULAR_STEP_DEG = 15 # 使用角度制，更直观
        self.ANGULAR_STEP_RAD = math.radians(self.ANGULAR_STEP_DEG)

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
        """调用 move_base/make_plan 服务来验证一个目标点是否可达。"""
        req = GetPlanRequest()
        req.start = start_pose
        req.goal = goal_pose
        req.tolerance = 0.1
        try:
            res = self.make_plan_client(req)
            return bool(res.plan.poses)
        except rospy.ServiceException as e:
            rospy.logwarn(f"调用 make_plan 服务异常: {e}")
            return False

    def find_best_goal(self, person_pose_stamped):
        """
        主逻辑函数：围绕人的位置搜索最佳导航目标点。
        采用两阶段策略：优先搜索理想环形区域，失败后启动“尽力而为”的最近点搜索。
        """
        # 1. 获取机器人当前位姿作为路径规划的起点
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return None

        person_point_x = person_pose_stamped[0]
        person_point_y = person_pose_stamped[1]
        
        robot_point = robot_pose.pose.position
        initial_angle = math.atan2(person_point_y - robot_point.y, person_point_x - robot_point.x)

        # --- 阶段一：在理想环形区域内搜索 (0.5m - 0.8m) --- 此处根据实际调整 0.4-0.7适合本项目
        rospy.loginfo("--- 阶段一：开始搜索理想交互点 ---")
        current_radius = self.MAX_SEARCH_RADIUS
        while current_radius >= self.MIN_SEARCH_RADIUS:
            rospy.loginfo(f"正在理想半径 {current_radius:.2f}m 处搜索...")
            
            for angle_offset_multiplier in range(0, int(180 / self.ANGULAR_STEP_DEG) + 1):
                for sign in ([1, -1] if angle_offset_multiplier > 0 else [1]):
                    angle_offset = math.radians(angle_offset_multiplier * self.ANGULAR_STEP_DEG * sign)
                    current_angle = initial_angle + angle_offset
                    
                    goal_x = person_point_x - current_radius * math.cos(current_angle)
                    goal_y = person_point_y - current_radius * math.sin(current_angle)

                    face_person_yaw = math.atan2(person_point_y - goal_y, person_point_x - goal_x)
                    q = tf_conversions.transformations.quaternion_from_euler(0, 0, face_person_yaw)
                    
                    candidate_goal = PoseStamped()
                    candidate_goal.header.frame_id = self.map_frame
                    candidate_goal.header.stamp = rospy.Time.now()
                    candidate_goal.pose.position.x = goal_x
                    candidate_goal.pose.position.y = goal_y
                    candidate_goal.pose.position.z = 0.138
                    candidate_goal.pose.orientation.x = q[0]
                    candidate_goal.pose.orientation.y = q[1]
                    candidate_goal.pose.orientation.z = q[2]
                    candidate_goal.pose.orientation.w = q[3]

                    rospy.loginfo(f"  -> 验证角度: {math.degrees(current_angle):.1f}°, 坐标: ({goal_x:.2f}, {goal_y:.2f})")
                    if self.validate_goal(robot_pose, candidate_goal):
                        rospy.loginfo(f"成功找到理想目标点！半径: {current_radius:.2f}m")
                        goodgoal = [[goal_x, goal_y, 0.138], [q[0], q[1], q[2], q[3]]]
                        return goodgoal

            current_radius -= self.RADIUS_STEP
        
        rospy.logwarn("阶段一失败：在理想环形区域内未找到有效目标。")

        # --- 阶段二：没有目标点也尽量靠过去 ---
        rospy.loginfo("--- 阶段二：开始搜索最近的可达点 ---")
        valid_fallback_goals = []
        
        # 从人身边一个很小的半径开始，一圈圈向外扩大搜索
        current_radius = self.FALLBACK_RADIUS_STEP + 0.3 
        while current_radius <= self.FALLBACK_MAX_RADIUS:
            rospy.loginfo(f"正在后备半径 {current_radius:.2f}m 处搜索...")
            for angle_deg in range(0, 360, self.ANGULAR_STEP_DEG):
                current_angle = math.radians(angle_deg)
                
                # 注意：这里是 +，因为我们是在人的坐标基础上向外扩展
                goal_x = person_point_x + current_radius * math.cos(current_angle)
                goal_y = person_point_y + current_radius * math.sin(current_angle)

                face_person_yaw = math.atan2(person_point_y - goal_y, person_point_x - goal_x)
                q = tf_conversions.transformations.quaternion_from_euler(0, 0, face_person_yaw)

                candidate_goal = PoseStamped()
                candidate_goal.header.frame_id = self.map_frame
                candidate_goal.header.stamp = rospy.Time.now()
                candidate_goal.pose.position.x = goal_x
                candidate_goal.pose.position.y = goal_y
                candidate_goal.pose.position.z = 0.138
                candidate_goal.pose.orientation.x = q[0]
                candidate_goal.pose.orientation.y = q[1]
                candidate_goal.pose.orientation.z = q[2]
                candidate_goal.pose.orientation.w = q[3]

                if self.validate_goal(robot_pose, candidate_goal):
                    # 找到了一个可达点，记录下来它的姿态和与人的距离
                    distance_to_person = math.sqrt((goal_x - person_point_x)**2 + (goal_y - person_point_y)**2)
                    valid_fallback_goals.append((distance_to_person, candidate_goal))
            
            # 如果在当前半径已经找到了至少一个点，就可以停止向外扩张，因为我们想找最近的
            if valid_fallback_goals:
                rospy.loginfo(f"在半径 {current_radius:.2f}m 处发现可达点，停止扩大搜索。")
                break
            
            current_radius += self.FALLBACK_RADIUS_STEP

        if not valid_fallback_goals:
            rospy.logerr("阶段二失败：在整个后备搜索区域内都未找到任何有效导航点。")
            return None
        
        # 按距离排序，选择第一个（即最近的）
        valid_fallback_goals.sort(key=lambda x: x[0])
        best_fallback_goal_pose = valid_fallback_goals[0][1]
        
        bf_x = best_fallback_goal_pose.pose.position.x
        bf_y = best_fallback_goal_pose.pose.position.y
        bf_q = best_fallback_goal_pose.pose.orientation
        
        rospy.loginfo(f"成功找到一个'尽力而为'的目标点！距离人物: {valid_fallback_goals[0][0]:.2f}m")
        goodgoal = [[bf_x, bf_y, 0.138], [bf_q.x, bf_q.y, bf_q.z, bf_q.w]]
        return goodgoal

#测试代码
if __name__ == '__main__':
    try:
        rospy.init_node('smart_goal_finder_test', anonymous=True)
        
        finder = SmartGoalFinder()

        # 等待TF等服务准备就绪
        rospy.sleep(1.0) 

        person_position_test = [2.0, 2.0, 0.0]

        rospy.loginfo(f"开始为虚拟的人的位置 {person_position_test} 搜索目标点...")
        
        # 调用核心函数
        best_goal_data = finder.find_best_goal(person_position_test)

        if best_goal_data:
            pos = best_goal_data[0]
            ori = best_goal_data[1]
            rospy.loginfo("\n--- 最终找到的最佳目标点 ---")
            rospy.loginfo(f"坐标 (x, y, z): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            rospy.loginfo(f"朝向四元数 (x, y, z, w): ({ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f})")
            # 在实际应用中，您可以在这里将 best_goal 发送给 move_base
            # goal_msg = PoseStamped()
            # ... (填充消息)
            # goal_publisher.publish(goal_msg)
        else:
            rospy.logerr("搜索失败，没有找到可达的目标点。")

    except rospy.ROSInterruptException:
        pass