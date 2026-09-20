#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import tf_conversions
from geometry_msgs.msg import Pose, Point, Quaternion

def calculate_facing_goal(robot_pose: Pose, target_point_map: list, distance: float = 0.3) -> Pose:
    # 1. 提取机器人和目标点的2D坐标
    robot_x = robot_pose.pose.position.x
    robot_y = robot_pose.pose.position.y
    target_x, target_y, _ = target_point_map

    # 2. 计算从机器人指向目标点的向量
    approach_vector_x = target_x - robot_x
    approach_vector_y = target_y - robot_y
    
    vector_magnitude = math.sqrt(approach_vector_x**2 + approach_vector_y**2)

    # 如果机器人与目标点重合，无法计算
    if vector_magnitude < 0.01:
        print("错误：机器人与目标点距离过近。")
        return None

    # 3. 计算单位方向向量
    unit_vector_x = approach_vector_x / vector_magnitude
    unit_vector_y = approach_vector_y / vector_magnitude

    # 4. 计算导航目标点的坐标 (从目标点沿反方向后退)
    goal_x = target_x - distance * unit_vector_x
    goal_y = target_y - distance * unit_vector_y

    # 5. 计算导航目标的姿态 (使其朝向目标点)
    face_target_yaw = math.atan2(target_y - goal_y, target_x - goal_x)
    q_tuple = tf_conversions.transformations.quaternion_from_euler(0, 0, face_target_yaw)

    # 6. 构建并返回最终的 Pose 对象
    goal_pose = [[goal_x, goal_y, 0.138], [q_tuple[0], q_tuple[1], q_tuple[2], q_tuple[3]]]
    return goal_pose