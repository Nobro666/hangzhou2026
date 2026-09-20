#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import PoseStamped, Point
import math

"""
该方法计算机器人面向人的朝向
传入人的三维维坐标 返回一个面向人的朝向点
格式为Location格式[[robot_pose.pose.position.x, robot_pose.pose.position.y, 0.138], [q[0], q[1], q[2], q[3]]]

created by zx 2025-10-16
"""


class facetoPerson:
    def __init__(self):

        self.robot_base_frame = 'base_link'
        self.map_frame = 'map'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_robot_pose(self):
        """获取机器人当前在map坐标系下的位姿"""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.robot_base_frame, 
                rospy.Time(0), 
                rospy.Duration(1.0)
            )
            
            robot_pose = PoseStamped()
            robot_pose.header.stamp = rospy.Time.now()
            robot_pose.header.frame_id = self.map_frame
            robot_pose.pose.position.x = transform.transform.translation.x
            robot_pose.pose.position.y = transform.transform.translation.y
            robot_pose.pose.position.z = transform.transform.translation.z
            robot_pose.pose.orientation = transform.transform.rotation
            
            return robot_pose
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"获取机器人位姿失败: {e}")
            return None

    def face_to_person(self, person_position):
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return None
        face_person_yaw = math.atan2(person_position[1] - robot_pose.pose.position.y, person_position[0] - robot_pose.pose.position.x)
        q = tf_conversions.transformations.quaternion_from_euler(0, 0, face_person_yaw)
        facegoal = [[robot_pose.pose.position.x, robot_pose.pose.position.y, 0.138], [q[0], q[1], q[2], q[3]]]
        return facegoal