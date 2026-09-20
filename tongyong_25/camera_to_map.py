#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：将相机坐标系中的坐标点转换成地图坐标系中的点 返回列表[x,y,z]
需要在launch文件中发布相机坐标系和底盘坐标系的静态关系

涉及相机和底盘坐标系正方向不同 所以手动转换正方向 本项目中 x和y转换正常 z有异常 
但是官网提到ros tf2_geometry_msgs库可以自动转换 后续如果其他项目出错可以先考虑手动变换部分的问题

created by zx 2025-10-03
"""

import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

class CoordinateConverter:
    def __init__(self):
        # 初始化 TF2 监听器 
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 坐标系的名称
        self.camera_frame = "camera_link"
        self.map_frame = "map"
        
        # 给 TF 时间来填充缓冲区
        rospy.sleep(1.0)

    def get_map_coords(self, camera_point):
        """
        接收一个在相机坐标系下的三维点，并返回其在地图坐标系中的对应点。

        参数:
            camera_point (list or tuple): 在相机坐标系下的点 [x, y, z]。

        返回:
            list: 在地图坐标系下的点 [x, y, z]，如果转换失败则返回 None。
        """
        if not camera_point:
            rospy.logwarn("收到的 camera_point 为空")
            return None
            
        try:
            # 为 TF 库创建一个 PointStamped 消息
            point_in_camera = PointStamped()
            point_in_camera.header.frame_id = self.camera_frame
            point_in_camera.header.stamp = rospy.Time(0) # 使用最新的可用变换
            point_in_camera.point.x = camera_point[2]
            point_in_camera.point.y = -camera_point[0]
            point_in_camera.point.z = -camera_point[1]

            # 查找从 camera_link 到 map 的变换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0) # 为变换等待最多1秒
            )

            # 应用变换
            point_in_map = do_transform_point(point_in_camera, transform)
            
            rospy.loginfo(f"成功转换坐标至地图 X: {point_in_map.point.x}, Y: {point_in_map.point.y}, Z: {point_in_map.point.z}")
            
            return [point_in_map.point.x, point_in_map.point.y, point_in_map.point.z]

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"坐标变换失败: {e}")
            return None