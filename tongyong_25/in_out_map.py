#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import OccupancyGrid
import time

"""
该函数验证一个点是否在地图内 传入参数为三维点 返回值为布尔值

created by zx 2025-10-11
"""

class MapValidator:

    def __init__(self):
        """
        初始化 MapValidator，订阅 /map 话题并等待地图数据。
        """
        self.map_info = None
        self.map_data = None
        self.map_received = False

        rospy.loginfo("正在初始化 MapValidator...")
        
        # 订阅 /map 话题。通常地图是 latched topic，所以回调函数只会执行一次。
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        
        rospy.loginfo("等待接收 /map 数据...")
        # 循环等待，直到 map_callback 成功接收到数据
        while not self.map_received and not rospy.is_shutdown():
            try:
                time.sleep(0.5)
            except rospy.ROSInterruptException:
                rospy.logerr("节点被关闭，未能接收到地图。")
                return
        
        if self.map_received:
            rospy.loginfo("成功接收到地图数据，MapValidator 已准备就绪。")

    def map_callback(self, msg):
        """
        接收 OccupancyGrid 消息的回调函数。
        """
        self.map_info = msg.info
        self.map_data = msg.data
        self.map_received = True
        rospy.loginfo("回调函数已执行，地图数据已存储。")
        # 接收到一次地图后就注销订阅者，因为地图通常不会改变
        self.map_sub.unregister()

    def is_in_map(self, point_map_frame):
        """
        判断一个在地图坐标系下的三维点是否在所建地图的已知区域内。
        对于2D栅格地图，只考虑 X 和 Y 坐标。

        参数:
            point_map_frame (list or tuple): 在地图坐标系下的点 [x, y, z]。

        返回:
            bool: 如果点在地图的已知区域内，返回 True，否则返回 False。
        """
        if not self.map_received:
            rospy.logwarn("尚未接收到地图数据，无法进行判断。")
            return False

        if not isinstance(point_map_frame, (list, tuple)) or len(point_map_frame) < 2:
            rospy.logerr("传入的坐标点格式不正确，应为 [x, y, z] 或 [x, y]。")
            return False

        # 从地图元数据中获取所需信息
        resolution = self.map_info.resolution  # 地图分辨率 (米/像素)
        width = self.map_info.width            # 地图宽度 (像素)
        height = self.map_info.height          # 地图高度 (像素)
        origin_x = self.map_info.origin.position.x  # 地图原点X坐标
        origin_y = self.map_info.origin.position.y  # 地图原点Y坐标
        
        # 世界坐标 (地图坐标系)
        world_x = point_map_frame[0]
        world_y = point_map_frame[1]

        # --- 核心转换逻辑：从世界坐标转换为栅格坐标 ---
        # 栅格坐标 (grid_x, grid_y) 是从地图左下角开始的像素索引
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        
        rospy.logdebug(f"世界坐标 ({world_x:.2f}, {world_y:.2f}) -> 栅格坐标 ({grid_x}, {grid_y})")

        # --- 检查1：判断栅格坐标是否在地图边界内 ---
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            rospy.loginfo(f"点 ({world_x:.2f}, {world_y:.2f}) 超出地图边界。")
            return False

        # --- 检查2：判断该点是否位于“未知区域” ---
        # 将二维栅格坐标转换为一维数组索引
        index = grid_x + grid_y * width
        
        # OccupancyGrid 中，-1 代表未知区域
        if self.map_data[index] == -1:
            rospy.loginfo(f"点 ({world_x:.2f}, {world_y:.2f}) 位于地图的未知区域。")
            return False
            
        # 如果通过以上所有检查，则该点在地图的已知区域内
        return True

if __name__ == '__main__':
    test_point = [1,1,1] #x y z (list)
    try:
        validator = MapValidator()
        is_valid = validator.is_in_map(test_point)  #in -> true out -> false
        if is_valid:
            print(f"坐标 {test_point[:2]} 在地图的已知区域内。")
        else:
            print(f"坐标 {test_point[:2]} 不在地图的已知区域内。")
    

    except rospy.ROSInterruptException:
        pass
