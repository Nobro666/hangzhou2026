#!/usr/bin/env python3
# !coding=utf-8
# Created by Cmoon - Optimized Version

import rospy
from std_srvs.srv import Empty
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


class Navigator:
    def __init__(self, location):
        """
        location是字典,键是地点名字(String),值是坐标列表
        例:'door': [[-4.35, -6.18, 0.0], [0.0, 0.0, -0.20, -0.97]]
        """
        self.location = location
        self.goal = MoveBaseGoal()  # 实例化MoveBaseGoal消息类型
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.client.wait_for_server()
        rospy.loginfo("Move_base action server connected.")
        # 定义清理代价地图的服务客户端
        self.clear_costmap_client = rospy.ServiceProxy('move_base/clear_costmaps', Empty)
        rospy.wait_for_service('move_base/clear_costmaps')
        rospy.loginfo("Clear_costmaps service connected.")

        rospy.loginfo('Navigator is ready.')

    def goto(self, place):
        """根据预设的地点名称进行导航"""
        if place not in self.location:
            rospy.logerr(f"Error: Location '{place}' not found in the dictionary.")
            return

        point = self.set_goal("map", self.location[place][0], self.location[place][1])
        self.go_to_location(point)
        rospy.loginfo(f"Successfully arrived at {place}.")

    def set_goal(self, frame_id, position, orientation):
        """设置导航目标点的坐标和姿态"""
        self.goal.target_pose.header.frame_id = frame_id
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = position[2]
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        return self.goal

    def go_to_location(self, location_goal):
        flag = False
        while not flag and not rospy.is_shutdown():  # 导航重试循环
            rospy.loginfo("Attempting to navigate...")
            self.clear_costmap_client()  # 每次尝试前清理代价地图

            self.client.send_goal(location_goal)
            self.client.wait_for_result()

            if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Navigation successful!")
                flag = True
            else:
                rospy.logwarn("Navigation failed. Retrying...")
                rospy.sleep(1)  # 短暂等待后重试

    def stop(self):
        """停止当前的导航任务"""
        rospy.loginfo("Cancelling all navigation goals.")
        self.client.cancel_all_goals()


if __name__ == '__main__':
    try:
        rospy.init_node('navigator_test')

        sample_locations = {
            'start_point': [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            'end_point': [[-1.0, -1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        }

        navigator = Navigator(sample_locations)

        # # --- 以下是调用示例 ---
        # rospy.loginfo("Start navigation test...")
        # navigator.goto('start_point')
        # rospy.sleep(2)
        # navigator.goto('end_point')

        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Navigation node interrupted.")