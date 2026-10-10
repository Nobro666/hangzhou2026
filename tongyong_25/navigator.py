#!/usr/bin/env python3
# !coding=utf-8
# Created by Cmoon

import rospy
from std_srvs.srv import Empty
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from soundplayer import Soundplayer
from base_controller import Base


class Navigator:
    def __init__(self, location):
        """
        location是字典,键是地点名字(String),值是坐标列表
        例:'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]]
        """
        self.location = location
        self.goal = MoveBaseGoal()  # 实例化MoveBaseGoal这一消息类型
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        self.base = Base()  # 我的底盘控制模块
        rospy.sleep(1)  # 等一秒,增加稳定性
        self.clear_costmap_client = rospy.ServiceProxy('move_base/clear_costmaps', Empty)  # 定义清理代价地图服务
        rospy.sleep(1)
        rospy.loginfo('Navigation ready...')

    def add(self,location,num):
        self.location[str(num)]=location
    def goto(self, place, server_timeout=7.0, goal_timeout=50.0,
             max_attempts=2):
        """调用传入地点导航，成功返回 True，失败返回 False。"""
        if place not in self.location:
            rospy.logerr("未知导航点：%s", place)
            return False
        point = self.set_goal("map", self.location[place][0], self.location[place][1])  # 设置导航点
        success = self.go_to_location(
            point,
            server_timeout=server_timeout,
            goal_timeout=goal_timeout,
            max_attempts=max_attempts,
        )
        if success:
            print('I have got the ' + place)
        else:
            rospy.logwarn("未能到达导航点：%s", place)
        # self.soundplayer.say('I have got the ' + place)
        return success

    def go_near(self, name, position):
        """配合深度相机获取坐标可靠近物体"""
        orientation = self.base.orientation
        now_pose = self.base.position
        print(now_pose)
        pose = []
        pose.append((position[0] + now_pose[0]) / 2)
        pose.append((position[1] + now_pose[1]) / 2)
        pose.append((position[2] + now_pose[2]) / 2)
        point = self.set_goal('map', pose, orientation)
        self.go_to_location(point)
        print('I am near the ' + name)
        self.soundplayer.say('I am near the ' + name)

    def set_goal(self, name, position, orientation):
        """设置导航目标点的坐标和四元数"""
        self.goal.target_pose.header.frame_id = name
        self.goal.target_pose.pose.position.x = position[0]
        self.goal.target_pose.pose.position.y = position[1]
        self.goal.target_pose.pose.position.z = position[2]
        self.goal.target_pose.pose.orientation.x = orientation[0]
        self.goal.target_pose.pose.orientation.y = orientation[1]
        self.goal.target_pose.pose.orientation.z = orientation[2]
        self.goal.target_pose.pose.orientation.w = orientation[3]
        print('Goal set.')
        return self.goal

    def go_to_location(self, location, server_timeout=10.0,
                       goal_timeout=120.0, max_attempts=3):
        """有限次数执行导航，避免 move_base 异常时无限等待。"""
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)  # 等待MoveBaseAction server启动
        if not self.client.wait_for_server(rospy.Duration(server_timeout)):
            rospy.logerr(
                "等待 move_base action server 超时（%.1f 秒）",
                server_timeout,
            )
            return False
        print('Ready to go.')

        for attempt in range(1, max_attempts + 1):
            if rospy.is_shutdown():
                return False

            print('尝试导航...（第{}/{}次）'.format(attempt, max_attempts))
            try:
                rospy.wait_for_service(
                    'move_base/clear_costmaps',
                    timeout=3.0,
                )
                self.clear_costmap_client()
            except (rospy.ROSException, rospy.ServiceException) as error:
                rospy.logwarn("清理代价地图失败：%s", error)

            self.client.send_goal(location)
            finished = self.client.wait_for_result(
                rospy.Duration(goal_timeout)
            )
            if not finished:
                rospy.logwarn(
                    "导航等待超时（%.1f 秒），取消当前目标",
                    goal_timeout,
                )
                self.client.cancel_goal()
                continue

            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                return True

            rospy.logwarn(
                "导航失败，状态码=%s，将按剩余次数重试", state
            )

        self.client.cancel_all_goals()
        return False

    def stop(self):
        self.client.cancel_all_goals()


if __name__ == '__main__':
    rospy.init_node('navigation')
    Navigator('location')
    rospy.spin()
