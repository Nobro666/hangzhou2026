#!/usr/bin/env python3
# coding: UTF-8

"""
xg_arm_waypoints.py —— 机械臂依次经过多个路点，最终到达终点（自包含，无 YOLO）

仿照 xg_arm_pose.py 的 6 自由度位姿控制，区别在于：本程序一次走多个路点。
    - 途经点 1 (WAYPOINT_1) -> 途经点 2 (WAYPOINT_2) -> 终点 (WAYPOINT_3)
    - 每个路点都是【基座坐标系】下的 6 自由度位姿 [x, y, z, tx, ty, tz]
        * 前三个 = 位置，单位：米
        * 后三个 = 末端姿态欧拉角 XYZ，单位：度
    - 逐个执行：每个路点到达(SUCCEEDED)后才走下一个；中途失败会中止并告警。

特点：
    - 不 import 项目里的 catch_ground / catch.py，不依赖 YOLO / ultralytics / pyrealsense2。
    - 内置最小化 Kinova 笛卡尔控制类 KinovaArm（只用到 actionlib + kinova_msgs）。
    - 只做：拿到 3 个路点 -> 依次移动 -> (可选)回 home。

运行方式：
    python3 xg_arm_waypoints.py                 # 依次走下面写死的三个路点
    python3 xg_arm_waypoints.py --home          # 只回 home 位
    python3 xg_arm_waypoints.py --pose x y z tx ty tz   # 只走一个临时位姿（调试用）

运行环境：
    ROS1 (Noetic) + Kinova j2n6s300 机械臂

所有需要现场标定的数值都集中在下方“可调参数”区域，请自行修改调试。
"""

import math
import argparse

import numpy as np
import rospy
import actionlib

from geometry_msgs.msg import Point
import std_msgs.msg
import geometry_msgs.msg
import kinova_msgs.msg


# =============================================================================
# 可调参数（现场标定，请自行修改）
# =============================================================================
KINOVA_TYPE = 'j2n6s300'                      # 机械臂型号

# 笛卡尔动作超时（秒）：大范围移动耗时较长，太小会误判超时、中途取消动作
ARM_TIMEOUT_SEC = 60.0                        # TODO: 现场标定（路径越长可再调大）

# home 位 [x, y, z, tx, ty, tz]（基座坐标系，米 + 度）
HOME_POSE = [0.324453, -0.419172, 0.447952, 90.136, 20.215, 2.928]  # TODO: 现场标定

# =============================================================================
# ★★★ 三个路点（在这里手动写死！）★★★
# =============================================================================
# 机械臂会依次经过 途经点1 -> 途经点2 -> 终点，逐个执行、每个到达(SUCCEEDED)后才走下一个。
# 每个路点都是【基座坐标系】下的 6 自由度位姿 [x, y, z, tx, ty, tz]：
#   前三个 = 位置(米)，后三个 = 末端姿态欧拉角 XYZ(度)。
# 若三个路点姿态一样，就把后三个数抄成一样的即可。
#
# ★★★ 只改下面三行！★★★
WAYPOINT_1 = [0.267814, 0.121936, 0.391259, -92.540, 41.364, -166.850]   # 途经点 1
WAYPOINT_2 = [0.360364, -0.070202, 0.629747, -82.299, 59.369, 125.176]   # 途经点 2
WAYPOINT_3 = [-0.114738, -0.365686, 0.762665, 83.859, 28.168, -69.590]   # 终点

# 所有路点依次执行（要加/减路点，改这一行即可）
WAYPOINTS = [WAYPOINT_1, WAYPOINT_2, WAYPOINT_3]


class KinovaArm:
    """最小化 Kinova 笛卡尔控制（只发 tool_pose 位姿，不含夹爪/YOLO）。"""

    def __init__(self, robot_type=KINOVA_TYPE):
        self.prefix = robot_type + '_'
        self.action_addr = '/' + self.prefix + 'driver/pose_action/tool_pose'
        self.client = actionlib.SimpleActionClient(
            self.action_addr, kinova_msgs.msg.ArmPoseAction)
        if not self.client.wait_for_server(rospy.Duration(5.0)):
            raise RuntimeError("[xg_arm_waypoints] 无法连接机械臂 action server: " + self.action_addr)
        self.goal = kinova_msgs.msg.ArmPoseGoal()
        self.goal.pose.header = std_msgs.msg.Header(frame_id=self.prefix + 'link_base')
        rospy.loginfo("[xg_arm_waypoints] 机械臂已连接: %s", self.action_addr)

    def arm_run(self, pose_target):
        """
        发送 6 自由度笛卡尔目标位姿。
        pose_target = [x, y, z, tx_deg, ty_deg, tz_deg]
        前三个是位置（米），后三个是欧拉角 XYZ（度）。
        返回 (state, result)；超时返回 (None, None)。
        """
        pos = pose_target[:3]
        qx, qy, qz, qw = self._euler_deg_to_quat(*pose_target[3:])
        self.goal.pose.pose.position = geometry_msgs.msg.Point(
            x=pos[0], y=pos[1], z=pos[2])
        self.goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=qx, y=qy, z=qz, w=qw)
        self.client.send_goal(self.goal)
        if not self.client.wait_for_result(rospy.Duration(ARM_TIMEOUT_SEC)):
            self.client.cancel_all_goals()
            rospy.logwarn("[xg_arm_waypoints] 笛卡尔动作超时")
            return None, None
        state = self.client.get_state()
        result = self.client.get_result()
        # 3=SUCCEEDED 才算成功；ABORTED/REJECTED/PREEMPTED 都会被驱动静默返回，
        # 必须显式判断并打印，否则会误报“已到达”。
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[xg_arm_waypoints] 动作成功 (SUCCEEDED)")
        else:
            rospy.logwarn("[xg_arm_waypoints] 动作未成功，state=%s (%s)",
                          state, actionlib.GoalStatus.to_string(state))
        return state, result

    @staticmethod
    def _euler_deg_to_quat(tx_deg, ty_deg, tz_deg):
        """欧拉角 XYZ（度） -> 四元数（与 kinova demo 的 EulerXYZ2Quaternion 一致）。"""
        tx, ty, tz = math.radians(tx_deg), math.radians(ty_deg), math.radians(tz_deg)
        sx, cx = math.sin(0.5 * tx), math.cos(0.5 * tx)
        sy, cy = math.sin(0.5 * ty), math.cos(0.5 * ty)
        sz, cz = math.sin(0.5 * tz), math.cos(0.5 * tz)
        qx = sx * cy * cz + cx * sy * sz
        qy = -sx * cy * sz + cx * sy * cz
        qz = sx * sy * cz + cx * cy * sz
        qw = -sx * sy * sz + cx * cy * cz
        return qx, qy, qz, qw


class WaypointArm:
    """依次经过多个路点，最终到达终点。"""

    def __init__(self):
        rospy.loginfo("[xg_arm_waypoints] 初始化 WaypointArm ...")
        self.arm = KinovaArm()

    def go_home(self):
        rospy.loginfo("[xg_arm_waypoints] 回 home 位 ...")
        return self.move_pose(HOME_POSE)

    def move_pose(self, pose6):
        """发送一个基座坐标系的完整 6 自由度位姿，返回是否成功。"""
        state, _ = self.arm.arm_run(list(pose6))
        return state == actionlib.GoalStatus.SUCCEEDED

    def run_waypoints(self, waypoints):
        """
        依次经过 waypoints 里的所有路点。
        每个路点到达(SUCCEEDED)后才走下一个；中途失败则中止，返回 False。
        全部到达返回 True。
        """
        total = len(waypoints)
        for i, wp in enumerate(waypoints, 1):
            tag = "终点" if i == total else "途经点 %d" % i
            rospy.loginfo("[xg_arm_waypoints] 前往 %s (%d/%d): %s", tag, i, total, list(wp))
            if not self.move_pose(wp):
                rospy.logerr("[xg_arm_waypoints] %s 未到达，中止后续移动", tag)
                return False
        rospy.loginfo("[xg_arm_waypoints] 已依次经过所有路点，到达终点")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="依次经过多个路点（6 自由度），最终到达终点（自包含，无 YOLO）")
    parser.add_argument('--pose', nargs=6, type=float, metavar=('X', 'Y', 'Z', 'TX', 'TY', 'TZ'),
                        help="只走一个基座坐标系的完整 6 自由度位姿(米 + 度)，执行一次后退出")
    parser.add_argument('--home', action='store_true',
                        help="仅回 home 位后退出")
    args = parser.parse_args()

    rospy.init_node('waypoint_arm', anonymous=True)
    arm = WaypointArm()

    if args.home:
        arm.go_home()
        rospy.loginfo("[xg_arm_waypoints] 已回 home 位")
        return

    if args.pose:
        rospy.loginfo("[xg_arm_waypoints] 单点调试模式: %s", args.pose)
        arm.move_pose(args.pose)
        return

    # 默认：依次经过文件顶部写死的三个路点
    rospy.loginfo("[xg_arm_waypoints] 开始依次走 %d 个路点: %s", len(WAYPOINTS), WAYPOINTS)
    arm.run_waypoints(WAYPOINTS)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
