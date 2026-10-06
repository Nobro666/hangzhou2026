#!/usr/bin/env python3
# coding: UTF-8

"""
xg_arm_waypoints.py —— 机械臂运动到"开/关"检测位置正上方并停留

对外接口：
    go_open_position()    # home → 开途经点 → 开终点 → 停2s → home
    go_close_position()   # home → 关途经点 → 关终点 → 停2s → home
    go_to(detect)         # 统一入口：detect='open' 或 'close'

说明：
    - "开"和"关"是两个检测位置，与夹爪无关
    - 夹爪全程闭合（初始化时合一次）
    - 每个动作走 4 个点：home → 途经点 → 终点 → home
    - kinova_robot.launch 只挂 server，不合爪、不回 home，故首次合爪必须自己做

运行：
    python3 xg_arm_waypoints.py --open
    python3 xg_arm_waypoints.py --close
"""

import time
import argparse

import math

import rospy
import actionlib

import std_msgs.msg
import geometry_msgs.msg
import kinova_msgs.msg


# =============================================================================
# 可调参数（现场标定，位姿 = [x, y, z, tx, ty, tz]，米 + 度）
# =============================================================================
KINOVA_TYPE = 'j2n6s300'
ARM_TIMEOUT_SEC = 60.0

FINGER_CLOSE = [85, 85, 85]    # 全程闭合

HOME_POSE = [0.324453, -0.419172, 0.447952, 90.136, 20.215, 2.928]  # TODO

OPEN_WAYPOINT = [0.360364, -0.070202, 0.629747, -82.299, 59.369, 125.176]   # TODO 开-途经点
OPEN_TARGET   = [-0.114738, -0.365686, 0.762665, 83.859, 28.168, -69.590]   # TODO 开-终点正上方

CLOSE_WAYPOINT = [0.360364, -0.070202, 0.629747, -82.299, 59.369, 125.176]  # TODO 关-途经点
CLOSE_TARGET   = [-0.114738, -0.365686, 0.762665, 83.859, 28.168, -69.590]  # TODO 关-终点正上方

STAY_SEC = 2.0                 # 终点停留时间


# =============================================================================
# Kinova 控制
# =============================================================================
class KinovaArm:
    def __init__(self, robot_type=KINOVA_TYPE):
        p = robot_type + '_'

        self.client = actionlib.SimpleActionClient(
            '/' + p + 'driver/pose_action/tool_pose', kinova_msgs.msg.ArmPoseAction)
        self.client.wait_for_server(rospy.Duration(5.0))
        self.goal = kinova_msgs.msg.ArmPoseGoal()
        self.goal.pose.header = std_msgs.msg.Header(frame_id=p + 'link_base')

        self.finger_client = actionlib.SimpleActionClient(
            '/' + p + 'driver/finger_action/gripper_command',
            kinova_msgs.msg.SetFingersPositionAction)
        self.finger_client.wait_for_server(rospy.Duration(5.0))

    def arm_run(self, pose):
        """[x,y,z,tx,ty,tz] 米+度。返回是否成功。"""
        qx, qy, qz, qw = euler_deg_to_quat(*pose[3:])
        self.goal.pose.pose.position = geometry_msgs.msg.Point(*pose[:3])
        self.goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(qx, qy, qz, qw)
        self.client.send_goal(self.goal)
        if not self.client.wait_for_result(rospy.Duration(ARM_TIMEOUT_SEC)):
            self.client.cancel_all_goals()
            rospy.logwarn("[arm] 超时")
            return False
        return self.client.get_state() == actionlib.GoalStatus.SUCCEEDED

    def finger_run(self, fingers):
        goal = kinova_msgs.msg.SetFingersPositionGoal()
        goal.fingers.finger1 = float(fingers[0])
        goal.fingers.finger2 = float(fingers[1])
        goal.fingers.finger3 = float(fingers[2])
        self.finger_client.send_goal(goal)
        self.finger_client.wait_for_result(rospy.Duration(10.0))


def euler_deg_to_quat(tx_deg, ty_deg, tz_deg):
    tx, ty, tz = math.radians(tx_deg), math.radians(ty_deg), math.radians(tz_deg)
    sx, cx = math.sin(0.5 * tx), math.cos(0.5 * tx)
    sy, cy = math.sin(0.5 * ty), math.cos(0.5 * ty)
    sz, cz = math.sin(0.5 * tz), math.cos(0.5 * tz)
    return (
        sx * cy * cz + cx * sy * sz,
        -sx * cy * sz + cx * sy * cz,
        sx * sy * cz + cx * cy * sz,
        -sx * sy * sz + cx * cy * cz,
    )


# =============================================================================
# 对外接口
# =============================================================================
_arm = None


def _get_arm():
    global _arm
    if _arm is None:
        _arm = KinovaArm()
        _arm.finger_run(FINGER_CLOSE)   # 全程闭合，只合这一次
    return _arm


def _go(waypoint, target):
    """home → 途经点 → 终点。任一步失败返回 False。"""
    arm = _get_arm()
    return arm.arm_run(HOME_POSE) and arm.arm_run(waypoint) and arm.arm_run(target)


def go_open_position():
    """home → 开途经点 → 开终点正上方 → 停2s → home。"""
    arm = _get_arm()
    if not _go(OPEN_WAYPOINT, OPEN_TARGET):
        rospy.logerr("[arm] 去开位置失败")
        return False
    rospy.loginfo("[arm] 到达开位置正上方，停留 %.1fs", STAY_SEC)
    time.sleep(STAY_SEC)
    arm.arm_run(HOME_POSE)
    return True


def go_close_position():
    """home → 关途经点 → 关终点正上方 → 停2s → home。"""
    arm = _get_arm()
    if not _go(CLOSE_WAYPOINT, CLOSE_TARGET):
        rospy.logerr("[arm] 去关位置失败")
        return False
    rospy.loginfo("[arm] 到达关位置正上方，停留 %.1fs", STAY_SEC)
    time.sleep(STAY_SEC)
    arm.arm_run(HOME_POSE)
    return True


def go_to(detect):
    """统一入口：detect='open' 或 'close'。"""
    return go_open_position() if detect == 'open' else go_close_position()


# =============================================================================
# 命令行
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--close', action='store_true')
    args = parser.parse_args()

    rospy.init_node('waypoint_arm', anonymous=True)

    if args.open:
        go_open_position()
    elif args.close:
        go_close_position()
    else:
        rospy.loginfo("用法: --open 或 --close")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass