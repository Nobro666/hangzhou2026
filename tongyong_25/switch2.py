#!/usr/bin/env python3
# coding: UTF-8

"""单按钮开关的机械臂轨迹：home → 目标点 → home。"""

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

HOME_POSE = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]

SWITCH_TARGET = [0.495388, 0.024662, 0.877768, 74.187, 85.102, 42.378]  # TODO 核对目标点是否能实际触碰按钮

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
            '/' + p + 'driver/fingers_action/finger_positions',
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


def _go(target):
    """home → 终点。任一步失败返回 False。"""
    arm = _get_arm()
    return arm.arm_run(HOME_POSE) and arm.arm_run(target)


def operate_switch():
    """沿原有单侧轨迹移动、停留并回到home。"""
    arm = _get_arm()
    if not _go(SWITCH_TARGET):
        rospy.logerr("[arm] 到达开关目标位置失败")
        return False
    rospy.loginfo("[arm] 到达开关目标位置，停留 %.1fs", STAY_SEC)
    time.sleep(STAY_SEC)
    return arm.arm_run(HOME_POSE)


# =============================================================================
# 命令行
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--press', action='store_true')
    args = parser.parse_args()

    rospy.init_node('waypoint_arm', anonymous=True)

    if args.press:
        operate_switch()
    else:
        rospy.loginfo("用法: --press")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
