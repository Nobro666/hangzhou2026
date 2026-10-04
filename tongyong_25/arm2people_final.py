#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把物体递给人。主流程里直接调用，不跑本文件的入口。

    import arm2people_final
    kinova.arm2people()

打点格式（7 个数）：[x, y, z, qx, qy, qz, qw]，位置米，朝向四元数。
默认先原地调到目标高度，再平移到目标 XY。
"""

from __future__ import print_function

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROBOT_SRC = "/home/zq/catkin_ws/src/cmoon/src"
if ROBOT_SRC not in sys.path:
    sys.path.append(ROBOT_SRC)

import rospy

from catch_ground.src.catch import KinovaRobot


# ---------------------------------------------------------------------------
# 打点位姿：直接粘贴 [x, y, z, qx, qy, qz, qw]，不要转欧拉角
# ---------------------------------------------------------------------------
x = 0.5992317795753479
y = -0.2664104402065277
z = 0.4930836856365204
qx = 0.49514734745025635
qy = 0.4847258925437927
qz = 0.5053961873054504
qw = 0.5142416954040527

TARGET_POSE = [x, y, z, qx, qy, qz, qw]

# True：先原地改高度，再平移到目标 XY（四元数用目标）
# False：一次直达
USE_SAFE_PATH = True


def move_to_pose_mq(kinova, pose, use_safe_path=True):
    """
    按 catch.py 的 mq 单位调用 arm_run。
    pose: [x, y, z, qx, qy, qz, qw]
    """
    if len(pose) != 7:
        raise ValueError("打点位姿必须是 7 个数: [x,y,z,qx,qy,qz,qw]，当前长度={}".format(len(pose)))

    target = [float(v) for v in pose]
    print("[Arm2People] 目标(mq) = {}".format(target))

    if not use_safe_path:
        print("[Arm2People] 直接 arm_run(unit='mq')")
        kinova.arm_run(unit="mq", pose_target=target)
        print("Cartesian pose sent!")
        return True

    kinova.getcurrentCartesianCommand()
    cur = kinova.currentCartesianCommand
    cx, cy, cz = float(cur[0]), float(cur[1]), float(cur[2])
    print("[Arm2People] 当前末端位置 ({:.3f},{:.3f},{:.3f})".format(cx, cy, cz))

    # 1) 保持当前 XY，高度与朝向用打点值
    lift = [cx, cy, target[2], target[3], target[4], target[5], target[6]]
    print("[Arm2People] 1.原地调高度 arm_run(mq) -> {}".format(lift))
    kinova.arm_run(unit="mq", pose_target=lift)

    # 2) 到目标 XY
    print("[Arm2People] 2.平移到目标 arm_run(mq) -> {}".format(target))
    kinova.arm_run(unit="mq", pose_target=target)
    return True


def arm2people(self, use_safe_path=None):
    """主流程调用 kinova.arm2people()。默认先调高度再平移。"""
    if use_safe_path is None:
        use_safe_path = USE_SAFE_PATH
    return move_to_pose_mq(self, TARGET_POSE, use_safe_path=use_safe_path)


KinovaRobot.arm2people = arm2people


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="臂递到人（打点 mq）")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="一次直达目标（默认先调高度再平移）",
    )
    opt = parser.parse_args()

    rospy.init_node("arm2people_final", anonymous=True)
    kinova = KinovaRobot("j2n6s300")
    if not hasattr(kinova, "client_arm") or kinova.client_arm is None:
        print("机械臂未连接，请先启动 kinova_bringup")
    else:
        ok = kinova.arm2people(use_safe_path=not opt.direct)
        print("完成" if ok else "失败")
