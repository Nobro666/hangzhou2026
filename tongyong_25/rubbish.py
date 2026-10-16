#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单独测试垃圾识别、导航靠近和机械臂抓取。"""

import argparse
import os
import sys

import cv2
import rospy

sys.path.append("/home/zq/catkin_ws/src/cmoon/src")

from camera_to_map import CoordinateConverter
from catch_ground.src.catch import KinovaRobot
from catch_ground.src.detector_items_c import ItemsDetector
from catch_ty import KinovaRobotGroud
from detect_people import KinectCamera
from navigator import Navigator
from position_last_second import SmartGoalFinder


DEFAULT_MODEL_PATH = (
    "/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/"
    "tongyong_25/model/best(9).pt"
)
TARGET_ITEMS = [
    "potatochips", "cookies", "icedtea", "cola",
    "shampoo", "apple", "banana", "paperball",
    "box_white_small", "box_white_large",
]
TRASH_MAP_MIN_HEIGHT = -0.15
TRASH_MAP_MAX_HEIGHT = 0.45


def parse_args():
    parser = argparse.ArgumentParser(
        description="测试K4A垃圾粗定位、导航靠近和RealSense机械臂抓取"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("TRASH_MODEL_PATH", DEFAULT_MODEL_PATH),
        help="垃圾YOLO模型路径",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_ITEMS,
        help="只测试指定类别；默认检测前八类及索引19、20的纸团",
    )
    parser.add_argument("--detect-timeout", type=float, default=8.0)
    parser.add_argument("--max-distance", type=float, default=5.0)
    parser.add_argument("--goal-max-radius", type=float, default=0.9)
    parser.add_argument("--goal-min-radius", type=float, default=0.8)
    return parser.parse_args()


def detect_trash(camera, detector, converter, targets, args):
    """使用K4A完成一次垃圾识别，并转换到map坐标系。"""
    print("\n[阶段1] 打开K4A并识别垃圾")
    camera.open_camera()
    try:
        trash_name, camera_coords = detector.detect_targets(
            camera,
            target_items=targets,
            max_distance=args.max_distance,
            timeout=args.detect_timeout,
        )
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if trash_name is None:
        print("K4A未识别到目标垃圾，测试结束")
        return None

    print(f"K4A识别成功：类别={trash_name}，相机坐标={camera_coords}")
    map_coords = converter.get_map_coords(camera_coords)
    if map_coords is None:
        print("垃圾坐标转换失败，测试结束")
        return None

    height = map_coords[2]
    print(f"垃圾地图坐标={map_coords}")
    if not TRASH_MAP_MIN_HEIGHT <= height <= TRASH_MAP_MAX_HEIGHT:
        print(
            f"垃圾地图高度{height:.3f}m不在允许范围"
            f"[{TRASH_MAP_MIN_HEIGHT:.2f}, {TRASH_MAP_MAX_HEIGHT:.2f}]m内"
        )
        return None

    return trash_name, map_coords


def main():
    args = parse_args()
    targets = [args.target] if args.target else list(TARGET_ITEMS)

    rospy.init_node("rubbish_test_26", anonymous=True)
    print("==============垃圾抓取单项测试开始==============")
    print(f"模型：{args.model}")
    print(f"K4A检测类别：{targets}")

    converter = CoordinateConverter()
    goal_finder = SmartGoalFinder()
    location = {}
    navigator = Navigator(location)
    camera = KinectCamera()
    detector = ItemsDetector(model_path=args.model)

    detected = detect_trash(
        camera,
        detector,
        converter,
        targets,
        args,
    )
    if detected is None:
        return 1

    trash_name, map_coords = detected

    print("\n[阶段2] 计算垃圾附近的安全导航点")
    trash_goal = goal_finder.find_best_goal(
        map_coords,
        max_radius=args.goal_max_radius,
        min_radius=args.goal_min_radius,
    )
    if trash_goal is None:
        print("没有找到垃圾附近的安全导航点，测试结束")
        return 2

    location["current_trash"] = trash_goal
    print(f"安全导航点={trash_goal}")

    print("\n[阶段3] 导航到垃圾附近")
    if not navigator.goto("current_trash"):
        print("导航到垃圾附近失败，未启动机械臂抓取")
        return 3
    print("已到达垃圾附近")

    print("\n[阶段4] RealSense精定位并执行机械臂抓取")
    kinova = KinovaRobot("j2n6s300")
    try:
        grasp_result = KinovaRobotGroud.catch_ground(
            kinova,
            weights_path=args.model,
            target_items=[trash_name],
        )
    except (Exception, SystemExit) as error:
        rospy.logerr("垃圾抓取发生异常：%s", error)
        return 4

    if not grasp_result:
        print("RealSense未确认目标或机械臂抓取失败")
        return 5

    print(
        "垃圾抓取测试成功："
        f"类别={trash_name}，"
        f"RealSense结果={grasp_result}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        print("测试被ROS中断")
        sys.exit(130)
