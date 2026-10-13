#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单独测试：注册一位主人，等待后寻找并靠近，再识别身份和姿态。"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import rospy

sys.path.append("/home/zq/catkin_ws/src/cmoon/src")

from base_controller import Base
from camera_to_map import CoordinateConverter
from detect_people import KinectCamera, PersonDetector
from face_detect_tongyong import Detector
from jujia26 import ACTION_FALL, Controller
from navigator import Navigator
from position_last_second import SmartGoalFinder
from speech_2026 import CompetitionVoiceService
from summer_tts_speaker import SummerTTSSpeaker


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FACE_DIR = PROJECT_DIR / "face_owner_behavior_test"


def parse_args():
    parser = argparse.ArgumentParser(
        description="测试单人脸注册、人物搜索靠近、主人识别和姿态识别"
    )
    parser.add_argument(
        "--name",
        default="测试主人",
        help="本次注册人脸绑定的主人姓名",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=10.0,
        help="注册完成后等待多少秒再开始寻找人物，默认10秒",
    )
    parser.add_argument(
        "--face-dir",
        default=str(DEFAULT_FACE_DIR),
        help="测试人脸保存目录，默认与正式face目录隔离",
    )
    return parser.parse_args()


def close_face_camera(detector):
    """尽力释放人脸模块占用的K4A资源。"""
    try:
        detector.close_k4a()
    except Exception as error:
        print(f"关闭人脸相机时发生异常：{error}")
    cv2.destroyAllWindows()


def create_test_controller(face_dir, owner_name):
    """绕过Controller.__init__，只初始化本测试实际使用的成员。"""
    controller = Controller.__new__(Controller)
    controller.base = Base()
    controller.location = {}
    controller.navigator = Navigator(controller.location)
    controller.transpoint = CoordinateConverter()
    controller.goalpoint = SmartGoalFinder()
    controller.camera = KinectCamera()
    controller.people_detector = PersonDetector()
    controller.face = Detector(str(face_dir))
    controller.speak = SummerTTSSpeaker()
    controller.voice = CompetitionVoiceService(
        owner_names=[owner_name],
        speaker=controller.speak,
    )
    controller.person_info = {}
    controller.valid_person_map_coords = None
    return controller


def register_one_owner(controller, owner_name):
    """调用正式人脸模块注册一位主人，并建立测试用ID—姓名映射。"""
    print("\n[阶段1] 注册一位主人")
    print("请保持正脸朝向K4A相机")
    controller.voice.say("开始注册人脸，请保持正脸朝向相机", wait=True)
    try:
        person_id = controller.face.register_new_face()
    finally:
        close_face_camera(controller.face)

    if person_id is None:
        print("人脸注册失败：没有采集到有效人脸")
        return None

    controller.person_info[person_id] = {
        "owner_index": 1,
        "name": owner_name,
    }
    print(
        "人脸注册成功："
        f"ID={person_id}，主人编号=1，姓名={owner_name}"
    )
    controller.voice.say("人脸注册成功", wait=True)
    return person_id


def main():
    args = parse_args()
    face_dir = Path(args.face_dir).expanduser().resolve()
    face_dir.mkdir(parents=True, exist_ok=True)

    rospy.init_node("owner_behavior_test_26", anonymous=True)
    print("==============主人与姿态综合测试开始==============")
    print(f"测试人脸目录：{face_dir}")
    print("本脚本不会执行任何人机交互或机械臂动作")

    controller = create_test_controller(face_dir, args.name)
    registered_id = register_one_owner(controller, args.name)
    if registered_id is None:
        return 1

    wait_seconds = max(0.0, args.wait_seconds)
    print(
        f"\n[阶段2] 等待{wait_seconds:.1f}秒后开始寻找人物；"
        "请在此期间离开注册位置并到测试位置就位"
    )
    controller.voice.say(
        f"请在{wait_seconds:.0f}秒内到测试位置就位",
        wait=True,
    )
    time.sleep(wait_seconds)

    print("\n[阶段3] 分段旋转寻找人物并导航靠近")
    controller.voice.say("开始寻找人物", wait=False)
    person_result = controller.find_owner_in_room("test_room")
    if person_result is None:
        print("测试失败：没有找到比赛区域内的有效人物")
        controller.voice.say("没有找到人物", wait=True)
        return 2

    controller.voice.say("发现人物，准备靠近", wait=False)
    if not controller.approach_owner(person_result["map_coords"]):
        print("测试失败：人物已找到，但无法导航到人物附近")
        controller.voice.say("无法靠近人物", wait=True)
        return 3
    print("人物靠近完成")
    controller.voice.say("已到达人物附近", wait=True)

    print("\n[阶段4] 识别主人身份")
    owner_result = controller.recognize_owner()
    if owner_result is None:
        print("主人识别失败：没有匹配到本次注册的人脸")
        controller.voice.say("没有识别出主人", wait=True)
        face_id = None
        person_name = "未知人物"
    else:
        face_id = owner_result["face_id"]
        person_name = owner_result["person_name"]
        print(
            "主人识别成功："
            f"ID={face_id}，主人编号={owner_result['owner_index']}，"
            f"姓名={person_name}"
        )
        controller.voice.announce_owner_recognized(
            person_name,
            owner_result["owner_index"],
        )

    print("\n[阶段5] 识别人物姿态")
    behavior = controller.recognize_behavior(face_id, person_name)
    print(f"姿态识别结果：{behavior}")
    controller.voice.announce_behavior(behavior, person_name)

    print("\n[阶段6] 根据姿态决定是否执行第二次靠近")
    close_approach_success = None
    if behavior == ACTION_FALL:
        print("检测到摔倒，执行第二次靠近")
        controller.voice.say("检测到摔倒，准备进一步靠近人物", wait=False)
        close_approach_success = controller.approach_owner_closer(
            person_result["map_coords"]
        )
        if close_approach_success:
            print("第二次靠近完成")
            controller.voice.say("第二次靠近完成", wait=True)
        else:
            print("第二次靠近失败")
            controller.voice.say("第二次靠近失败", wait=True)
    else:
        print(f"当前姿态为{behavior}，不是摔倒，跳过第二次靠近")

    print("\n==============综合测试结果==============")
    print(f"注册人脸ID：{registered_id}")
    print(f"找到的人物地图坐标：{person_result['map_coords']}")
    print(f"主人身份：{person_name}")
    print(f"主人姿态：{behavior}")
    if close_approach_success is None:
        print("第二次靠近：未执行（只有摔倒才执行）")
    else:
        print(f"第二次靠近：{'成功' if close_approach_success else '失败'}")

    if owner_result is None:
        return 4
    if face_id != registered_id:
        print("测试失败：识别ID与本次注册ID不一致")
        return 5
    if close_approach_success is False:
        return 6

    print("测试通过：已识别并靠近人物，主人身份匹配成功，姿态识别已完成")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        print("测试被ROS中断")
        sys.exit(130)
    except Exception as error:
        print(f"测试过程发生异常：{error}")
        sys.exit(10)
