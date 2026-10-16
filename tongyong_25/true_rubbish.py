#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在固定抓取站位测试真假物品识别、语音播报及真垃圾抓取。

运行前将机器人停在垃圾抓取站位，确认机械臂工作范围内没有人员。
默认执行抓取；传入 --detect-only 可只查看识别和播报，不初始化机械臂。
本脚本不导航至垃圾桶，也不自动松开夹爪。
"""

import argparse
import os
from pathlib import Path

import rospy

from catch_ground.src.catch import KinovaRobot
from catch_ty import KinovaRobotGroud
from jujia26 import (
    TRASH_OBJECT_LABELS,
    TRASH_TARGET_CLASSES,
    TRASH_TRUE_CLASS_IDS,
)
from realsense_yolo11 import RealSenseYolo11Detector
from summer_tts_speaker import SummerTTSSpeaker


DEFAULT_MODEL = Path(__file__).resolve().parent / "model" / "best(9).pt"


def format_scene(detections):
    """与主流程相同：按模型索引区分真假，合并不同颜色的同类物品。"""
    announcements = []
    announced = set()
    for item in detections:
        class_id = item["class_id"]
        object_label = TRASH_OBJECT_LABELS.get(class_id, item["name"])
        kind = "真垃圾" if class_id in TRASH_TRUE_CLASS_IDS else "假垃圾"
        key = (kind, object_label)
        if key not in announced:
            announcements.append(f"{kind}{object_label}")
            announced.add(key)
    return "；".join(announcements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("TRASH_MODEL_PATH", str(DEFAULT_MODEL))),
        help="默认使用本项目 model/best(9).pt",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="仅检测和播报，不初始化机械臂或执行抓取",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="仅在终端打印，不向语音节点发送播报",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"模型文件不存在：{args.model}")

    rospy.init_node("true_false_rubbish_test_26", anonymous=True)
    speaker = None if args.no_voice else SummerTTSSpeaker()
    print(f"模型：{args.model}")
    print("真垃圾索引：0～7、19、20；其余为假物品")

    def announce_scene(detections):
        if detections:
            for item in detections:
                print(
                    f"模型索引={item['class_id']}，英文类别={item['name']}，"
                    f"置信度={item['confidence']:.2f}"
                )
        else:
            print("未检测到模型中的物品")
        message = format_scene(detections)
        if not message:
            return
        print(f"播报内容：{message}")
        if speaker is not None:
            try:
                speaker.speak(message, wait_done=True, timeout=30.0)
            except TypeError:
                speaker.speak(message)

    if args.detect_only:
        detector = RealSenseYolo11Detector(weights=args.model)
        target = detector.detect_targets(
            target_items=TRASH_TARGET_CLASSES,
            report_all=True,
        )
        announce_scene(detector.last_detections)
        if target is None:
            print("没有检测到可抓取的真垃圾；机械臂未启动")
            return 1
        print(f"可抓取真垃圾：{target.name}；检测模式下不执行抓取")
        return 0

    print("即将初始化机械臂并移动到地面检测位，请确认工作范围安全")
    arm = KinovaRobot("j2n6s300")
    try:
        grasp_result = KinovaRobotGroud.catch_ground(
            arm,
            weights_path=args.model,
            target_items=TRASH_TARGET_CLASSES,
            on_scene_detected=announce_scene,
        )
    except (Exception, SystemExit) as error:
        rospy.logerr("真假垃圾测试发生异常：%s", error)
        return 2

    if not grasp_result:
        print("没有真垃圾或抓取失败；假物品未抓取")
        return 1
    print(f"真垃圾抓取成功：{grasp_result.name}")
    print("机械臂已回检测程序的 home 位，夹爪仍持物；本脚本不导航或投放")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except rospy.ROSInterruptException:
        raise SystemExit(130)
