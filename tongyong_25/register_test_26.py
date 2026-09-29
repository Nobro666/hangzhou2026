#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单独测试 jujia26.py 中一位主人的人脸注册流程。"""

from pathlib import Path

import rospy

from jujia26 import (
    Controller,
    Detector,
    FuzzyKeywordMatcher,
    SummerTTSSpeaker,
    get_recognizer_instance,
    target_name,
)


def create_register_controller():
    """只初始化 Controller.register() 实际依赖的模块。"""
    controller = Controller.__new__(Controller)

    project_dir = Path(__file__).resolve().parent
    controller.photo_path = str(project_dir / "face")
    controller.face = Detector(controller.photo_path)

    controller.speak = SummerTTSSpeaker()
    get_recognizer_instance("zh")
    controller.name_matcher = FuzzyKeywordMatcher(keywords=target_name)

    controller.person_info = {}
    return controller


def main():
    rospy.init_node("register_test_26", anonymous=True)
    controller = create_register_controller()

    print("==============开始单人人脸注册测试==============")
    print(f"人脸保存目录：{controller.photo_path}")

    try:
        # 直接调用 jujia26.Controller.register()，保证注册逻辑完全一致。
        result = controller.register(1)

        if result is None:
            print("主人1注册失败")
            return

        print(
            "主人1注册成功："
            f"人脸ID={result['person_id']}，"
            f"姓名={result['person_name']}"
        )
        print(f"主人信息表：{controller.person_info}")
    finally:
        try:
            controller.face.close_k4a()
        except Exception as error:
            print(f"关闭人脸相机发生异常：{error}")
        print("==============单人人脸注册测试结束==============")


if __name__ == "__main__":
    main()
