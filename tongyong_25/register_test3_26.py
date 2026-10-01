#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单独测试 jujia26.py 中三位主人的注册、姓名采集与人脸识别。"""

from pathlib import Path

import rospy

from jujia26 import (
    Controller,
    Detector,
    target_name,
)
from speech_2026 import CompetitionVoiceService
from summer_tts_speaker import SummerTTSSpeaker


REGISTER_COUNT = 3
RECOGNITION_MAX_ATTEMPTS = 3


def create_register_controller():
    """只初始化三人注册与人脸识别实际依赖的模块。"""
    controller = Controller.__new__(Controller)

    project_dir = Path(__file__).resolve().parent
    controller.photo_path = str(project_dir / "face")
    controller.face = Detector(controller.photo_path)

    controller.speak = SummerTTSSpeaker()
    controller.voice = CompetitionVoiceService(
        owner_names=target_name,
        speaker=controller.speak,
    )

    controller.person_info = {}
    return controller


def speak_and_wait(controller, text):
    """使用新版语音服务播报，并等待播报结束。"""
    controller.voice.say(text, wait=True)


def print_existing_face_warning(photo_path):
    """提示测试目录中已有的人脸ID，但不自动删除任何数据。"""
    face_root = Path(photo_path)
    existing_ids = sorted(
        int(path.name)
        for path in face_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )

    if existing_ids:
        print("警告：人脸目录中已经存在注册数据")
        print(f"现有人脸ID：{existing_ids}")
        print("若测试者已经注册，重复人脸检查可能拒绝再次注册")


def register_owner(controller, owner_index):
    """使用新版语音服务完成一位主人的人脸和姓名注册。"""
    person_id = None
    max_face_attempts = 3

    speak_and_wait(
        controller,
        f"请主人{owner_index}站在我面前，保持静止",
    )

    for face_attempt in range(1, max_face_attempts + 1):
        if rospy.is_shutdown():
            return None

        speak_and_wait(controller, "开始人脸注册，请看向我")
        print(
            f">>> 正在进行人脸注册..."
            f"（第{face_attempt}/{max_face_attempts}次）"
        )

        try:
            person_id = controller.face.register_new_face()
        except Exception as error:
            print(f"人脸注册发生异常：{error}")
            person_id = None
        finally:
            try:
                controller.face.close_k4a()
            except Exception as close_error:
                print(f"关闭人脸相机发生异常：{close_error}")

        if person_id is not None:
            break

        print("未检测到有效人脸")
        if face_attempt < max_face_attempts:
            speak_and_wait(controller, "注册失败，请再试一次")

    if person_id is None:
        if rospy.is_shutdown():
            print("ROS 已关闭，人脸注册终止")
        else:
            print(
                "人脸检测已达到最大尝试次数"
                f"（{max_face_attempts}次）"
            )
            speak_and_wait(controller, "人脸注册失败")
        return None

    print(f"人脸注册成功，ID：{person_id}")
    speak_and_wait(controller, "人脸注册成功")

    print(">>> [阶段2] 正在采集姓名...")
    try:
        person_name, name_score, raw_name = (
            controller.voice.ask_owner_name(
                retries=3,
                duration=5.0,
            )
        )
    except Exception as error:
        print(f"姓名语音识别发生异常：{error}")
        person_name = None
        name_score = 0.0
        raw_name = ""

    print(f"姓名识别原文：{raw_name}")
    print(f"姓名匹配结果：{person_name}")
    print(f"姓名匹配置信度：{name_score:.2%}")

    if person_name is None:
        print("姓名语音识别已达到最大尝试次数（3次）")
        speak_and_wait(controller, "姓名识别失败")
        return None

    controller.person_info[person_id] = {
        "owner_index": owner_index,
        "name": person_name,
    }
    register_result = {
        "person_id": person_id,
        "owner_index": owner_index,
        "person_name": person_name,
    }

    print("--------------------------------")
    print("主人注册完成")
    print(f"主人顺序：{owner_index}")
    print(f"人脸ID：{person_id}")
    print(f"主人姓名：{person_name}")
    print("--------------------------------")
    controller.voice.announce_owner_registered(
        person_name,
        owner_index,
    )
    return register_result


def register_three_owners(controller):
    """使用新版语音处理接口依次注册三位主人。"""
    register_results = []

    for owner_index in range(1, REGISTER_COUNT + 1):
        if rospy.is_shutdown():
            break

        print("==================================================")
        print(f"开始注册主人{owner_index}/{REGISTER_COUNT}")
        print("==================================================")

        try:
            result = register_owner(controller, owner_index)
        except Exception as error:
            print(f"主人{owner_index}注册发生异常：{error}")
            result = None

        if result is None:
            print(f"主人{owner_index}注册失败")
        else:
            register_results.append(result)
            print(
                f"主人{owner_index}注册成功："
                f"人脸ID={result['person_id']}，"
                f"姓名={result['person_name']}"
            )

        if owner_index < REGISTER_COUNT and not rospy.is_shutdown():
            speak_and_wait(
                controller,
                "当前主人注册结束，请下一位主人做好准备",
            )

    return register_results


def recognize_one_owner(controller, expected_result):
    """识别指定主人，最多尝试三次，并核对人脸ID和姓名。"""
    expected_face_id = expected_result["person_id"]
    expected_owner_index = expected_result["owner_index"]
    expected_name = expected_result["person_name"]

    speak_and_wait(
        controller,
        f"请主人{expected_owner_index}，{expected_name}站在我面前看向我",
    )

    for attempt in range(1, RECOGNITION_MAX_ATTEMPTS + 1):
        if rospy.is_shutdown():
            return None

        print(
            f">>> 正在识别主人{expected_owner_index}..."
            f"（第{attempt}/{RECOGNITION_MAX_ATTEMPTS}次）"
        )

        # 直接复用 jujia26.Controller.recognize_owner()，
        # 识别结果会根据 controller.person_info 还原主人编号和姓名。
        recognized_result = controller.recognize_owner()

        if recognized_result is None:
            print("本次没有识别到已注册主人")
        elif recognized_result["face_id"] != expected_face_id:
            print("本次识别到了其他已注册主人")
            print(f"期望人脸ID：{expected_face_id}")
            print(f"实际人脸ID：{recognized_result['face_id']}")
        elif recognized_result["person_name"] != expected_name:
            print("人脸ID正确，但绑定姓名与注册结果不一致")
            print(f"期望姓名：{expected_name}")
            print(f"实际姓名：{recognized_result['person_name']}")
        else:
            print("主人识别成功")
            print(f"主人编号：{recognized_result['owner_index']}")
            print(f"人脸ID：{recognized_result['face_id']}")
            print(f"主人姓名：{recognized_result['person_name']}")
            controller.voice.announce_owner_recognized(
                recognized_result["person_name"],
                recognized_result["owner_index"],
            )
            return recognized_result

        if attempt < RECOGNITION_MAX_ATTEMPTS:
            speak_and_wait(
                controller,
                "没有正确识别，请调整位置并再次看向我",
            )

    print(
        f"主人{expected_owner_index}已达到最大识别次数，"
        "本轮识别失败"
    )
    return None


def recognize_three_owners(controller, register_results):
    """让成功注册的主人依次进行人脸识别验证。"""
    recognition_results = []

    print("==================================================")
    print("三位主人注册阶段结束，开始依次验证人脸识别")
    print("==================================================")
    speak_and_wait(
        controller,
        "注册阶段结束，开始验证三位主人的人脸识别",
    )

    for index, expected_result in enumerate(register_results):
        if rospy.is_shutdown():
            break

        recognized_result = recognize_one_owner(
            controller,
            expected_result,
        )
        if recognized_result is not None:
            recognition_results.append(recognized_result)

        if index < len(register_results) - 1 and not rospy.is_shutdown():
            speak_and_wait(
                controller,
                "本轮识别结束，请当前主人离开，下一位主人做好准备",
            )

    return recognition_results


def print_test_summary(register_results, recognition_results, person_info):
    """打印三人注册与识别测试汇总。"""
    print("==================================================")
    print("三人注册与识别测试汇总")
    print("==================================================")
    print(f"计划注册人数：{REGISTER_COUNT}")
    print(f"成功注册人数：{len(register_results)}")
    print(f"成功识别人数：{len(recognition_results)}")
    print(f"主人信息表：{person_info}")

    for result in register_results:
        recognition_passed = any(
            recognized["face_id"] == result["person_id"]
            for recognized in recognition_results
        )
        status = "识别通过" if recognition_passed else "识别未通过"
        print(
            f"主人{result['owner_index']}："
            f"人脸ID={result['person_id']}，"
            f"姓名={result['person_name']}，"
            f"{status}"
        )


def main():
    rospy.init_node("register_test3_26", anonymous=True)
    controller = create_register_controller()

    print("==============开始三人注册与识别测试==============")
    print(f"人脸保存目录：{controller.photo_path}")
    print(f"主人姓名候选：{controller.voice.owner_names}")
    print(f"中文语音模型：{controller.voice.recognizer.model_path}")
    print_existing_face_warning(controller.photo_path)

    try:
        register_results = register_three_owners(controller)

        if not register_results:
            print("没有主人注册成功，无法进行人脸识别测试")
            return

        recognition_results = recognize_three_owners(
            controller,
            register_results,
        )
        print_test_summary(
            register_results,
            recognition_results,
            controller.person_info,
        )
    finally:
        try:
            controller.face.close_k4a()
        except Exception as error:
            print(f"关闭人脸相机发生异常：{error}")
        print("==============三人注册与识别测试结束==============")


if __name__ == "__main__":
    main()
