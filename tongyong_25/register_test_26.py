#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""单独测试一位主人的人脸注册与匹配流程，不采集姓名。"""

from pathlib import Path

import rospy

from jujia26 import (
    Controller,
    Detector,
    SummerTTSSpeaker,
)


def create_register_controller():
    """只初始化人脸注册、匹配和提示语音依赖的模块。"""
    controller = Controller.__new__(Controller)

    project_dir = Path(__file__).resolve().parent
    controller.photo_path = str(project_dir / "face")
    controller.face = Detector(controller.photo_path)

    controller.speak = SummerTTSSpeaker()
    return controller


def speak_and_wait(controller, text, wait_seconds):
    """发送非阻塞语音后留出合成和播放时间。"""
    controller.speak.speak(text)
    rospy.sleep(wait_seconds)


def register_face_only(controller, max_attempts=3):
    """沿用 jujia26.register() 的人脸采集逻辑，但跳过姓名采集。"""
    person_id = None
    face_attempts = 0

    speak_and_wait(controller, "请站在我面前，保持静止", 2.0)

    while (
        person_id is None
        and face_attempts < max_attempts
        and not rospy.is_shutdown()
    ):
        face_attempts += 1
        speak_and_wait(controller, "开始人脸注册，请看向我", 1.5)
        print(
            f">>> 正在进行人脸注册..."
            f"（第{face_attempts}/{max_attempts}次）"
        )

        try:
            person_id = controller.face.register_new_face()
        except Exception as error:
            print(f"人脸注册发生异常: {error}")
            person_id = None
        finally:
            try:
                controller.face.close_k4a()
            except Exception as close_error:
                print(f"关闭人脸相机发生异常: {close_error}")

        if person_id is None:
            print("未检测到有效人脸")
            if face_attempts < max_attempts:
                speak_and_wait(controller, "注册失败，请再试一次", 1.5)

    if person_id is None:
        if rospy.is_shutdown():
            print("ROS 已关闭，人脸注册终止")
        else:
            print(f"人脸检测已达到最大尝试次数（{max_attempts}次）")
            speak_and_wait(controller, "人脸注册失败", 1.5)
        return None

    print(f"人脸注册成功，ID: {person_id}")
    speak_and_wait(controller, "人脸注册成功", 1.5)
    return person_id


def recognize_registered_face(controller, registered_face_id):
    """使用与 jujia26.recognize_owner() 相同的接口验证人脸 ID。"""
    detected_face_id = None
    try:
        controller.face.detect_known_faces()
        detected_face_id = controller.face.detect_result
    except Exception as error:
        print(f"人脸识别发生异常：{error}")
    finally:
        try:
            controller.face.close_k4a()
        except Exception:
            pass

    if detected_face_id is None:
        print("人脸匹配失败：未识别到已注册人脸")
        return False

    print(f"刚注册的人脸ID：{registered_face_id}")
    print(f"本次识别的人脸ID：{detected_face_id}")

    if detected_face_id != registered_face_id:
        print("人脸匹配失败：两次人脸 ID 不一致")
        return False

    print("人脸匹配成功：识别结果与刚注册的人脸一致")
    return True


def main():
    rospy.init_node("register_test_26", anonymous=True)
    controller = create_register_controller()

    print("==============开始单人人脸注册测试==============")
    print(f"人脸保存目录：{controller.photo_path}")

    try:
        # 仅保留 jujia26.register() 中的人脸采集部分，不询问姓名。
        registered_face_id = register_face_only(controller)

        if registered_face_id is None:
            print("人脸注册测试失败")
            return

        print("==============开始人脸匹配测试==============")
        speak_and_wait(
            controller,
            "请刚刚注册的人站在我面前，看向我",
            2.0,
        )

        if recognize_registered_face(controller, registered_face_id):
            speak_and_wait(controller, "人脸匹配成功", 1.5)
        else:
            speak_and_wait(controller, "人脸匹配失败", 1.5)
    finally:
        try:
            controller.face.close_k4a()
        except Exception as error:
            print(f"关闭人脸相机发生异常：{error}")
        print("==============单人人脸注册测试结束==============")


if __name__ == "__main__":
    main()
