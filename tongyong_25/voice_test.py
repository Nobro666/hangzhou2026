#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_test.py

2026 居家生活/具身服务语音全流程交互式 Demo。

目标：
- 真实语音播报；
- 真实语音识别；
- 外部视觉/行为/物品模型尚未接入时，通过语音询问人工输入模拟结果；
- 每一步都遵循“播报提示 -> 等待说话/录音识别 -> 解析/播报结果 -> 再进入下一步”。

运行前要求：
1. ROS 环境正确，/summer_tts_topic 有订阅者；
2. 麦克风可用；
3. vosk、pyaudio、中文 Vosk 模型可用；
4. 与 speech_2026.py、summer_tts_speaker.py 放在同一目录。

运行：
    python3 voice_test.py
"""

import time
from typing import Optional

import rospy

from summer_tts_speaker import SummerTTSSpeaker
from speech_2026 import CompetitionVoiceService, DEFAULT_OWNER_NAMES


# ===== 2026-09-30 现场/调试可修改配置 START =====
# 现场确认真实主人姓名后，改这里；也要同步到 jujia26.py 的 target_name。
TEST_OWNER_NAMES = DEFAULT_OWNER_NAMES.copy()

# 录音时长。若现场说话慢或环境嘈杂，可适当调大。
NAME_RECORD_SECONDS = 5
BEHAVIOR_RECORD_SECONDS = 6
SWITCH_RECORD_SECONDS = 7
SWITCH_CONFIRM_SECONDS = 3
HELP_RECORD_SECONDS = 7
TRASH_RECORD_SECONDS = 5
OBJECT_RECORD_SECONDS = 5

# 模拟识别几件垃圾/物品；只是测试轮数，可按需改成 1~3。
TRASH_TEST_COUNT = 3
OBJECT_TEST_COUNT = 3
# ===== 2026-09-30 现场/调试可修改配置 END =====


BEHAVIOR_KEYWORDS = {
    "躺下睡觉": ["躺下", "睡觉", "躺", "睡"],
    "坐在床上休息": ["坐在床", "床上", "坐床", "坐在床上"],
    "坐在沙发或椅子休息": ["沙发", "椅子", "坐下", "休息", "坐着"],
    "摔倒": ["摔倒", "倒地", "跌倒", "倒在地上"],
    "挥手": ["挥手", "招手", "示意"],
}


def print_title(title: str):
    print("\n" + "=" * 24 + f" {title} " + "=" * 24)


def ensure_tts_connected(speaker: SummerTTSSpeaker):
    """提示 TTS topic 连接状态，不强制退出。"""
    connected = speaker.is_connected()
    if connected:
        print("[TTS检查] /summer_tts_topic 已检测到订阅者，可以真实播报。")
        return True

    print("\n[TTS警告] /summer_tts_topic 当前没有订阅者，播报文本可能不会出声。")
    print("请检查：")
    print("  rostopic info /summer_tts_topic")
    print("  rosnode list | grep -i tts")
    print("  rostopic pub /summer_tts_topic std_msgs/String \"data: '测试语音'\" -1")
    print("如果上述 rostopic pub 能出声，再重新运行本文件。\n")


def listen_text(voice: CompetitionVoiceService, prompt: str, duration: float, retries: int = 3) -> str:
    """
    通用交互：播报提示 -> 录音识别 -> 返回原始文本。
    用于模拟尚未接入的视觉/行为/物品模型输入。
    """
    last_text = ""
    for attempt in range(1, retries + 1):
        # 这里必须等待机器人把提示语说完再开始录音；
        # 否则会录到机器人自己的播报，表现为还没等主人回答就进入下一步。
        voice.say(prompt, wait=True)
        print(f"[等待语音输入] 第 {attempt}/{retries} 次，录音 {duration} 秒：{prompt}")
        # ===== 2026-09-30 自由文本识别修正 START =====
        # 这里用于模拟“视觉/行为/物品模型结果”，人可能说任意物品名/垃圾名。
        # 如果继续使用比赛 grammar，Vosk 会把静音或未知词硬匹配成词表里的“开关/垃圾/物品”等怪结果。
        # 因此此处临时关闭 grammar，做自由中文识别；识别完成后恢复原 grammar。
        old_grammar = list(voice.recognizer.grammar) if voice.recognizer.grammar else None
        voice.recognizer.reset_grammar(None)
        try:
            result = voice.recognizer.listen_once(duration=duration)
        finally:
            voice.recognizer.reset_grammar(old_grammar)
        # ===== 2026-09-30 自由文本识别修正 END =====
        text = (result.text or "").strip()
        print(f"[识别原文] {text}")
        if text:
            return text
        last_text = text
        if attempt < retries:
            voice.say("没有听清，请再说一遍", wait=True)
    return last_text


def parse_behavior(text: str) -> str:
    """把人工口述的行为文本归一成赛规行为名称。"""
    normalized = (text or "").replace(" ", "")
    for behavior, words in BEHAVIOR_KEYWORDS.items():
        if any(word in normalized for word in words):
            return behavior
    return text or "未知行为"


def is_rest_behavior(behavior: str) -> bool:
    return any(key in behavior for key in ["躺", "睡", "坐", "休息", "床", "沙发", "椅子"])


def is_wave_behavior(behavior: str) -> bool:
    return "挥手" in behavior or "招手" in behavior


def is_fall_behavior(behavior: str) -> bool:
    return "摔倒" in behavior or "倒地" in behavior


def main():
    rospy.init_node("voice_test_2026", anonymous=True)

    print_title("0. 初始化真实播报与真实语音识别")
    speaker = SummerTTSSpeaker()
    if not ensure_tts_connected(speaker):
        print("[TTS错误] 未连接到 /summer_tts_topic 订阅者，停止测试，避免无播报时继续录音。")
        return

    voice = CompetitionVoiceService(
        owner_names=TEST_OWNER_NAMES,
        speaker=speaker,
    )

    print("当前主人姓名候选：", TEST_OWNER_NAMES)
    print("说明：外部视觉/行为/物品模型未接入时，本测试会通过语音询问人工输入模拟结果。")
    voice.say("语音测试开始", wait=True)
    time.sleep(1)

    print_title("1. 注册人脸：模拟人脸注册成功，真实询问主人姓名")
    voice.say("请主人到我面前", wait=True)
    voice.say("请看向我", wait=True)
    simulated_face_id = 101
    owner_index = 1
    print(f"[模拟外部输入] 人脸注册成功，face_id={simulated_face_id}")
    voice.say("人脸注册成功", wait=True)

    person_name, name_score, raw_name = voice.ask_owner_name(
        retries=3,
        duration=NAME_RECORD_SECONDS,
    )
    if not person_name:
        person_name = "主人"
        voice.say("未识别到候选姓名，后续测试使用主人称呼", wait=True)

    person_info = {
        simulated_face_id: {
            "owner_index": owner_index,
            "name": person_name,
        }
    }
    print("姓名识别原文：", raw_name)
    print("姓名匹配结果：", person_name)
    print("姓名匹配置信度：", name_score)
    print("[模拟记录 person_info]", person_info)
    voice.announce_owner_registered(person_name, owner_index)

    print_title("2. 识别人脸：模拟识别到已注册主人，并播报姓名")
    recognized_face_id = simulated_face_id
    recognized_info = person_info[recognized_face_id]
    print(f"[模拟外部输入] 识别到 face_id={recognized_face_id}")
    voice.announce_owner_recognized(
        recognized_info["name"],
        recognized_info["owner_index"],
    )

    print_title("3. 姿态/行为：模型未接入，询问人工口述行为并播报")
    behavior_text = listen_text(
        voice,
        "请说主人行为",
        duration=BEHAVIOR_RECORD_SECONDS,
        retries=3,
    )
    behavior = parse_behavior(behavior_text)
    print(f"[行为解析结果] 原文={behavior_text}, 行为={behavior}")
    voice.announce_behavior(behavior, person_name)

    print_title("4. 根据行为进入对应语音交互")
    if is_rest_behavior(behavior) and not is_wave_behavior(behavior) and not is_fall_behavior(behavior):
        print("[分支] 坐/躺/休息：进入开关需求询问与确认")
        switch_cmd, confirmed = voice.ask_switch_command_confirmed(
            person_name=person_name,
            retries=3,
            duration=SWITCH_RECORD_SECONDS,
            confirm_retries=2,
            confirm_duration=SWITCH_CONFIRM_SECONDS,
        )
        print("[开关需求解析结果]")
        print("  raw_text:", switch_cmd.raw_text)
        print("  action:", switch_cmd.action)
        print("  color:", switch_cmd.color)
        print("  number:", switch_cmd.number)
        print("  target:", switch_cmd.target)
        print("  confidence:", switch_cmd.confidence)
        print("  is_valid:", switch_cmd.is_valid)
        print("  confirmed:", confirmed)
    elif is_wave_behavior(behavior):
        print("[分支] 挥手：进入帮助需求咨询")
        help_request = voice.ask_help_request(
            person_name=person_name,
            retries=3,
            duration=HELP_RECORD_SECONDS,
        )
        print("[帮助需求解析结果]")
        print("  raw_text:", help_request.raw_text)
        print("  request_text:", help_request.request_text)
        print("  intent:", help_request.intent)
        print("  confidence:", help_request.confidence)
    elif is_fall_behavior(behavior):
        print("[分支] 摔倒：播报帮助，机械臂/人体定位尚未接入")
        voice.say(f"{person_name}，请保持不动，我来帮助你", wait=True)
    else:
        print("[分支] 未知行为：不进入开关或帮助需求分支")
        voice.say("暂时没有识别出可执行的行为需求", wait=True)

    print_title("5. 找垃圾：模型未接入，询问人工口述垃圾名称并播报")
    for idx in range(1, TRASH_TEST_COUNT + 1):
        trash_name = listen_text(
            voice,
            f"请说第{idx}个垃圾名称",
            duration=TRASH_RECORD_SECONDS,
            retries=2,
        )
        trash_name = trash_name or "垃圾"
        print(f"[垃圾模拟输入] 第{idx}个：{trash_name}")
        voice.announce_trash_found(trash_name, "地上")

    print_title("6. 具身服务/智能赛项：模型未接入，询问人工口述物品名称并播报")
    for idx in range(1, OBJECT_TEST_COUNT + 1):
        object_name = listen_text(
            voice,
            f"请说第{idx}个物品名称",
            duration=OBJECT_RECORD_SECONDS,
            retries=2,
        )
        object_name = object_name or "未知物品"
        print(f"[物品模拟输入] 第{idx}个：{object_name}")
        voice.announce_object_recognized(object_name, f"第{idx}个目标物品")

    print_title("测试结束")
    voice.say("语音测试结束", wait=True)
    print("voice_test.py 已完成：真实语音播报 + 真实语音识别 + 人工口述模拟外部模型输入。")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
