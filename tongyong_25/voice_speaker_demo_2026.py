#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 居家生活语音识别 + 语音播报独立联调主文件

用途：
- 不启动居家生活完整流程，不依赖导航点 LOCATION；
- 只测试 speech_2026.py 与 summer_tts_speaker.py 的耦合；
- 测试内容包括：主人姓名识别、开关指令识别、挥手需求识别、确认/否认识别；
- 该文件中使用的 CompetitionVoiceService 与 jujia26.py 完全相同，后续接入主流程只需放到合适位置。

运行前现场依赖：
1. ROS 已启动：roscore
2. SummerTTS 播报节点已启动，并订阅 /summer_tts_topic
3. 麦克风可用
4. 已安装 vosk、pyaudio
5. 已下载中文 Vosk 模型，并在 speech_2026.py 的 DEFAULT_MODEL_PATHS["zh"] 中配置，
   或运行前设置环境变量：export VOSK_ZH_MODEL=/实际/中文模型路径

运行：
    python3 voice_speaker_demo_2026.py
"""

import time

import rospy

from summer_tts_speaker import SummerTTSSpeaker
from speech_2026 import CompetitionVoiceService, DEFAULT_OWNER_NAMES


# ===== 2026-09-27 现场可修改配置 START =====
# 当前为临时自定义姓名，保证联调文件可直接运行。
# 依赖现场公布/领队会确认的地方：如果比赛现场指定了真实主人姓名，改这里即可。
TEST_OWNER_NAMES = DEFAULT_OWNER_NAMES.copy()

# 每次录音时长。赛场嘈杂或说话较慢时可调大，例如 6~8 秒。
NAME_RECORD_SECONDS = 5
SWITCH_RECORD_SECONDS = 5
HELP_RECORD_SECONDS = 6
CONFIRM_RECORD_SECONDS = 3
# ===== 2026-09-27 现场可修改配置 END =====


def print_section(title):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def main():
    rospy.init_node("voice_speaker_demo_2026", anonymous=True)

    print_section("初始化播报器")
    speaker = SummerTTSSpeaker()
    speaker.speak("语音识别和语音播报联调开始")
    time.sleep(1)

    print_section("初始化语音识别服务")
    print("当前测试主人姓名候选：", TEST_OWNER_NAMES)
    print("如现场主人姓名不同，请修改本文件 TEST_OWNER_NAMES。")
    voice = CompetitionVoiceService(
        owner_names=TEST_OWNER_NAMES,
        speaker=speaker,
    )

    print_section("测试一：主人姓名识别")
    speaker.speak("测试一，主人姓名识别")
    person_name, name_score, name_raw = voice.ask_owner_name(
        retries=3,
        duration=NAME_RECORD_SECONDS,
    )
    if not person_name:
        person_name = "主人"
        speaker.speak("未识别到候选姓名，后续测试使用主人称呼")
    print("姓名识别原文：", name_raw)
    print("姓名匹配结果：", person_name)
    print("姓名匹配置信度：", name_score)

    print_section("测试二：开关指令识别")
    speaker.speak("测试二，开关指令识别。请说，例如，打开红色开关，或者关闭二号开关")
    switch_cmd = voice.ask_switch_command(
        person_name=person_name,
        retries=3,
        duration=SWITCH_RECORD_SECONDS,
    )
    print("开关指令解析结果：")
    print("  原始文本：", switch_cmd.raw_text)
    print("  动作：", switch_cmd.action)
    print("  颜色：", switch_cmd.color)
    print("  编号：", switch_cmd.number)
    print("  目标：", switch_cmd.target)
    print("  置信度：", switch_cmd.confidence)
    print("  是否有效：", switch_cmd.is_valid)

    print_section("测试三：挥手需求识别")
    speaker.speak("测试三，挥手需求识别。请说出你需要什么帮助")
    help_request = voice.ask_help_request(
        person_name=person_name,
        retries=3,
        duration=HELP_RECORD_SECONDS,
    )
    print("挥手需求解析结果：")
    print("  原始文本：", help_request.raw_text)
    print("  需求文本：", help_request.request_text)
    print("  意图分类：", help_request.intent)
    print("  置信度：", help_request.confidence)

    print_section("测试四：确认/否认识别")
    confirm = voice.ask_confirm(
        "请回答是或不是，确认是否结束语音测试",
        retries=2,
        duration=CONFIRM_RECORD_SECONDS,
    )
    print("确认结果：", confirm)

    speaker.speak("语音识别和语音播报联调结束")
    print_section("测试结束")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
