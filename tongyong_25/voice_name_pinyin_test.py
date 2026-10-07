#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_name_pinyin_test.py

实机测试“播报 -> 等待 done -> 录音 -> Vosk 转文字 -> 姓名拼音匹配 -> 播报匹配结果”。
默认运行 5 轮真实测试。

默认运行：
    python3 voice_name_pinyin_test.py

常用参数：
    python3 voice_name_pinyin_test.py --names 张三 李四 王五 --rounds 5 --score 0.55 --rms 500
    python3 voice_name_pinyin_test.py --offline-only   # 只做离线拼音匹配样例，不启动 ROS/麦克风

依赖：
    pip3 install pypinyin

运行前要求：
1. roscore / ROS_MASTER_URI / ROS_IP 正确；
2. summer_tts_node_with_done 已启动，并订阅 /summer_tts_topic；
3. C++ TTS 节点会在播报完成后发布 /summer_tts_done；
4. Vosk 中文模型路径正确；
5. 麦克风可用。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import List, Tuple

from speech_2026 import (
    CompetitionVoiceParser,
    CompetitionVoiceService,
    DEFAULT_MODEL_PATHS,
    pyaudio,
    text_to_pinyin_key,
)
from summer_tts_speaker import SummerTTSSpeaker


def check_pypinyin() -> bool:
    ok = importlib.util.find_spec("pypinyin") is not None
    if ok:
        print("[环境检查] pypinyin 已安装：拼音匹配可用")
    else:
        print("[环境检查][警告] 未检测到 pypinyin：请执行 pip3 install pypinyin")
        print("[环境检查][警告] 未安装时不会崩溃，但中文转拼音效果会退化")
    return ok


def check_vosk_model() -> bool:
    model_path = os.environ.get("VOSK_ZH_MODEL", DEFAULT_MODEL_PATHS["zh"])
    ok = Path(model_path).exists()
    if ok:
        print(f"[环境检查] Vosk 中文模型存在：{model_path}")
    else:
        print(f"[环境检查][错误] Vosk 中文模型路径不存在：{model_path}")
        print("[环境检查][提示] 请设置 export VOSK_ZH_MODEL=/你的/vosk-model-small-cn-0.22")
    return ok


def check_pyaudio() -> bool:
    ok = pyaudio is not None
    if ok:
        print("[环境检查] pyaudio 可用：可以调用麦克风录音")
    else:
        print("[环境检查][错误] pyaudio 不可用：无法录音")
    return ok


def resolve_by_parser(parser: CompetitionVoiceParser, raw_text: str) -> Tuple[str, float]:
    name, score = parser.parse_owner_name(raw_text)
    return name or "", score


def run_offline_samples(names: List[str]):
    """环境检查后的离线样例，便于先看拼音匹配效果。"""
    print("\n========== 离线拼音匹配样例 ==========")
    print("候选姓名 target_name:", names)
    parser = CompetitionVoiceParser(owner_names=names)
    samples = ["张三", "我叫张三", "李四", "王五", "晚雾", "王武", "找我啊", "张王", "里斯", "环境噪声"]
    for raw in samples:
        matched, score = resolve_by_parser(parser, raw)
        print(
            f"raw={raw:<8} raw_py={text_to_pinyin_key(raw):<14} "
            f"=> matched={matched or '未匹配':<6} score={score:.2f}"
        )


def run_live_rounds(names: List[str], duration: float, rounds: int):
    """真实播报和录音测试，循环 rounds 轮。"""
    print("\n========== 实机姓名播报 + 录音 + 拼音匹配测试 ==========")
    print("候选姓名 target_name:", names)
    print(f"测试轮数：{rounds}，每轮录音 {duration:.1f} 秒")

    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("voice_name_pinyin_test", anonymous=True)

    speaker = SummerTTSSpeaker()
    if not speaker.is_connected():
        print("[环境检查][错误] /summer_tts_topic 当前没有订阅者，无法真实播报。")
        print("请先启动 summer_tts_node_with_done，并确认 rostopic info /summer_tts_topic 有 subscriber。")
        return
    print("[环境检查] 已连接到 /summer_tts_topic 订阅者")

    voice = CompetitionVoiceService(owner_names=names, speaker=speaker)

    voice.say("姓名拼音匹配测试开始", wait=True)
    for idx in range(1, rounds + 1):
        if rospy.is_shutdown():
            break

        print(f"\n========== 第 {idx}/{rounds} 轮 ==========")
        raw_text = voice.ask_raw_text(
            prompt=f"第{idx}轮，请说出你的名字",
            duration=duration,
            retries=1,
            repeat=False,
            free_grammar=True,
        ).strip()

        print(f"[第{idx}轮] 识别原文：{raw_text}")
        print(f"[第{idx}轮] 原文拼音：{text_to_pinyin_key(raw_text)}")

        matched_name, score = voice.parser.parse_owner_name(raw_text)
        if matched_name:
            print(f"[第{idx}轮] 匹配结果：{matched_name}, score={score:.2f}")
            voice.say(f"匹配结果，{matched_name}", wait=True)
        else:
            print(f"[第{idx}轮] 未匹配到候选姓名，score={score:.2f}")
            voice.say("没有匹配到候选姓名", wait=True)

    voice.say("姓名拼音匹配测试结束", wait=True)
    print("\n测试结束。")


def main():
    parser = argparse.ArgumentParser(description="循环测试姓名拼音容错匹配")
    parser.add_argument("--names", nargs="+", default=["张三", "李四", "王五"], help="候选主人姓名列表")
    parser.add_argument("--rounds", type=int, default=5, help="实机测试轮数，默认 5")
    parser.add_argument("--duration", type=float, default=5.0, help="每轮录音时长")
    parser.add_argument("--score", type=float, default=None, help="拼音匹配阈值，对应 ROBO_NAME_PINYIN_MIN_SCORE")
    parser.add_argument("--rms", type=int, default=None, help="录音能量阈值，对应 ROBO_MIN_RMS")
    parser.add_argument("--offline-only", action="store_true", help="只做离线样例，不启动 ROS/TTS/麦克风")
    parser.add_argument("--skip-offline", action="store_true", help="跳过离线样例，直接实机测试")
    args = parser.parse_args()

    if args.score is not None:
        os.environ["ROBO_NAME_PINYIN_MIN_SCORE"] = str(args.score)
    if args.rms is not None:
        os.environ["ROBO_MIN_RMS"] = str(args.rms)

    print("========== 环境检查 ==========")
    print("ROBO_NAME_PINYIN_MIN_SCORE=", os.environ.get("ROBO_NAME_PINYIN_MIN_SCORE", "0.55 默认"))
    print("ROBO_MIN_RMS=", os.environ.get("ROBO_MIN_RMS", "300 默认"))
    check_pypinyin()

    if not args.skip_offline:
        run_offline_samples(args.names)

    if args.offline_only:
        return

    model_ok = check_vosk_model()
    audio_ok = check_pyaudio()
    if not model_ok or not audio_ok:
        print("[环境检查][错误] 模型或麦克风环境不满足，停止实机测试。")
        return

    run_live_rounds(args.names, duration=args.duration, rounds=args.rounds)


if __name__ == "__main__":
    main()
