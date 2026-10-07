#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_name_pinyin_test.py

用于单独测试“主人姓名识别 -> 拼音容错 -> target_name 列表匹配”的脚本。

测试目标：
1. 离线测试：不需要 ROS/TTS/麦克风，只测试 parse_owner_name() 是否能把
   “晚雾”匹配到“王五”、“找我啊”尽量匹配到候选姓名。
2. 实机测试：播报“你叫什么名字”，等待 /summer_tts_done 后录音识别，
   再把识别文本与 target_name 的姓名做拼音相似度匹配，最后播报标准姓名。

运行示例：
    # 只做离线拼音匹配测试
    python3 voice_name_pinyin_test.py --offline

    # 使用默认姓名列表做实机测试
    python3 voice_name_pinyin_test.py --live

    # 指定姓名列表与阈值
    python3 voice_name_pinyin_test.py --live --names 张三 李四 王五 --score 0.55 --rms 500

依赖：
    pip3 install pypinyin
否则脚本仍可运行，但拼音匹配会退化为普通文本相似度，效果会差很多。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from typing import List, Tuple

from speech_2026 import CompetitionVoiceParser, CompetitionVoiceService, text_to_pinyin_key
from summer_tts_speaker import SummerTTSSpeaker


def check_pypinyin():
    """提示 pypinyin 是否可用。"""
    ok = importlib.util.find_spec("pypinyin") is not None
    if ok:
        print("[依赖检查] pypinyin 已安装：拼音匹配可用")
    else:
        print("[依赖检查] 未检测到 pypinyin：请在机器人上执行 pip3 install pypinyin")
        print("[依赖检查] 当前会退化为文本相似度，无法真正验证中文转拼音效果")
    return ok


def resolve_by_parser(parser: CompetitionVoiceParser, raw_text: str) -> Tuple[str, float]:
    """调用 speech_2026 中的统一姓名解析逻辑。"""
    name, score = parser.parse_owner_name(raw_text)
    return name or "", score


def run_offline_tests(names: List[str]):
    """离线测试若干常见误识别文本。"""
    print("\n========== 离线姓名拼音匹配测试 ==========")
    print("候选姓名 target_name:", names)
    parser = CompetitionVoiceParser(owner_names=names)

    samples = [
        "张三",
        "我叫张三",
        "李四",
        "王五",
        "晚雾",      # 期望接近 王五 / wangwu
        "王武",      # 期望接近 王五 / wangwu
        "找我啊",    # 可能接近 张王/张?，用于观察分数
        "张王",
        "里斯",      # 期望接近 李四 / lisi
        "环境噪声",
    ]

    for raw in samples:
        matched, score = resolve_by_parser(parser, raw)
        print(
            f"raw={raw:<8} raw_py={text_to_pinyin_key(raw):<14} "
            f"=> matched={matched or '未匹配':<6} score={score:.2f}"
        )

    print("\n说明：")
    print("- 如果 raw='晚雾' 能匹配到 '王五'，说明拼音容错生效。")
    print("- 如果环境噪声也匹配到姓名，说明 ROBO_NAME_PINYIN_MIN_SCORE 过低，需要调高。")


def run_live_test(names: List[str], duration: float, retries: int):
    """实机测试：播报问题，录音识别，再做姓名拼音匹配并播报结果。"""
    print("\n========== 实机姓名识别 + 拼音匹配测试 ==========")
    print("候选姓名 target_name:", names)

    import rospy

    rospy.init_node("voice_name_pinyin_test", anonymous=True)
    speaker = SummerTTSSpeaker()
    voice = CompetitionVoiceService(owner_names=names, speaker=speaker)

    raw_text = voice.ask_raw_text(
        prompt="你叫什么名字？",
        duration=duration,
        retries=retries,
        repeat=False,
        free_grammar=True,
    ).strip()

    print(f"[实机识别原文] {raw_text}")
    print(f"[实机识别拼音] {text_to_pinyin_key(raw_text)}")

    matched_name, score = voice.parser.parse_owner_name(raw_text)
    if matched_name:
        print(f"[姓名匹配结果] {matched_name}, score={score:.2f}")
        voice.say(f"好的，{matched_name}", wait=True)
    else:
        print(f"[姓名匹配结果] 未匹配到候选姓名，score={score:.2f}")
        voice.say("没有匹配到候选姓名", wait=True)


def main():
    parser = argparse.ArgumentParser(description="测试姓名拼音容错匹配")
    parser.add_argument("--names", nargs="+", default=["张三", "李四", "王五"], help="候选主人姓名列表")
    parser.add_argument("--offline", action="store_true", help="只运行离线样例测试")
    parser.add_argument("--live", action="store_true", help="运行实机播报、录音、识别测试")
    parser.add_argument("--duration", type=float, default=5.0, help="实机录音时长")
    parser.add_argument("--retries", type=int, default=3, help="实机重试次数")
    parser.add_argument("--score", type=float, default=None, help="拼音匹配阈值，对应 ROBO_NAME_PINYIN_MIN_SCORE")
    parser.add_argument("--rms", type=int, default=None, help="录音能量阈值，对应 ROBO_MIN_RMS")
    args = parser.parse_args()

    if args.score is not None:
        os.environ["ROBO_NAME_PINYIN_MIN_SCORE"] = str(args.score)
    if args.rms is not None:
        os.environ["ROBO_MIN_RMS"] = str(args.rms)

    print("ROBO_NAME_PINYIN_MIN_SCORE=", os.environ.get("ROBO_NAME_PINYIN_MIN_SCORE", "0.55 默认"))
    print("ROBO_MIN_RMS=", os.environ.get("ROBO_MIN_RMS", "300 默认"))
    check_pypinyin()

    # 默认至少跑离线测试，避免用户忘记参数后什么都不做。
    if args.offline or not args.live:
        run_offline_tests(args.names)

    if args.live:
        run_live_test(args.names, duration=args.duration, retries=args.retries)


if __name__ == "__main__":
    main()
