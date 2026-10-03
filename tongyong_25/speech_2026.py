#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
当前speech_2026.py 主要负责：
- 录音；
- 中文语音识别；
- 文本解析；
- 开关指令解析；
- 主人姓名解析；
- 挥手需求解析；
- 确认/否认解析。

设计目标：
1. 全中文离线语音识别，默认使用 Vosk 中文模型；
2. 支持比赛固定词表/语法约束，提高嘈杂赛场中的稳定性；
3. 提供主人姓名、开关控制、挥手咨询需求、确认/否认等解析接口；
4. 保持与现有 `jujia26.py` 的调用方式兼容，可逐步替换原 `vosk_speech_recognition.py`。

现场需要改的地方：
- `DEFAULT_OWNER_NAMES`：领队会/赛前现场确定 3 位主人姓名后，在这里改；也可在初始化时传入。
- `DEFAULT_SWITCH_COLORS` / `DEFAULT_SWITCH_NUMBERS`：若现场开关标记颜色、编号、贴纸文字不同，在这里改。
- `DEFAULT_TRASH_WORDS`：若垃圾类型固定，可在这里补充。
- `DEFAULT_MODEL_PATHS["zh"]`：按现场电脑实际 Vosk 中文模型路径修改。

依赖：
- vosk
- pyaudio
- wave/json/os/time/datetime 等标准库
- 模型：https://alphacephei.com/vosk/models（Linux：wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip vosk-model-small-cn-0.22.zip）
推荐中文模型：
- vosk-model-small-cn-0.22：体积小，速度快；
- vosk-model-cn-0.22：准确率更高，但加载慢、占用大。
"""

from __future__ import annotations

import datetime
import audioop
import json
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import rospy

try:
    import pyaudio
except Exception:  # pragma: no cover - 机器人现场环境才会有麦克风依赖
    pyaudio = None


# =========================
# 现场配置区
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_PATHS = {
    # TODO 现场如果模型放在别处，直接改这一行，或初始化 VoskChineseRecognizer(model_path=...)。
    "zh": os.environ.get(
        "VOSK_ZH_MODEL",
        "/home/zq/catkin_ws/src/cmoon/src/vosk_speech_recognition/models/vosk-model-small-cn-0.22",
    ),
    "en": os.environ.get(
        "VOSK_EN_MODEL",
        "/home/zq/catkin_ws/src/cmoon/src/vosk_speech_recognition/models/vosk-model-en-us-0.22-lgraph",
    ),
}

# TODO 赛前/领队会确认后，把这里改成现场 3 位主人的候选名。
# 若现场是随机志愿者，建议提前录入所有可能出现的名字，或改用自由识别后人工确认。
DEFAULT_OWNER_NAMES = ["张三", "李四", "王五"]

DEFAULT_SWITCH_COLORS = ["红色", "黄色", "蓝色", "绿色", "白色", "黑色", "红", "黄", "蓝", "绿", "白", "黑"]
DEFAULT_SWITCH_NUMBERS = ["一", "二", "三", "四", "五", "六", "1", "2", "3", "4", "5", "6", "第一", "第二", "第三", "第四", "第五", "第六"]
DEFAULT_SWITCH_ACTIONS = ["打开", "关闭", "开", "关", "开启", "关掉", "打开一下", "关闭一下"]
DEFAULT_SWITCH_OBJECTS = ["开关", "电灯", "灯", "空调", "电视", "风扇", "插座", "电器"]
DEFAULT_HELP_WORDS = ["帮忙", "帮助", "需要", "请问", "咨询", "拿", "取", "找", "开", "关"]
DEFAULT_TRASH_WORDS = ["垃圾", "纸团", "空瓶", "瓶子", "水果皮", "果皮", "废纸", "纸杯"]
YES_WORDS = ["是", "对", "正确", "没错", "确认", "可以", "好", "好的"]
NO_WORDS = ["不是", "不对", "错误", "否", "不要", "取消", "重新"]


# =========================
# 数据结构
# =========================

@dataclass
class RecognitionResult:
    text: str
    audio_file: Optional[str] = None
    raw: Optional[dict] = None
    success: bool = False


@dataclass
class SwitchCommand:
    raw_text: str
    action: Optional[str] = None      # "开" / "关"
    color: Optional[str] = None       # 红色/黄色/...
    number: Optional[str] = None      # 1/2/3/...
    target: Optional[str] = None      # 灯/空调/开关/...
    confidence: float = 0.0

    @property
    def is_valid(self) -> bool:
        # 赛规中核心是判断主人对电气开关的需求，至少要知道开/关，并尽量知道目标。
        return self.action is not None and (self.color is not None or self.number is not None or self.target is not None)


@dataclass
class HelpRequest:
    raw_text: str
    request_text: str
    intent: str = "general_help"
    confidence: float = 0.0


# =========================
# 工具函数
# =========================

_CN_NUM_MAP = {
    "一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5", "六": "6",
    "七": "7", "八": "8", "九": "9", "十": "10",
    "第一": "1", "第二": "2", "第三": "3", "第四": "4", "第五": "5", "第六": "6",
}

_COLOR_NORMALIZE = {
    "红": "红色", "黄": "黄色", "蓝": "蓝色", "绿": "绿色", "白": "白色", "黑": "黑色",
    "红色": "红色", "黄色": "黄色", "蓝色": "蓝色", "绿色": "绿色", "白色": "白色", "黑色": "黑色",
}

_ACTION_NORMALIZE = {
    "打开": "开", "开": "开", "开启": "开", "打开一下": "开",
    "关闭": "关", "关": "关", "关掉": "关", "关闭一下": "关",
}


def normalize_text(text: Optional[str]) -> str:
    """去掉空格、标点和常见语气词，便于关键词匹配。"""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s,，。.!！?？:：;；、\"'‘’“”()（）\[\]【】]", "", text)
    for filler in ("那个", "就是", "嗯", "啊", "呀", "吧", "请", "帮我", "麻烦你"):
        text = text.replace(filler, "")
    return text


def unique_keep_order(words: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for w in words:
        if not w:
            continue
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out


def build_grammar_phrases(
    owner_names: Sequence[str],
    extra_words: Sequence[str] = (),
) -> List[str]:
    """
    构造 Vosk grammar 词表。
    Vosk grammar 不是严格自然语言理解，但能显著提高固定短语识别稳定性。
    """
    base_words = []
    base_words += list(owner_names)
    base_words += DEFAULT_SWITCH_ACTIONS + DEFAULT_SWITCH_COLORS + DEFAULT_SWITCH_NUMBERS + DEFAULT_SWITCH_OBJECTS
    base_words += DEFAULT_HELP_WORDS + DEFAULT_TRASH_WORDS + YES_WORDS + NO_WORDS

    # 常见完整句式，利于识别开关需求。
    phrases = []
    for action in ["打开", "关闭", "开", "关"]:
        for color in ["红色", "黄色", "蓝色", "绿色", "白色", "黑色"]:
            phrases.append(f"{action}{color}开关")
        for num in ["一", "二", "三", "四", "五", "六"]:
            phrases.append(f"{action}第{num}个开关")
            phrases.append(f"{action}{num}号开关")
        for obj in ["灯", "空调", "电视", "风扇"]:
            phrases.append(f"{action}{obj}")

    phrases += ["我需要帮助", "请问你需要什么帮助", "是", "不是", "确认", "重新"]
    phrases += list(extra_words)
    return unique_keep_order(base_words + phrases)


# =========================
# Vosk 中文识别器
# =========================

class VoskChineseRecognizer:
    """中文优先的 Vosk 离线识别器。"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "zh",
        sample_rate: int = 16000,
        audio_dir: Optional[str] = None,
        grammar: Optional[Sequence[str]] = None,
        log_file: Optional[str] = None,
        min_rms: Optional[int] = None,
    ):
        self.language = language
        self.sample_rate = sample_rate
        self.model_path = model_path or DEFAULT_MODEL_PATHS.get(language, DEFAULT_MODEL_PATHS["zh"])
        self.audio_dir = Path(audio_dir or os.environ.get("ROBO_AUDIO_DIR", str(PROJECT_ROOT / "audio")))
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = Path(log_file or (PROJECT_ROOT / "voice_recognition_log.txt"))
        self.grammar = list(grammar) if grammar else None
        # ===== 2026-09-30 静音误识别修正 START =====
        # Vosk 在 grammar 约束下，安静环境/底噪/残留机器人播报也可能被硬匹配成某个词。
        # 先用录音 RMS 能量做静音门限；低于门限则直接判定“未听到有效人声”，不送入 Vosk。
        # 现场可按麦克风灵敏度调整：
        #   export ROBO_MIN_RMS=300
        self.min_rms = int(os.environ.get("ROBO_MIN_RMS", str(min_rms or 300)))
        # ===== 2026-09-30 静音误识别修正 END =====
        self.model = None
        self.recognizer = None
        self._load_model()

    def _load_model(self):
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("未安装 vosk，请在机器人环境执行：pip install vosk") from exc

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Vosk 模型路径不存在：{}\n"
                "请下载中文模型 vosk-model-small-cn-0.22，并修改 DEFAULT_MODEL_PATHS['zh'] 或设置环境变量 VOSK_ZH_MODEL。".format(
                    self.model_path
                )
            )

        self.model = vosk.Model(self.model_path)
        self._reset_recognizer()

    def _reset_recognizer(self):
        """只重置 KaldiRecognizer，不重复加载大模型。"""
        import vosk
        if self.grammar:
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate, json.dumps(self.grammar, ensure_ascii=False))
        else:
            self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

    def reset_grammar(self, grammar: Optional[Sequence[str]] = None):
        """按不同环节切换词表，如注册姓名/开关控制/确认。"""
        self.grammar = list(grammar) if grammar else None
        self._reset_recognizer()

    def record_audio(self, duration: float = 5.0, filename: Optional[str] = None) -> str:
        if pyaudio is None:
            raise RuntimeError("未安装 pyaudio，无法从麦克风录音")

        if filename is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = str(self.audio_dir / f"recording_{stamp}.wav")

        chunk = 1024
        fmt = pyaudio.paInt16
        channels = 1
        rate = self.sample_rate

        pa = pyaudio.PyAudio()
        stream = pa.open(format=fmt, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = []
        try:
            for _ in range(0, int(rate / chunk * duration)):
                frames.append(stream.read(chunk, exception_on_overflow=False))
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pa.get_sample_size(fmt))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
        return filename

    # ===== 2026-09-30 静音误识别修正 START =====
    def _audio_rms(self, audio_file: str) -> int:
        """计算 16-bit 单声道 WAV 的 RMS 能量，用于过滤静音/底噪误识别。"""
        with wave.open(audio_file, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            if not frames:
                return 0
            return audioop.rms(frames, wf.getsampwidth())
    # ===== 2026-09-30 静音误识别修正 END =====

    def recognize_file(self, audio_file: str) -> RecognitionResult:
        if self.recognizer is None:
            self._load_model()

        with wave.open(audio_file, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise ValueError("音频必须是 16-bit 单声道 WAV")
            if wf.getframerate() != self.sample_rate:
                raise ValueError(f"音频采样率必须是 {self.sample_rate}Hz")
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                self.recognizer.AcceptWaveform(data)
            raw = json.loads(self.recognizer.FinalResult())

        text = raw.get("text", "") or ""
        text = text.replace(" ", "")
        result = RecognitionResult(text=text, audio_file=audio_file, raw=raw, success=bool(text))
        self._log(result)
        return result

    def listen_once(self, duration: float = 5.0) -> RecognitionResult:
        print(f"[开始录音] 录音 {duration:.1f} 秒，请现在说话")
        audio_file = self.record_audio(duration=duration)
        # ===== 2026-09-30 静音误识别修正 START =====
        rms = self._audio_rms(audio_file)
        print(f"[录音能量] RMS={rms}, 阈值={self.min_rms}")
        if rms < self.min_rms:
            result = RecognitionResult(
                text="",
                audio_file=audio_file,
                raw={"reason": "low_rms", "rms": rms, "min_rms": self.min_rms},
                success=False,
            )
            self._log(result)
            print("[语音识别] 录音能量低于阈值，判定为无人声/静音，跳过 Vosk 识别")
            return result
        # ===== 2026-09-30 静音误识别修正 END =====
        return self.recognize_file(audio_file)

    def listen_with_retry(
        self,
        duration: float = 5.0,
        retries: int = 3,
        prompt_speaker=None,
        retry_prompt: str = "没有听清，请再说一遍",
    ) -> RecognitionResult:
        last = RecognitionResult(text="", success=False)
        for idx in range(retries):
            last = self.listen_once(duration=duration)
            if last.success:
                return last
            if prompt_speaker is not None and idx < retries - 1:
                prompt_speaker.speak(retry_prompt)
        return last

    def _log(self, result: RecognitionResult):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] file={result.audio_file} text={result.text} raw={result.raw}\n")


# =========================
# 赛项语义解析
# =========================

class CompetitionVoiceParser:
    """将识别文本解析成 2026 居家生活赛项需要的结构化指令。"""

    def __init__(
        self,
        owner_names: Optional[Sequence[str]] = None,
        switch_colors: Optional[Sequence[str]] = None,
        switch_numbers: Optional[Sequence[str]] = None,
        switch_objects: Optional[Sequence[str]] = None,
    ):
        self.owner_names = list(owner_names or DEFAULT_OWNER_NAMES)
        self.switch_colors = list(switch_colors or DEFAULT_SWITCH_COLORS)
        self.switch_numbers = list(switch_numbers or DEFAULT_SWITCH_NUMBERS)
        self.switch_objects = list(switch_objects or DEFAULT_SWITCH_OBJECTS)

    def parse_owner_name(self, text: str) -> Tuple[Optional[str], float]:
        """匹配主人姓名。后续只需修改 DEFAULT_OWNER_NAMES 或初始化传入 owner_names。"""
        norm = normalize_text(text)
        if not norm:
            return None, 0.0
        # 直接包含优先。
        for name in self.owner_names:
            if name and name in norm:
                return name, 1.0
        # 简单字符重叠兜底，避免现场没有 pypinyin/fuzzywuzzy 时不可用。
        best_name, best_score = None, 0.0
        for name in self.owner_names:
            if not name:
                continue
            common = sum(1 for ch in name if ch in norm)
            score = common / max(len(name), 1)
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= 0.67:
            return best_name, best_score
        return None, best_score

    def parse_switch_command(self, text: str) -> SwitchCommand:
        """解析主人对开关/电器的需求。"""
        norm = normalize_text(text)
        cmd = SwitchCommand(raw_text=text or "")
        if not norm:
            return cmd

        # 动作：先识别长词，避免“关闭”中也有“关”。
        for word in sorted(DEFAULT_SWITCH_ACTIONS, key=len, reverse=True):
            if word in norm:
                cmd.action = _ACTION_NORMALIZE.get(word, word)
                break

        # 颜色。
        for color in sorted(self.switch_colors, key=len, reverse=True):
            if color in norm:
                cmd.color = _COLOR_NORMALIZE.get(color, color)
                break

        # 编号。
        for num in sorted(self.switch_numbers, key=len, reverse=True):
            if num in norm:
                cmd.number = _CN_NUM_MAP.get(num, num)
                break
        # “2号/3号”这类阿拉伯数字兜底。
        m = re.search(r"([1-9])(?:号|个)?", norm)
        if cmd.number is None and m:
            cmd.number = m.group(1)

        # 目标对象。
        for obj in sorted(self.switch_objects, key=len, reverse=True):
            if obj in norm:
                cmd.target = obj
                break
        if cmd.target is None and "开关" in norm:
            cmd.target = "开关"

        score = 0.0
        score += 0.45 if cmd.action else 0.0
        score += 0.25 if cmd.color else 0.0
        score += 0.20 if cmd.number else 0.0
        score += 0.10 if cmd.target else 0.0
        cmd.confidence = min(score, 1.0)
        return cmd

    def parse_help_request(self, text: str) -> HelpRequest:
        """挥手主人咨询需求。赛规只要求中文语音交互并播报，默认保留原话。"""
        raw = text or ""
        norm = normalize_text(raw)
        intent = "general_help"
        conf = 0.5 if norm else 0.0
        if any(w in norm for w in ["拿", "取", "给我", "递给"]):
            intent, conf = "fetch_item", 0.75
        elif any(w in norm for w in ["开", "关", "打开", "关闭"]):
            intent, conf = "switch_control", 0.75
        elif any(w in norm for w in ["在哪里", "找", "寻找"]):
            intent, conf = "find_something", 0.7
        return HelpRequest(raw_text=raw, request_text=raw.strip(), intent=intent, confidence=conf)

    def parse_yes_no(self, text: str) -> Optional[bool]:
        norm = normalize_text(text)
        if not norm:
            return None
        if any(w in norm for w in NO_WORDS):
            return False
        if any(w in norm for w in YES_WORDS):
            return True
        return None


# =========================
# 对主流程友好的服务封装
# =========================

class CompetitionVoiceService:
    """
    主流程建议只调用这个类。

    示例：
        voice = CompetitionVoiceService(owner_names=["张三", "李四", "王五"], speaker=self.speak)
        name = voice.ask_owner_name()
        cmd = voice.ask_switch_command("张三")
        req = voice.ask_help_request("李四")
    """

    def __init__(
        self,
        owner_names: Optional[Sequence[str]] = None,
        speaker=None,
        model_path: Optional[str] = None,
        audio_dir: Optional[str] = None,
        wait_after_tts: bool = True,
    ):
        self.owner_names = list(owner_names or DEFAULT_OWNER_NAMES)
        self.speaker = speaker
        # ===== 2026-09-30 语音监听时序修正 START =====
        # SummerTTSSpeaker.speak() 是非阻塞发布 topic，函数返回时机器人可能还在说话。
        # 如果立刻开始录音，会把机器人自己的播报录进去，表现为“没等主人回答就往下跑”。
        # 因此在需要听主人回答的提示语后，按文本长度估算等待时间，再开始录音。
        self.wait_after_tts = wait_after_tts
        # ===== 2026-09-30 语音监听时序修正 END =====
        self.parser = CompetitionVoiceParser(owner_names=self.owner_names)
        grammar = build_grammar_phrases(self.owner_names)
        self.recognizer = VoskChineseRecognizer(
            model_path=model_path,
            language="zh",
            audio_dir=audio_dir,
            grammar=grammar,
        )

    def _estimate_tts_seconds(self, text: str) -> float:
        """按中文播报长度估算等待时间，避免录到机器人自己的声音。"""
        if not text:
            return 0.0
        # ===== 2026-09-30 语音监听时序修正 START =====
        # 现场 summer_tts_node 是“生成 wav + play 播放”的异步链路，Python publish 返回时，
        # TTS 可能还没开始播；因此等待必须比单纯文本时长更保守。
        # 当前没有 TTS“播放完成”回传 topic，因此采用：
        #   等待时长 = 固定启动/合成缓冲 + 中文字数 * 单字播报时长
        # 可现场按语速调参：
        #   export ROBO_TTS_SEC_PER_CHAR=0.45
        #   export ROBO_TTS_WAIT_EXTRA=2.5
        #   export ROBO_TTS_WAIT_SCALE=1.0
        scale = float(os.environ.get("ROBO_TTS_WAIT_SCALE", "1.0"))
        extra = float(os.environ.get("ROBO_TTS_WAIT_EXTRA", "2.5"))
        sec_per_char = float(os.environ.get("ROBO_TTS_SEC_PER_CHAR", "0.45"))
        return min(max((len(text) * sec_per_char + extra) * scale, 2.0), 30.0)
        # ===== 2026-09-30 语音监听时序修正 END =====

    def say(self, text: str, wait: bool = False):
        # ===== 2026-09-30 TTS播放完成同步 START =====
        # 优先使用新版 SummerTTSSpeaker.speak(wait_done=True)，等待 C++ 节点发布 /summer_tts_done。
        # 如果现场还是旧版 TTS 节点/旧版 speaker，则回退到“按文本长度估算等待”。
        published = False
        if self.speaker is not None:
            if wait:
                try:
                    print(f"[TTS同步] 发送播报并等待完成：{text}")
                    ok = self.speaker.speak(text, wait_done=True, timeout=30.0)
                    published = True
                    if ok:
                        print("[TTS同步] 已收到 /summer_tts_done，下一步将开始录音/识别")
                        return
                    print("[TTS同步] 未收到 /summer_tts_done，回退到估算等待")
                except TypeError:
                    # 兼容旧版 SummerTTSSpeaker.speak(text)
                    self.speaker.speak(text)
                    published = True
            else:
                self.speaker.speak(text)
                published = True
        else:
            print(f"[TTS] {text}")
        # ===== 2026-09-30 TTS播放完成同步 END =====

        # ===== 2026-09-30 语音监听时序修正 START =====
        if wait and self.wait_after_tts:
            wait_seconds = self._estimate_tts_seconds(text)
            if not published:
                print(f"[TTS等待] 无真实 speaker，仅控制台输出：{text}")
            else:
                print(f"[TTS等待] 已发送播报：{text}")
            print(f"[TTS等待] 按 {len(text)} 字估算等待约 {wait_seconds:.1f} 秒，等待播报完成后再录音")
            time.sleep(wait_seconds)
            print("[TTS等待] 播报等待结束，下一步将开始录音/识别")
        # ===== 2026-09-30 语音监听时序修正 END =====

    def ask_raw_text(
        self,
        prompt: str,
        duration: float = 5.0,
        retries: int = 3,
        repeat: bool = True,
        repeat_template: str = "你说的是，{}",
        retry_prompt: str = "没有听清，请再说一遍",
        free_grammar: bool = True,
    ) -> str:
        """
        固定话术播报 -> 等待 TTS 完成 -> 录音识别 -> 原文复述。

        这是当前主流程推荐使用的“交互原话”接口：不切割词条、不匹配关键词，
        只返回 Vosk 识别出的原始文本，并可按 repeat_template 复述。
        free_grammar=True 时临时关闭 grammar，避免词条不完整时被强行匹配到固定词表。
        """
        last_text = ""
        for attempt in range(1, retries + 1):
            if rospy.is_shutdown():
                return last_text

            self.say(prompt, wait=True)

            old_grammar = list(self.recognizer.grammar) if self.recognizer.grammar else None
            if free_grammar:
                self.recognizer.reset_grammar(None)
            try:
                result = self.recognizer.listen_once(duration=duration)
            finally:
                if free_grammar:
                    self.recognizer.reset_grammar(old_grammar)

            text = (result.text or "").strip()
            last_text = text
            print(f"[语音交互原文] 第 {attempt}/{retries} 次：{text}")
            if text:
                if repeat:
                    self.say(repeat_template.format(text), wait=True)
                return text

            if attempt < retries:
                self.say(retry_prompt, wait=True)

        return last_text

    # ===== 2026-09-30 赛规语音覆盖补充 START =====
    # 以下 announce_* 方法不做识别，只负责把主流程/视觉/行为识别结果用统一话术播报。
    # 这样居家生活赛项 jujia26.py 和具身服务/智能赛项 zhineng 主流程都可以复用本文件。
    def announce_owner_registered(self, person_name: str, owner_index: Optional[int] = None):
        """人脸注册后播报注册结果。"""
        if owner_index is None:
            self.say(f"{person_name}，注册完成", wait=True)
        else:
            self.say(f"主人{owner_index}，{person_name}，注册完成", wait=True)

    def announce_owner_recognized(self, person_name: str, owner_index: Optional[int] = None):
        """识别人脸后播报主人姓名。"""
        if owner_index is None:
            self.say(f"识别到主人，{person_name}", wait=True)
        else:
            self.say(f"识别到主人{owner_index}，{person_name}", wait=True)

    def announce_behavior(self, behavior: str, person_name: Optional[str] = None):
        """识别姿态/行为后播报行为名称。"""
        prefix = f"{person_name}的行为是" if person_name else "识别到的行为是"
        self.say(f"{prefix}{behavior}", wait=True)

    def announce_trash_found(self, trash_name: str = "垃圾", location_desc: Optional[str] = None):
        """居家生活：找到地面垃圾后播报。"""
        if location_desc:
            self.say(f"发现{location_desc}的{trash_name}", wait=True)
        else:
            self.say(f"发现{trash_name}", wait=True)

    def announce_object_recognized(self, object_name: str, extra_desc: Optional[str] = None):
        """具身服务/智能赛项：成功识别物品后播报。"""
        if extra_desc:
            self.say(f"识别到{extra_desc}，{object_name}", wait=True)
        else:
            self.say(f"识别到物品，{object_name}", wait=True)
    # ===== 2026-09-30 赛规语音覆盖补充 END =====

    def ask_owner_name(self, retries: int = 3, duration: float = 4.0) -> Tuple[Optional[str], float, str]:
        """询问并识别主人姓名。返回：(姓名, 置信度, 原始文本)。"""
        for _ in range(retries):
            self.say("你叫什么名字？", wait=True)
            result = self.recognizer.listen_once(duration=duration)
            name, score = self.parser.parse_owner_name(result.text)
            if name:
                self.say(f"好的，{name}", wait=True)
                return name, score, result.text
            self.say("没有听清，请再说一遍", wait=True)
        return None, 0.0, ""

    def ask_switch_command(self, person_name: str = "主人", retries: int = 3, duration: float = 5.0) -> SwitchCommand:
        """询问坐/躺主人需要操作哪个开关。"""
        last_cmd = SwitchCommand(raw_text="")
        for _ in range(retries):
            self.say("请说开关需求", wait=True)
            result = self.recognizer.listen_once(duration=duration)
            cmd = self.parser.parse_switch_command(result.text)
            last_cmd = cmd
            if cmd.is_valid:
                desc = []
                if cmd.action:
                    desc.append(cmd.action)
                if cmd.color:
                    desc.append(cmd.color)
                if cmd.number:
                    desc.append(f"{cmd.number}号")
                desc.append(cmd.target or "开关")
                self.say("收到，你的需求是" + "".join(desc), wait=True)
                return cmd
            self.say("没有识别清楚开关需求，请再说一遍", wait=True)
        return last_cmd

    # ===== 2026-09-30 赛规语音覆盖补充 START =====
    def ask_switch_command_confirmed(
        self,
        person_name: str = "主人",
        retries: int = 3,
        duration: float = 6.0,
        confirm_retries: int = 2,
        confirm_duration: float = 3.0,
    ) -> Tuple[SwitchCommand, Optional[bool]]:
        """
        坐/躺/休息主人开关需求：询问、解析并二次确认。

        返回：
            (SwitchCommand, confirmed)
            confirmed=True：主人确认；
            confirmed=False：主人否认；
            confirmed=None：未听清确认。
        """
        cmd = self.ask_switch_command(person_name=person_name, retries=retries, duration=duration)
        if not cmd.is_valid:
            return cmd, False

        desc = []
        if cmd.action:
            desc.append(cmd.action)
        if cmd.color:
            desc.append(cmd.color)
        if cmd.number:
            desc.append(f"{cmd.number}号")
        desc.append(cmd.target or "开关")
        prompt = "确认" + "".join(desc) + "吗？请回答是或不是"
        confirmed = self.ask_confirm(prompt, retries=confirm_retries, duration=confirm_duration)
        if confirmed is True:
            self.say("好的，我将执行这个开关需求", wait=True)
        elif confirmed is False:
            self.say("好的，请重新说明开关需求", wait=True)
        else:
            self.say("没有听清确认结果，暂不执行开关操作", wait=True)
        return cmd, confirmed
    # ===== 2026-09-30 赛规语音覆盖补充 END =====

    def ask_help_request(self, person_name: str = "主人", retries: int = 3, duration: float = 6.0) -> HelpRequest:
        """询问挥手主人需求。赛规要求中文语音交互，默认识别后复述。"""
        last = HelpRequest(raw_text="", request_text="", confidence=0.0)
        for _ in range(retries):
            self.say("请说需要什么帮助", wait=True)
            result = self.recognizer.listen_once(duration=duration)
            req = self.parser.parse_help_request(result.text)
            last = req
            if req.request_text:
                self.say(f"你的需求是，{req.request_text}", wait=True)
                return req
            self.say("没有听清，请再说一遍", wait=True)
        return last

    def ask_confirm(self, prompt: str, retries: int = 2, duration: float = 3.0) -> Optional[bool]:
        """确认/否认问答。"""
        for _ in range(retries):
            self.say(prompt, wait=True)
            result = self.recognizer.listen_once(duration=duration)
            yn = self.parser.parse_yes_no(result.text)
            if yn is not None:
                return yn
            self.say("请回答是或不是", wait=True)
        return None


# =========================
# 兼容旧模块的便捷函数
# =========================

_global_service: Optional[CompetitionVoiceService] = None


def get_voice_service(owner_names: Optional[Sequence[str]] = None, speaker=None, model_path: Optional[str] = None) -> CompetitionVoiceService:
    global _global_service
    if _global_service is None:
        _global_service = CompetitionVoiceService(owner_names=owner_names, speaker=speaker, model_path=model_path)
    return _global_service


def record_and_recognize_zh(duration: float = 5.0, model_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """兼容 `record_and_recognize('zh')` 的简化接口。"""
    service = get_voice_service(model_path=model_path)
    result = service.recognizer.listen_once(duration=duration)
    return (result.text if result.success else None), result.audio_file


if __name__ == "__main__":
    # ===== 2026-09-27 模块职责说明 START =====
    # speech_2026.py 只作为“识别 + 解析”功能模块使用，不负责直接耦合 TTS。
    # 真实“识别 + 播报”联调请运行 voice_speaker_demo_2026.py；
    # jujia26.py 中也使用同一个 CompetitionVoiceService 接口，因此联调通过后可直接迁移。
    # ===== 2026-09-27 模块职责说明 END =====
    parser_obj = CompetitionVoiceParser(owner_names=DEFAULT_OWNER_NAMES)
    print("=== speech_2026.py 解析模块自测 ===")
    print("当前临时主人姓名 DEFAULT_OWNER_NAMES:", DEFAULT_OWNER_NAMES)
    print("真实识别和播报联调请运行：python3 voice_speaker_demo_2026.py")
    for sample in ["打开红色开关", "关闭二号开关", "我需要帮助拿一下水", "是", "不是"]:
        print("\n输入:", sample)
        print("开关解析:", parser_obj.parse_switch_command(sample))
        print("需求解析:", parser_obj.parse_help_request(sample))
        print("确认解析:", parser_obj.parse_yes_no(sample))
