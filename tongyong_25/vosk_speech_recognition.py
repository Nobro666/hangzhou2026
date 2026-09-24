#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Vosk的离线语音识别程序
适用于Ubuntu系统
支持中文和英文语音识别
功能：将语音转换为文本字符串
"""

import json
import os
import sys
import datetime
import argparse
import threading
import time
from typing import Optional, List, Tuple, Callable
import wave
import pyaudio

# 全局变量
_global_recognizer = None
_global_model = None

class VoskSpeechRecognition:
    def __init__(self, model_path: str = None, language: str = 'en'):
        """
        初始化Vosk语音识别器
        
        Args:
            model_path: 模型文件路径
            language: 语言代码，'zh'为中文，'en'为英文
        """
        self.language = language
        self.log_file = "vosk_speech_recognition_log.txt"
        # self.model_path = "/home/zq/catkin_ws/src/cmoon/src/vosk_speech_recognition/models/vosk-model-en-us-0.22"
        self.model_path = "/home/zq/catkin_ws/src/cmoon/src/vosk_speech_recognition/models/vosk-model-en-us-0.22-lgraph"
        self.model = None
        self.recognizer = None
        
        # 初始化Vosk
        self._init_vosk()
        
        print(f"Vosk语音识别器初始化完成，语言: {language}")
    
    def _init_vosk(self):
        """初始化Vosk模型和识别器"""
        try:
            import vosk
            
            # 如果没有指定模型路径，使用默认路径
            if not self.model_path:
                if self.language == 'zh':
                    self.model_path = "vosk-model-small-cn-0.22"
                else:
                    self.model_path = "vosk-model-small-en-us-0.15"
            
            # 检查模型是否存在
            if not os.path.exists(self.model_path):
                print(f"模型路径不存在: {self.model_path}")
                print("请先下载对应的语言模型")
                self._download_model_info()
                return
            
            # 加载模型
            print(f"正在加载模型: {self.model_path}")
            self.model = vosk.Model(self.model_path)
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
            
            print("模型加载成功")
            
        except ImportError:
            print("Vosk库未安装，请运行: pip install vosk")
            sys.exit(1)
        except Exception as e:
            print(f"初始化Vosk失败: {e}")
            sys.exit(1)
    
    def _download_model_info(self):
        """显示模型下载信息"""
        print("\n=== 模型下载信息 ===")
        if self.language == 'zh':
            print("中文模型下载地址:")
            print("https://alphacephei.com/vosk/models")
            print("推荐模型: vosk-model-small-cn-0.22")
            print("下载命令:")
            print("wget https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip")
            print("unzip vosk-model-small-cn-0.22.zip")
        else:
            print("英文模型下载地址:")
            print("https://alphacephei.com/vosk/models")
            print("推荐模型: vosk-model-small-en-us-0.15")
            print("下载命令:")
            print("wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
            print("unzip vosk-model-small-en-us-0.15.zip")
    
    def recognize_from_microphone(self, timeout=5, phrase_time_limit=10) -> Optional[str]:
        """
        从麦克风实时识别语音
        
        Args:
            timeout: 等待语音开始的超时时间（秒）
            phrase_time_limit: 单次语音的最大时长（秒）
            
        Returns:
            str: 识别到的文本，如果失败返回None
        """
        if not self.recognizer:
            print("识别器未初始化")
            return None
        
        try:
            print(f"开始录音，语言: {self.language}")
            print("请说话...")
            
            # 设置音频参数
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            # 初始化PyAudio
            p = pyaudio.PyAudio()
            
            # 打开音频流
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            print("开始录音...")
            frames = []
            
            # 录音
            start_time = time.time()
            while time.time() - start_time < phrase_time_limit:
                data = stream.read(CHUNK)
                frames.append(data)
                
                # 检查是否有语音输入
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        stream.stop_stream()
                        stream.close()
                        p.terminate()
                        
                        timestamp = datetime.datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒")
                        text = result['text']
                        
                        print(f"[{timestamp}] 识别结果: {text}")
                        self._log_recognition(timestamp, text)
                        
                        return text
            
            # 获取最终结果
            result = json.loads(self.recognizer.FinalResult())
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if result.get('text'):
                timestamp = datetime.datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒")
                text = result['text']
                
                print(f"[{timestamp}] 识别结果: {text}")
                self._log_recognition(timestamp, text)
                
                return text
            else:
                print("未识别到语音内容")
                return None
                
        except Exception as e:
            print(f"录音错误: {e}")
            return None
    
    def recognize_from_file(self, audio_file: str) -> Optional[str]:
        """
        从音频文件识别语音
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            str: 识别到的文本，如果失败返回None
        """
        if not self.recognizer:
            print("识别器未初始化")
            return None
        
        try:
            if not os.path.exists(audio_file):
                print(f"音频文件不存在: {audio_file}")
                return None
            
            print(f"正在处理音频文件: {audio_file}")
            
            # 打开音频文件
            wf = wave.open(audio_file, 'rb')
            
            # 检查音频格式
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print("音频格式不支持，需要: 单声道, 16位, 16kHz")
                wf.close()
                return None
            
            # 读取音频数据
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    if result.get('text'):
                        wf.close()
                        
                        timestamp = datetime.datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒")
                        text = result['text']
                        
                        print(f"[{timestamp}] 识别结果: {text}")
                        self._log_recognition(timestamp, text, audio_file)
                        
                        return text
            
            # 获取最终结果
            result = json.loads(self.recognizer.FinalResult())
            wf.close()
            
            if result.get('text'):
                timestamp = datetime.datetime.now().strftime("%Y年%m月%d日 %H时%M分%S秒")
                text = result['text']
                
                print(f"[{timestamp}] 识别结果: {text}")
                self._log_recognition(timestamp, text, audio_file)
                
                return text
            else:
                print("未识别到语音内容")
                return None
                
        except Exception as e:
            print(f"处理音频文件错误: {e}")
            return None
    
    def record_audio(self, filename: str = None, duration: int = 5) -> str:
        """
        录制音频并保存为文件
        
        Args:
            filename: 保存的文件名，如果为None则自动生成
            duration: 录制时长（秒）
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            save_directory = "/home/zq/catkin_ws/src/cmoon/src/vosk_speech_recognition/audio/"
            os.makedirs(save_directory, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_directory, f"recording_{timestamp}.wav")
            #filename = f"recording_{timestamp}.wav"
        
        try:
            print(f"开始录制音频，时长: {duration}秒")
            print("请说话...")
            
            # 设置音频参数
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            # 初始化PyAudio
            p = pyaudio.PyAudio()
            
            # 打开音频流
            stream = p.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
            
            frames = []
            
            # 录音
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            # 停止录音
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # 保存音频文件
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"音频已保存: {filename}")
            return filename
            
        except Exception as e:
            print(f"录制音频错误: {e}")
            return None
    
    def continuous_recognition(self, callback: Callable[[str], None] = None, stop_event: threading.Event = None):
        """
        连续语音识别
        
        Args:
            callback: 识别结果回调函数
            stop_event: 停止事件
        """
        if not self.recognizer:
            print("识别器未初始化")
            return
        
        print("开始连续语音识别，按Ctrl+C停止")
        
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                
                text = self.recognize_from_microphone(timeout=1, phrase_time_limit=5)
                
                if text and callback:
                    callback(text)
                
                time.sleep(0.1)  # 短暂休息
                
        except KeyboardInterrupt:
            print("\n连续识别已停止")
    
    def _log_recognition(self, timestamp: str, text: str, audio_file: str = None):
        """记录识别结果到文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if audio_file:
                    f.write(f"[{timestamp}] 文件: {audio_file} -> {text}\n")
                else:
                    f.write(f"[{timestamp}] 麦克风 -> {text}\n")
        except Exception as e:
            print(f"记录文件失败: {e}")
    
    def show_log(self):
        """显示识别记录"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("=== 识别记录 ===")
                    print(content)
            else:
                print("记录文件不存在")
        except Exception as e:
            print(f"读取文件失败: {e}")
    
    def clear_log(self):
        """清空记录文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("=== Vosk语音识别记录日志 ===\n")
                f.write(f"创建时间: {datetime.datetime.now().strftime('%Y年%m月%d日 %H时%M分%S秒')}\n")
                f.write("-" * 50 + "\n")
            print(f"已清空记录文件: {self.log_file}")
        except Exception as e:
            print(f"清空文件失败: {e}")
    
    def set_language(self, language: str):
        """设置识别语言"""
        self.language = language
        print(f"语言已设置为: {language}")

# ==================== 便捷函数接口 ====================

def get_recognizer_instance(language='en', model_path=None):
    """获取全局识别器实例，单例模式"""
    global _global_recognizer
    if _global_recognizer is None:
        _global_recognizer = VoskSpeechRecognition(model_path, language)
    return _global_recognizer

def recognize_speech(language='en', model_path=None, timeout=5, phrase_time_limit=10) -> Optional[str]:
    """
    便捷函数：识别语音
    
    Args:
        language: 语言代码，'zh'为中文，'en'为英文
        model_path: 模型路径
        timeout: 等待语音开始的超时时间
        phrase_time_limit: 单次语音的最大时长
        
    Returns:
        str: 识别到的文本
        
    Example:
        >>> text = recognize_speech('zh')
        >>> print(text)
    """
    recognizer = get_recognizer_instance(language, model_path)
    return recognizer.recognize_from_microphone(timeout, phrase_time_limit)

def recognize_from_file(audio_file: str, language='cn', model_path=None) -> Optional[str]:
    """
    便捷函数：从文件识别语音
    
    Args:
        audio_file: 音频文件路径
        language: 语言代码
        model_path: 模型路径
        
    Returns:
        str: 识别到的文本
        
    Example:
        >>> text = recognize_from_file('audio.wav', 'zh')
    """
    recognizer = get_recognizer_instance(language, model_path)
    return recognizer.recognize_from_file(audio_file)

def record_and_recognize(language='en', model_path=None, duration=5) -> Tuple[Optional[str], str]:
    """
    便捷函数：录制并识别语音
    
    Args:
        language: 语言代码
        model_path: 模型路径
        duration: 录制时长
        
    Returns:
        tuple: (识别结果, 音频文件路径)
        
    Example:
        >>> text, audio_file = record_and_recognize('zh', duration=5)
    """
    recognizer = get_recognizer_instance(language, model_path)
    audio_file = recognizer.record_audio(duration=duration)
    if audio_file:
        text = recognizer.recognize_from_file(audio_file)
        return text, audio_file
    return None, None

def show_recognition_log():
    """便捷函数：显示识别记录"""
    recognizer = get_recognizer_instance()
    recognizer.show_log()

def clear_recognition_log():
    """便捷函数：清空识别记录"""
    recognizer = get_recognizer_instance()
    recognizer.clear_log()

# ==================== 命令行接口 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于Vosk的离线语音识别程序')
    parser.add_argument('-l', '--language', choices=['zh', 'en'], 
                       default='zh', help='识别语言 (默认: zh)')
    parser.add_argument('-m', '--model', help='模型路径')
    parser.add_argument('-f', '--file', help='从音频文件识别')
    parser.add_argument('-r', '--record', type=int, help='录制音频时长（秒）')
    parser.add_argument('-c', '--continuous', action='store_true', help='连续识别模式')
    parser.add_argument('-s', '--show', action='store_true', help='显示识别记录')
    parser.add_argument('--clear', action='store_true', help='清空记录文件')
    
    args = parser.parse_args()
    
    # 创建识别器实例
    recognizer = VoskSpeechRecognition(args.model, args.language)
    
    if args.clear:
        recognizer.clear_log()
    elif args.show:
        recognizer.show_log()
    elif args.file:
        text = recognizer.recognize_from_file(args.file)
        if text:
            print(f"识别结果: {text}")
    elif args.record:
        audio_file = recognizer.record_audio(duration=args.record)
        if audio_file:
            text = recognizer.recognize_from_file(audio_file)
            if text:
                print(f"识别结果: {text}")
    elif args.continuous:
        def callback(text):
            print(f"识别到: {text}")
        
        stop_event = threading.Event()
        try:
            recognizer.continuous_recognition(callback, stop_event)
        except KeyboardInterrupt:
            stop_event.set()
    else:
        # 默认单次识别
        text = recognizer.recognize_from_microphone()
        if text:
            print(f"识别结果: {text}")

if __name__ == "__main__":
    get_recognizer_instance('en')
    time.sleep(1)
    while True:
        text0, audio_file = record_and_recognize('en', duration=5)
        print(audio_file)
        print(f"识别结果: {text0}")

