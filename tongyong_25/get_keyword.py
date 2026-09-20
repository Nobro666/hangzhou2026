#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖库需要安装
pip install pypinyin fuzzywuzzy[speedup]

本模块主要解决文字同音转换 如可乐转换成可了 也能捕获关键词 以及对杂音的处理 应对嘈杂环境
param min_confidence 为主要参数 可以调整 默认为0.85 即相似度大于85%时才认为匹配成功

created by zx 2025-10-11
"""

from fuzzywuzzy import fuzz
from pypinyin import pinyin, Style

class FuzzyKeywordMatcher:
    """
    一个基于拼音的模糊关键词匹配器。
    它可以在包含噪声和错别字的文本中，找出预设的关键词。
    """
    def __init__(self, keywords):
        """
        初始化匹配器。
        :param keywords: 一个包含所有目标关键词的列表 (list of strings)。
        """
        if not keywords:
            raise ValueError("关键词列表不能为空")
            
        self.keywords = keywords
        # 预先计算并缓存所有关键词的拼音，提高后续匹配效率
        self.keyword_pinyin_map = {kw: self._to_pinyin(kw) for kw in self.keywords}
        
        # 计算关键词的最小和最大长度，用于优化滑动窗口
        # 减1和加2是为了让窗口更有弹性，能容忍多字或少字的情况
        self.min_len = min(len(kw) for kw in self.keywords) - 1
        self.max_len = max(len(kw) for kw in self.keywords) + 2
        # 确保最小长度不小于1
        if self.min_len < 1:
            self.min_len = 1

    def _to_pinyin(self, text):
        """
        将文本字符串转换为无声调的拼音字符串，用空格分隔。
        例如："可乐" -> "ke le"
        """
        return ' '.join([item[0] for item in pinyin(text, style=Style.NORMAL)])

    # def find_best_match(self, text, min_confidence=0.85):
    #     """
    #     在输入文本中查找最匹配的关键词。
    #     :param text: 语音识别后的原始文本。
    #     :param min_confidence: 最小置信度（0到1之间），低于此分数的匹配将被忽略。
    #     :return: 一个元组 (匹配到的关键词, 相似度分数)，如果没有找到则返回 (None, 0)。
    #     """
    #     best_match_keyword = None
    #     highest_score = 0

    #     # 使用滑动窗口遍历输入文本的所有可能子串
    #     # 窗口大小由关键词的最短和最长长度动态决定
    #     for length in range(self.min_len, self.max_len + 1):
    #         if length > len(text):
    #             continue
            
    #         for i in range(len(text) - length + 1):
    #             substring = text[i:i+length]
    #             substring_pinyin = self._to_pinyin(substring)

    #             # 将子串的拼音与每个标准关键词的拼音进行比较
    #             for keyword, keyword_pinyin in self.keyword_pinyin_map.items():
    #                 # 使用 fuzz.ratio 计算两个字符串的相似度 (0-100)
    #                 score = fuzz.ratio(substring_pinyin, keyword_pinyin) / 100.0
                    
    #                 if score > highest_score:
    #                     highest_score = score
    #                     best_match_keyword = keyword
        
    #     # 只有当最高分超过设定的阈值时，才认为找到了匹配
    #     if highest_score >= min_confidence:
    #         return (best_match_keyword, highest_score)
    #     else:
    #         return (None, 0)
    def find_best_match(self, text, min_confidence=0.7):
        """
        在输入文本中查找最匹配的关键词。
        :param text: 语音识别后的原始文本。
        :param min_confidence: 最小置信度 0到1之间 低于此分数的匹配将被忽略。
        :return: 一个元组 (匹配到的关键词, 相似度分数)，如果没有找到则返回 (None, 0)。
        """
        best_match_keyword = None
        highest_score = 0

        # 在开始匹配前 先移除所有空格
        processed_text = text.replace(" ", "").replace("\n", "")

        
        # 后续的逻辑都使用处理过的 processed_text
        if not processed_text:
            return (None, 0)

        # 使用滑动窗口遍历输入文本的所有可能子串
        # 窗口大小由关键词的最短和最长长度动态决定
        for length in range(self.min_len, self.max_len + 1):
            if length > len(processed_text):
                continue
            
            for i in range(len(processed_text) - length + 1):
                substring = processed_text[i:i+length]
                substring_pinyin = self._to_pinyin(substring)

                # 将子串的拼音与每个标准关键词的拼音进行比较
                for keyword, keyword_pinyin in self.keyword_pinyin_map.items():
                    # 使用 fuzz.ratio 计算两个字符串的相似度 (0-100)
                    score = fuzz.ratio(substring_pinyin, keyword_pinyin) / 100.0
                    
                    if score > highest_score:
                        highest_score = score
                        best_match_keyword = keyword
                    elif score == highest_score and score > 0:   #如果分数相同 用长的关键字替换短的关键字 (主要应对 乐事薯片 )
                        if best_match_keyword is None or len(keyword) > len(best_match_keyword):
                            best_match_keyword = keyword
        
        # 只有当最高分超过设定的阈值时，才认为找到了匹配
        if highest_score >= min_confidence:
            return (best_match_keyword, highest_score)
        else:
            return (None, 0)
# --- 主程序入口与使用示例 ---
if __name__ == '__main__':
    # 1. 定义你的目标关键词列表 此处可根据实际情况进行调整
    target_keywords = [
        "饼干", "乐事薯片", "薯片","曲奇", "洗手液", "洗洁精", 
        "水", "雪碧", "可乐", "芬达", "洗发水"
    ]

    # 2. 初始化关键词匹配器
    matcher = FuzzyKeywordMatcher(keywords=target_keywords)

    # 3. 准备一些测试用的句子
    test_sentences = [
        "那个扒拉啊可了马路旁边有吗",   # 包含噪声和错词
        "我想要一瓶那个雪比",         # 典型错词
        "来一包乐是薯片谢谢",       # 另一个典型错词
        "请帮我找一下洗发水在哪里",   # 正常句子
        "今天天气真不错啊",           # 完全不相关的句子
        "给我拿瓶芬达吧",             # 正常句子
        "这里有曲奇嘛"              # 包含语气词
    ]

    # 4. 遍历测试
    for sentence in test_sentences:
        # 调用 find_best_match 方法进行匹配
        # 你可以调整 min_confidence 参数来改变匹配的严格程度
        # 0.85 是一个比较均衡的推荐值
        keyword, score = matcher.find_best_match(sentence, min_confidence=0.85)
        
        print(f"原始文本: '{sentence}'")
        if keyword:
            print(f"匹配成功: '{keyword}' (相似度: {score:.2%})\n")
        else:
            print(f"未找到匹配项\n")
