#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Azure Kinect Body Tracking 的居家行为识别器。

识别 4 种静态姿态：站立 / 坐 / 躺 / 摔倒，并在“站立”基础上通过时序检测“挥手”。

【坐标约定】关节坐标单位 mm，Kinect 深度相机坐标系：+X 右，+Y 下，+Z 前。
因此“向上”是 -Y 方向，某关节“离地高度”越大，其 Y 值越小。

【判断依据 —— 为什么能解决高矮/角度问题】
全部用“身体自身的比例/夹角”，而不是某个点的绝对高度：

① 躯干夹角 θ：向量(髋 PELVIS → 颈 NECK) 与竖直方向(0,-1,0)的夹角。
   θ 只取决于躯干是竖直还是躺平，跟人高矮、相机俯仰角都无关。

② 髋-踝高度差 h：|踝Y - 髋Y|，即“腿的竖直投影长度”。
   这是自身比例，不依赖绝对身高。

③ 髋部离地高度 hip_y_ground：|地面Y - 髋Y|，辅助区分“摔倒贴地” vs “躺床”。

结合 θ 和 h 区分：
    站立：θ 小 且 h 大
    坐：  θ 小 且 h 中
    躺：  θ 大 且 h 小（躺床/沙发，身体离地有一定高度）
    摔倒：θ 大 且 h 极小（整个人贴地）
"""

import math

import numpy as np

import _k4abt
from pyKinectAzure import pyKinectAzure


# 关节名 → k4abt 关节 ID 映射
JOINT = {
    "PELVIS": _k4abt.K4ABT_JOINT_PELVIS,
    "SPINE_NAVEL": _k4abt.K4ABT_JOINT_SPINE_NAVEL,
    "NECK": _k4abt.K4ABT_JOINT_NECK,
    "SHOULDER_LEFT": _k4abt.K4ABT_JOINT_SHOULDER_LEFT,
    "ELBOW_LEFT": _k4abt.K4ABT_JOINT_ELBOW_LEFT,
    "WRIST_LEFT": _k4abt.K4ABT_JOINT_WRIST_LEFT,
    "SHOULDER_RIGHT": _k4abt.K4ABT_JOINT_SHOULDER_RIGHT,
    "ELBOW_RIGHT": _k4abt.K4ABT_JOINT_ELBOW_RIGHT,
    "WRIST_RIGHT": _k4abt.K4ABT_JOINT_WRIST_RIGHT,
    "HIP_LEFT": _k4abt.K4ABT_JOINT_HIP_LEFT,
    "KNEE_LEFT": _k4abt.K4ABT_JOINT_KNEE_LEFT,
    "ANKLE_LEFT": _k4abt.K4ABT_JOINT_ANKLE_LEFT,
    "HIP_RIGHT": _k4abt.K4ABT_JOINT_HIP_RIGHT,
    "KNEE_RIGHT": _k4abt.K4ABT_JOINT_KNEE_RIGHT,
    "ANKLE_RIGHT": _k4abt.K4ABT_JOINT_ANKLE_RIGHT,
    "HEAD": _k4abt.K4ABT_JOINT_HEAD,
    "NOSE": _k4abt.K4ABT_JOINT_NOSE,
}

# ===== 需现场标定的参数 =====
# 地面在相机坐标系里的 Y 值(mm)，约等于“相机离地高度”。
# TODO: 写死占位，必须按实际相机安装高度/俯仰角现场标定。
# 标定方法：让一人站直，读其脚踝(ANKLE)Y 值，即为地面 Y 的近似。
GROUND_Y = 1000.0

# 髋部离地高度阈值(米)：低于此值判“摔倒(贴地)”，高于判“躺(床/沙发)”。
HIP_GROUND = 0.25

# 躯干与竖直方向夹角阈值(度)：超过则视为躯干接近水平。
TRUNK_ANGLE_THRESHOLD = 60.0

# 髋-踝高度差阈值(米)：低于则视为“坐”(站立时整条腿竖直投影应更大)。
LEG_HEIGHT_SIT_THRESHOLD = 0.7

# 挥手判定：手腕高于鼻子的最小高度差(mm)。
WAVE_HEIGHT_DIFF = 150.0


class BehaviorDetector:
    def __init__(self, k4abt_lib_path, ground_y=GROUND_Y, hip_ground=HIP_GROUND):
        self.ground_y = ground_y
        self.hip_ground = hip_ground

        # 打开 Azure Kinect 并启动骨架追踪
        self.kinect = pyKinectAzure()  # k4a 库用默认 Linux 路径
        self.kinect.device_open()
        self.kinect.device_start_cameras()
        self.kinect.bodyTracker_start(k4abt_lib_path)  # 传入 libk4abt.so 路径
        self.tracker = self.kinect.body_tracker

    # ---------- 关节读取 ----------
    def _joint(self, body, name):
        """取某个关节的 3D 坐标(mm)，置信度过低返回 None。"""
        j = body.skeleton.joints[JOINT[name]]
        if j.confidence_level < _k4abt.K4ABT_JOINT_CONFIDENCE_MEDIUM:
            return None
        return np.array([j.position.v[0], j.position.v[1], j.position.v[2]], dtype=float)

    def _update(self):
        """取一帧并刷新骨架。"""
        self.kinect.device_get_capture()
        self.kinect.bodyTracker_update()

    def _get_joints(self):
        """取一帧，返回第一个有效人体的关节字典 {关节名: np.array 或 None}。无人/失败返回 None。"""
        self._update()
        bodies = getattr(self.tracker, "bodiesNow", [])
        if not bodies:
            return None
        body = bodies[0]  # 居家场景每房间 1 人，取第一个
        return {name: self._joint(body, name) for name in JOINT}

    # ---------- 单帧姿态分类 ----------
    def _classify_pose(self, j):
        """
        根据单帧骨架判断静态姿态。
        j: {关节名: np.array([x,y,z]) 或 None}
        返回: '站立' / '坐' / '躺' / '摔倒' / None(关节不全)
        """
        pelvis = j.get("PELVIS")
        neck = j.get("NECK")
        ankle_l = j.get("ANKLE_LEFT")
        ankle_r = j.get("ANKLE_RIGHT")
        if pelvis is None or neck is None:
            return None

        # ① 躯干与竖直方向的夹角
        trunk = neck - pelvis
        up = np.array([0.0, -1.0, 0.0])  # 竖直向上（Y 向下坐标系里为 -Y）
        cos_t = np.dot(trunk, up) / (np.linalg.norm(trunk) + 1e-6)
        theta = math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0)))

        # ② 髋-踝高度差（腿竖直投影，米）
        ankle = None
        if ankle_l is not None and ankle_r is not None:
            ankle = (ankle_l + ankle_r) / 2.0
        elif ankle_l is not None:
            ankle = ankle_l
        elif ankle_r is not None:
            ankle = ankle_r
        if ankle is None:
            return None
        h = abs(ankle[1] - pelvis[1]) / 1000.0  # 米

        # ③ 髋部绝对离地高度（辅助区分“摔倒贴地” vs “躺床”）
        hip_y_ground = abs(self.ground_y - pelvis[1]) / 1000.0

        if theta > TRUNK_ANGLE_THRESHOLD:
            # 躯干接近水平
            if hip_y_ground < self.hip_ground:
                return "摔倒"
            return "躺"
        # 躯干竖直：靠腿投影长度分站/坐
        if h < LEG_HEIGHT_SIT_THRESHOLD:
            return "坐"
        return "站立"

    # ---------- 挥手检测(时序) ----------
    def _is_wrist_above_nose(self, j):
        """单帧判断是否有任一手腕明显高于鼻子。"""
        nose = j.get("NOSE")
        if nose is None:
            return False
        for wrist_name in ("WRIST_LEFT", "WRIST_RIGHT"):
            wrist = j.get(wrist_name)
            if wrist is not None and (nose[1] - wrist[1]) > WAVE_HEIGHT_DIFF:
                return True
        return False

    def detect_wave(self, frames=15, hit_threshold=5):
        """连续多帧检测手腕是否高于鼻子，判定挥手。返回 True/False。"""
        hit = 0
        for _ in range(frames):
            j = self._get_joints()
            if j is None:
                continue
            if self._is_wrist_above_nose(j):
                hit += 1
            else:
                hit = 0
            if hit >= hit_threshold:
                return True
        return False

    # ---------- 对外统一入口 ----------
    def recognize(self):
        """
        返回最终行为标签：'站立' / '坐' / '躺' / '摔倒' / '挥手' / None(无人或关节不全)。
        流程：先做静态姿态分类，若为“站立”再检测挥手。
        """
        j = self._get_joints()
        if j is None:
            return None
        pose = self._classify_pose(j)
        if pose == "站立" and self.detect_wave():
            return "挥手"
        return pose

    def close(self):
        """释放 Kinect 设备与骨架追踪器(分时独占，用完必须关)。"""
        try:
            self.tracker.shutdown()
        except Exception:
            pass
        try:
            self.tracker.destroyTracker()
        except Exception:
            pass
        try:
            self.kinect.device_stop_cameras()
        except Exception:
            pass
        try:
            self.kinect.device_close()
        except Exception:
            pass


if __name__ == "__main__":
    import time

    # libk4abt.so 路径按实际环境填（通常与 libk4a.so 同目录）
    det = BehaviorDetector("/usr/lib/x86_64-linux-gnu/libk4abt.so")
    try:
        for _ in range(10):
            label = det.recognize()
            print("行为:", label)
            time.sleep(0.2)
    finally:
        det.close()
