#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Azure Kinect Body Tracking 的居家行为识别器。

目标对外识别 4 类中文行为：
    1. 躺下睡觉
    2. 坐着休息
    3. 摔倒
    4. 挥手

实际判断逻辑采用“先找人、再在时间窗口内投票”的保守方案：

1) 找人：不再只拿 bodiesNow[0]，而是选择“关键关节可用最多、距离相机最近”的人体；
   同时关节置信度放宽到 LOW，避免 K4A 偶发把脚踝/手腕置信度降到 LOW 后整帧被判无人。

2) 挥手优先：每一帧先看挥手线索，不要求先被判为站立；坐着挥手、站着挥手都会优先输出“挥手”。
   挥手由两部分组成：
      - 手举起：手腕高于肩/鼻；
      - 有摆动：同一只手在窗口内 X 向位移达到阈值。
   如果现场挥手幅度较小，也允许“连续多帧手举起”作为挥手，便于比赛/演示稳定触发。

3) 静态姿态：先判断躯干是否接近水平来区分躺/摔倒；竖直躯干再区分坐/站。
   坐/站不再只靠固定 0.7m 腿高阈值，而是优先用膝关节弯曲角：
      - 站立：膝角接近 180°；
      - 坐姿：膝角明显弯曲，或髋-踝垂直投影相对身体尺度偏短。

4) 输出稳定：recognize() 默认采 20 帧，多数投票；短时间跟踪丢失会复用最近一次有效行为，
   减少“nobody/None”抖动。若连续较长时间无人，才返回 None。

坐标约定：关节坐标单位 mm，Kinect 深度相机坐标系：+X 右，+Y 下，+Z 前；因此向上为 -Y。
"""

import argparse
import math
from collections import Counter, defaultdict, deque

import cv2
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

# ===== 【需现场重点标定的参数！！！】 =====
# K4A 坐标：+Y 向下，+Z 向前。相机如果有俯仰角，则地面在相机坐标系中
# 通常不是一个固定 Y，而是随距离 Z 变化，可近似为：ground_y = a * z + b。
#
# Peilin：
#   1.python behavior_detector.py --calibrate-ground，三次窗口回车采样；返回参数修改到本文件GROUND_Y_BY_Z_A、GROUND_Y_BY_Z_B，并设置 USE_GROUND_Y_BY_Z = True。
#   2.fallen和lying可再改HIP_GROUND，fallen过识别则降低数值使得fallen判定更严苛。
#
# 如果来不及做多点标定，可保持 USE_GROUND_Y_BY_Z=False，只用单点 GROUND_Y；
# 但人离相机远近变化较大时，“躺下睡觉/摔倒”区分会变差。
USE_GROUND_Y_BY_Z = True
GROUND_Y_BY_Z_A = -0.32719578 # 多点标定后填写：ground_y = a * z + b 中的 a
GROUND_Y_BY_Z_B = 1281.368   # 多点标定后填写：ground_y = a * z + b 中的 b

# 单点地面 Y(mm)：USE_GROUND_Y_BY_Z=False 时使用。
# 简易标定：人站直，左右脚踝 Y 的平均值约等于当前位置地面 Y。
GROUND_Y = 1000.0

# 髋部离地高度阈值(米)：低于此值才判“摔倒贴地”，高于判“躺床/沙发”。
# 注意：床上躺容易误判 fallen 时，应调小；地上摔倒漏检时再调大。
# 现场建议范围：0.16~0.22。测得0.25对较低床/沙发偏宽，容易把床上躺判成摔倒。
HIP_GROUND = 0.12
# 躯干与竖直方向夹角阈值(度)：超过则视为躯干接近水平。
TRUNK_ANGLE_THRESHOLD = 58.0
# 坐/站辅助阈值。优先使用膝角，腿高比例只作兜底。
KNEE_SIT_ANGLE = 145.0          # 膝角小于该值，认为腿明显弯曲
LEG_HEIGHT_RATIO_SIT = 1.05     # 兜底阈值：明显小于站立腿长才判坐，避免站立误判 sitting
# 挥手判定阈值(mm)。
# 低位挥手常见于坐姿/老人/近距离场景，所以不强制“手腕必须高过肩或鼻”。
# 判定顺序：高举手直接算候选；低位挥手要求“手腕高于肘部且不低于肩太多”，再结合横向摆动。
HAND_ABOVE_SHOULDER = 20.0      # 手腕高于肩至少 20mm 即认为高举；原 80mm 对低位挥手太苛刻
HAND_ABOVE_NOSE = 20.0          # 或者手腕高于鼻至少 20mm
HAND_ABOVE_ELBOW = 40.0         # 低位挥手：手腕至少高于肘部 40mm
HAND_BELOW_SHOULDER_MAX = 180.0 # 低位挥手：手腕最多可低于肩 180mm
WAVE_X_SWING = 90.0             # 同一只手窗口内横向摆动幅度；原 120mm 容易漏检小幅挥手

# recognize() 的默认时间窗口。
RECOGNIZE_FRAMES = 20
MIN_VALID_FRAMES = 3
WAVE_RAISED_HITS = 3
NO_BODY_TOLERANCE = 8
# 摔倒需要多帧确认。床上躺误判 fallen 时可调大；摔倒反应太慢可调小到 2。
FALL_CONFIRM_FRAMES = 3

# 对外输出只保留这 4 类；站立只是内部中间状态，不作为最终播报动作。
POSE_LYING = "躺下睡觉"
POSE_SITTING = "坐着休息"
POSE_FALLEN = "摔倒"
POSE_WAVING = "挥手"
POSE_STANDING = "站立"


class BehaviorDetector:
    def __init__(
        self,
        k4abt_lib_path,
        ground_y=GROUND_Y,
        hip_ground=HIP_GROUND,
        use_ground_y_by_z=USE_GROUND_Y_BY_Z,
        ground_y_by_z_a=GROUND_Y_BY_Z_A,
        ground_y_by_z_b=GROUND_Y_BY_Z_B,
    ):
        self.ground_y = ground_y
        self.hip_ground = hip_ground
        self.use_ground_y_by_z = use_ground_y_by_z
        self.ground_y_by_z_a = ground_y_by_z_a
        self.ground_y_by_z_b = ground_y_by_z_b
        self.last_behavior = None
        self.no_body_count = 0

        # 记录各阶段是否成功，确保初始化中途失败时也能安全清理。
        self.kinect = None
        self.tracker = None
        self._device_opened = False
        self._cameras_started = False
        self._tracker_started = False

        # 打开 Azure Kinect 并启动骨架追踪
        try:
            self.kinect = pyKinectAzure()  # k4a 库用默认 Linux 路径
            self.kinect.device_open()
            self._device_opened = True

            # Body Tracker 必须有深度图；显示画面需要彩色图。
            # 如果现场经常 capture 不到，可尝试把 synchronized_images_only 设为 False，
            # 但默认仍保持同步，避免彩色画面和骨架不同步。
            self.kinect.config.synchronized_images_only = True
            self.kinect.device_start_cameras()
            self._cameras_started = True

            self.kinect.bodyTracker_start(k4abt_lib_path)  # 传入 libk4abt.so 路径
            self.tracker = self.kinect.body_tracker
            self._tracker_started = True
        except BaseException:
            self.close()
            raise

    # ---------- 关节读取 ----------
    def _joint(self, body, name, min_confidence=None):
        """取某个关节的 3D 坐标(mm)，置信度为 NONE 才丢弃。"""
        if min_confidence is None:
            min_confidence = getattr(_k4abt, "K4ABT_JOINT_CONFIDENCE_LOW", 1)
        j = body.skeleton.joints[JOINT[name]]
        if j.confidence_level < min_confidence:
            return None
        return np.array([j.position.v[0], j.position.v[1], j.position.v[2]], dtype=float)

    def _update(self, return_color=False):
        """取一帧并刷新骨架；需要显示时返回独立的彩色图像副本。"""
        capture_acquired = False
        color_image_handle = None
        color_frame = None
        try:
            self.kinect.device_get_capture()
            capture_acquired = True

            if return_color:
                color_image_handle = self.kinect.capture_get_color_image()
                if bool(color_image_handle):
                    color_frame = self.kinect.image_convert_to_numpy(color_image_handle).copy()

            self.kinect.bodyTracker_update()
            return color_frame
        finally:
            if color_image_handle is not None and bool(color_image_handle):
                self.kinect.image_release(color_image_handle)
            if capture_acquired:
                self.kinect.capture_release()

    @staticmethod
    def _body_score(joints):
        """优先选择关键关节更完整、距离相机更近的人。"""
        important = (
            "PELVIS", "NECK", "HEAD", "SHOULDER_LEFT", "SHOULDER_RIGHT",
            "HIP_LEFT", "HIP_RIGHT", "KNEE_LEFT", "KNEE_RIGHT", "ANKLE_LEFT", "ANKLE_RIGHT",
            "WRIST_LEFT", "WRIST_RIGHT",
        )
        visible = sum(joints.get(name) is not None for name in important)
        pelvis = joints.get("PELVIS")
        z_bonus = 0.0 if pelvis is None else -pelvis[2] / 10000.0
        return visible + z_bonus

    def _get_joints(self, return_color=False):
        """取一帧；可同时返回彩色画面和一个最可信人体的关节。"""
        frame_acquired = False
        try:
            color_frame = self._update(return_color=return_color)
            frame_acquired = True

            bodies = getattr(self.tracker, "bodiesNow", [])
            best_joints = None
            best_score = -1e9
            for body in bodies:
                joints = {name: self._joint(body, name) for name in JOINT}
                score = self._body_score(joints)
                if score > best_score:
                    best_score = score
                    best_joints = joints

            if return_color:
                return color_frame, best_joints
            return best_joints
        finally:
            if frame_acquired:
                self.tracker.release_frame()

    # ---------- 地面标定与地面高度估计 ----------
    def _ground_y_at_z(self, z):
        """返回指定距离 z(mm) 处的地面 Y(mm)。

        USE_GROUND_Y_BY_Z=True 时使用多点标定线 ground_y = a*z+b；
        否则退化为单点 GROUND_Y。
        """
        if self.use_ground_y_by_z and z is not None:
            return self.ground_y_by_z_a * float(z) + self.ground_y_by_z_b
        return self.ground_y

    def _hip_height_from_ground(self, pelvis):
        """根据骨盆点估计髋部离地高度，单位 m。"""
        if pelvis is None:
            return None
        ground_y_here = self._ground_y_at_z(pelvis[2])
        return abs(ground_y_here - pelvis[1]) / 1000.0

    def _ankle_ground_sample(self, j):
        """从一帧骨架中取左右脚踝平均值，返回 (ankle_y, ankle_z)，单位 mm。"""
        if not j:
            return None
        ankle = self._mean_existing(j.get("ANKLE_LEFT"), j.get("ANKLE_RIGHT"))
        if ankle is None:
            return None
        return float(ankle[1]), float(ankle[2])

    def collect_ground_calibration_point(self, frames=45):
        """采集一个站立位置的地面标定点。

        使用方法：人站直不动，左右脚都在地面上，调用本函数。
        函数会连续采若干帧脚踝，取中位数降低抖动，返回 (ankle_y, ankle_z)。
        """
        samples = []
        for _ in range(frames):
            sample = self._ankle_ground_sample(self._get_joints())
            if sample is not None:
                samples.append(sample)
        if len(samples) < max(5, frames // 5):
            return None
        arr = np.array(samples, dtype=float)
        ankle_y = float(np.median(arr[:, 0]))
        ankle_z = float(np.median(arr[:, 1]))
        return ankle_y, ankle_z

    @staticmethod
    def fit_ground_y_by_z(points):
        """由多个 (ankle_y, ankle_z) 标定点拟合 ground_y = a*z+b。"""
        if len(points) < 2:
            raise ValueError("至少需要 2 个标定点，推荐近/中/远 3 个点")
        zs = np.array([p[1] for p in points], dtype=float)
        ys = np.array([p[0] for p in points], dtype=float)
        a, b = np.polyfit(zs, ys, 1)
        return float(a), float(b)

    @staticmethod
    def _to_bgr(color_frame):
        """把 K4A 彩色图转成 OpenCV 可显示的 BGR。"""
        if color_frame is None:
            return None
        if color_frame.ndim == 3 and color_frame.shape[2] == 4:
            return cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        return color_frame

    def _draw_calibration_overlay(self, frame, pos_name, sample, collected, target_frames, message):
        """地面标定窗口叠加英文提示；OpenCV putText 不稳定支持中文。"""
        if frame is None:
            return None
        cv2.putText(frame, "Ground calibration", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Position: {pos_name}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, message, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Collected: {collected}/{target_frames}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        if sample is not None:
            ankle_y, ankle_z = sample
            cv2.putText(frame, f"ankle_y={ankle_y:.1f} mm  ankle_z={ankle_z:.1f} mm", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No stable ankle joints", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Click this window first. SPACE/S/ENTER: sample   R: retry   Q/ESC: quit", (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def collect_ground_calibration_point_visual(self, pos_name, frames=45, window_name="Kinect Ground Calibration"):
        """带实时画面的单点地面标定。

        先显示彩色画面和当前脚踝读数；用户确认人已站稳后按 SPACE/ENTER，
        程序继续采集 frames 帧脚踝点并取中位数，返回 (ankle_y, ankle_z)。
        """
        samples = []
        attempts = 0
        sampling = False
        last_sample = None
        print(f"\n请让人站到【{pos_name}】，看见画面后站稳，按 SPACE 或 Enter 开始采样；Q/Esc 退出。")
        while True:
            color_frame, joints = self._get_joints(return_color=True)
            display_frame = self._to_bgr(color_frame)
            if display_frame is None:
                continue

            sample = self._ankle_ground_sample(joints)
            if sample is not None:
                last_sample = sample

            if sampling:
                attempts += 1
                if sample is not None:
                    samples.append(sample)
                msg = "Sampling... keep standing still"
                # 正常采满 frames 个有效脚踝点就返回；如果脚踝偶发丢失，最多等待 frames*4 帧，
                # 只要已有足够样本也返回，避免看起来“一直不采样”。
                min_samples = max(8, frames // 3)
                if len(samples) >= frames or (attempts >= frames * 4 and len(samples) >= min_samples):
                    arr = np.array(samples, dtype=float)
                    ankle_y = float(np.median(arr[:, 0]))
                    ankle_z = float(np.median(arr[:, 1]))
                    self._draw_calibration_overlay(display_frame, pos_name, (ankle_y, ankle_z), len(samples), frames, "Done")
                    cv2.imshow(window_name, display_frame)
                    cv2.waitKey(300)
                    return ankle_y, ankle_z
            else:
                msg = "Stand still, then press SPACE/ENTER"

            self._draw_calibration_overlay(display_frame, pos_name, last_sample, len(samples), frames, msg)
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                return "QUIT"
            if key in (ord('r'), ord('R')):
                samples = []
                attempts = 0
                sampling = False
            if not sampling and key in (32, 13, 10, ord('s'), ord('S')):
                samples = []
                attempts = 0
                sampling = True

    def interactive_ground_calibration(self, positions=("近处", "中间", "远处"), frames=45):
        """交互式地面标定：打开摄像头画面，按 SPACE/ENTER 采近/中/远脚踝点。"""
        points = []
        print("\n=== 地面多点标定 ===")
        print("要求：相机已固定；人自然站直；两只脚都踩在同一地面；每个位置保持 1~2 秒不动。")
        print("会打开 OpenCV 画面窗口；窗口内按 SPACE/Enter 采样，R 重采，Q/Esc 退出。")
        window_name = "Kinect Ground Calibration"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            for name in positions:
                point = self.collect_ground_calibration_point_visual(name, frames=frames, window_name=window_name)
                if point == "QUIT":
                    print("用户退出标定。")
                    break
                if point is None:
                    print(f"【{name}】采样失败：没有稳定读取到脚踝，请调整站位/光照/遮挡后重试。")
                    continue
                ankle_y, ankle_z = point
                points.append(point)
                print(f"【{name}】采样结果：ankle_y={ankle_y:.1f} mm, ankle_z={ankle_z:.1f} mm")
        finally:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

        if len(points) >= 2:
            a, b = self.fit_ground_y_by_z(points)
            print("\n=== 请把下面 3 行复制回 behavior_detector.py 的常量区 ===")
            print("USE_GROUND_Y_BY_Z = True")
            print(f"GROUND_Y_BY_Z_A = {a:.8f}")
            print(f"GROUND_Y_BY_Z_B = {b:.3f}")
            print("\n校验：ground_y = GROUND_Y_BY_Z_A * z + GROUND_Y_BY_Z_B")
        elif len(points) == 1:
            print("\n只采到 1 个点，不能拟合距离修正。可临时使用单点：")
            print(f"GROUND_Y = {points[0][0]:.1f}")
        else:
            print("\n没有采到有效点。")
        return points

    # ---------- 几何工具 ----------
    @staticmethod
    def _angle(a, b, c):
        """返回 ∠ABC，单位度；任一点缺失时返回 None。"""
        if a is None or b is None or c is None:
            return None
        v1 = a - b
        v2 = c - b
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return None
        cos_t = np.dot(v1, v2) / denom
        return math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0)))

    @staticmethod
    def _mean_existing(*points):
        valid = [p for p in points if p is not None]
        if not valid:
            return None
        return np.mean(valid, axis=0)

    @staticmethod
    def _trunk_angle(pelvis, neck):
        if pelvis is None or neck is None:
            return None
        trunk = neck - pelvis
        up = np.array([0.0, -1.0, 0.0])
        denom = np.linalg.norm(trunk)
        if denom < 1e-6:
            return None
        cos_t = np.dot(trunk, up) / denom
        return math.degrees(math.acos(np.clip(cos_t, -1.0, 1.0)))

    # ---------- 单帧姿态分类 ----------
    def _classify_pose(self, j):
        """
        根据单帧骨架判断静态姿态。
        返回: '坐着休息' / '躺下睡觉' / '摔倒' / '站立' / None(关节不全)
        """
        if not j:
            return None

        pelvis = j.get("PELVIS")
        neck = j.get("NECK")
        if neck is None:
            neck = j.get("SPINE_NAVEL")
        head = j.get("HEAD")
        if head is None:
            head = j.get("NOSE")
        if pelvis is None or neck is None:
            return None

        theta = self._trunk_angle(pelvis, neck)
        trunk_len = np.linalg.norm(neck - pelvis) / 1000.0
        trunk_len = max(trunk_len, 0.25)  # 防止异常骨架导致比例爆炸

        ankle = self._mean_existing(j.get("ANKLE_LEFT"), j.get("ANKLE_RIGHT"))
        hip_y_ground = self._hip_height_from_ground(pelvis)
        if hip_y_ground is None:
            return None

        # 躯干接近水平：先判躺/摔倒。
        if theta is not None and theta > TRUNK_ANGLE_THRESHOLD:
            if hip_y_ground < self.hip_ground:
                return POSE_FALLEN
            return POSE_LYING

        # 如果头、髋、脚整体 Y 向展开很小，也说明人基本横着（脚踝偶发缺失时的兜底）。
        body_points = [p for p in (head, neck, pelvis, ankle) if p is not None]
        if len(body_points) >= 3:
            arr = np.vstack(body_points)
            y_span = float(arr[:, 1].max() - arr[:, 1].min()) / 1000.0
            xz_span = max(
                float(arr[:, 0].max() - arr[:, 0].min()),
                float(arr[:, 2].max() - arr[:, 2].min()),
            ) / 1000.0
            if y_span < 0.65 and xz_span > 0.55:
                return POSE_FALLEN if hip_y_ground < self.hip_ground else POSE_LYING

        # 竖直躯干：坐/站。优先用膝角，避免“站着被 0.7m 阈值误判为 sitting”。
        knee_angles = []
        for side in ("LEFT", "RIGHT"):
            hip = j.get(f"HIP_{side}")
            if hip is None:
                hip = pelvis
            knee = j.get(f"KNEE_{side}")
            ankle_side = j.get(f"ANKLE_{side}")
            angle = self._angle(hip, knee, ankle_side)
            if angle is not None:
                knee_angles.append(angle)
        if knee_angles and min(knee_angles) < KNEE_SIT_ANGLE:
            return POSE_SITTING

        if ankle is not None:
            leg_h = abs(ankle[1] - pelvis[1]) / 1000.0
            if leg_h / trunk_len < LEG_HEIGHT_RATIO_SIT:
                return POSE_SITTING

        return POSE_STANDING

    # ---------- 挥手检测 ----------
    def _raised_wrist_sides(self, j):
        """返回当前帧中举起的手：['LEFT', 'RIGHT']。"""
        if not j:
            return []
        nose = j.get("NOSE")
        if nose is None:
            nose = j.get("HEAD")
        sides = []
        for side in ("LEFT", "RIGHT"):
            wrist = j.get(f"WRIST_{side}")
            shoulder = j.get(f"SHOULDER_{side}")
            elbow = j.get(f"ELBOW_{side}")
            if wrist is None:
                continue
            above_shoulder = shoulder is not None and (shoulder[1] - wrist[1]) > HAND_ABOVE_SHOULDER
            above_nose = nose is not None and (nose[1] - wrist[1]) > HAND_ABOVE_NOSE
            # 低位挥手：手腕不一定高过肩，但通常会高于肘部，并且不会垂在身体下方。
            # Kinect 坐标 Y 向下，所以 wrist[1] - shoulder[1] 越大表示手越低。
            low_wave_pose = (
                elbow is not None
                and shoulder is not None
                and (elbow[1] - wrist[1]) > HAND_ABOVE_ELBOW
                and (wrist[1] - shoulder[1]) < HAND_BELOW_SHOULDER_MAX
            )
            if above_shoulder or above_nose or low_wave_pose:
                sides.append(side)
        return sides

    def _is_wrist_above_nose(self, j):
        """兼容旧接口：单帧是否有手举起。"""
        return bool(self._raised_wrist_sides(j))

    def _detect_wave_in_window(self, frames_joints):
        """窗口内挥手检测；优先级只高于站/坐，不覆盖躺/摔倒。

        躺下或摔倒时，手腕/肘部相对肩部的几何关系很容易满足“低位挥手”条件，
        但这不是真正挥手。因此这里会跳过已被静态姿态判为躺/摔倒的帧。
        """
        raised_hits = Counter()
        xs = defaultdict(list)
        for j in frames_joints:
            pose = self._classify_pose(j)
            if pose in (POSE_LYING, POSE_FALLEN):
                continue
            for side in self._raised_wrist_sides(j):
                raised_hits[side] += 1
                wrist = j.get(f"WRIST_{side}")
                if wrist is not None:
                    xs[side].append(float(wrist[0]))

        for side in ("LEFT", "RIGHT"):
            if raised_hits[side] >= WAVE_RAISED_HITS:
                if len(xs[side]) >= 2 and (max(xs[side]) - min(xs[side])) >= WAVE_X_SWING:
                    return True
                # 现场演示中常见“举手挥动幅度不大/被平滑”，连续举手也视为挥手。
                if raised_hits[side] >= max(WAVE_RAISED_HITS + 2, len(frames_joints) // 4):
                    return True
        return False

    def detect_wave(self, frames=25, hit_threshold=3):
        """连续多帧检测挥手。保留旧接口。"""
        joints_window = []
        for _ in range(frames):
            j = self._get_joints()
            if j is not None:
                joints_window.append(j)
        if not joints_window:
            return False
        old_threshold = globals().get("WAVE_RAISED_HITS", WAVE_RAISED_HITS)
        try:
            globals()["WAVE_RAISED_HITS"] = hit_threshold
            return self._detect_wave_in_window(joints_window)
        finally:
            globals()["WAVE_RAISED_HITS"] = old_threshold

    # ---------- 对外统一入口 ----------
    def recognize(self, frames=RECOGNIZE_FRAMES):
        """
        返回最终中文行为标签：'躺下睡觉' / '坐着休息' / '摔倒' / '挥手' / None。
        注意：挥手优先级最高；站立未挥手不是目标行为，返回最近稳定行为或 None。
        """
        joints_window = []
        pose_votes = []
        for _ in range(frames):
            j = self._get_joints()
            if j is None:
                continue
            joints_window.append(j)
            pose = self._classify_pose(j)
            if pose is not None:
                pose_votes.append(pose)

        if len(joints_window) < MIN_VALID_FRAMES:
            self.no_body_count += 1
            # 短时丢跟踪不要立刻 nobody；但不要复用“挥手”，否则人躺下/丢骨架后容易一直显示 waving。
            if (
                self.last_behavior is not None
                and self.last_behavior != POSE_WAVING
                and self.no_body_count <= NO_BODY_TOLERANCE
            ):
                return self.last_behavior
            return None

        self.no_body_count = 0

        if not pose_votes:
            return None if self.last_behavior == POSE_WAVING else self.last_behavior

        # 多数投票。安全/水平姿态优先于挥手：挥手只覆盖站/坐，不覆盖躺/摔倒。
        counts = Counter(pose_votes)
        if counts[POSE_FALLEN] >= FALL_CONFIRM_FRAMES:
            result = POSE_FALLEN
        elif counts[POSE_LYING] >= max(2, len(pose_votes) // 3):
            result = POSE_LYING
        elif self._detect_wave_in_window(joints_window):
            self.last_behavior = POSE_WAVING
            return POSE_WAVING
        else:
            result = counts.most_common(1)[0][0]

        # 站立不是四类服务动作之一，但作为调试/上层状态应正常返回；
        # 语音播报层会选择不播报“站立”。
        if result == POSE_STANDING:
            return POSE_STANDING

        self.last_behavior = result
        return result

    @staticmethod
    def announcement_text(behavior):
        """语音播报文本：例如 '识别到主人挥手'。"""
        if behavior is None or behavior == POSE_STANDING:
            return None
        return f"识别到主人{behavior}"

    def recognize_and_announce(self, speaker=None, frames=RECOGNIZE_FRAMES):
        """
        识别并通过语音接口播报。speaker 可为：
          - 具有 speak(text) 方法的对象；
          - 可直接调用的函数 speaker(text)。
        返回识别出的中文行为标签或 None。
        """
        behavior = self.recognize(frames=frames)
        text = self.announcement_text(behavior)
        if text and speaker is not None:
            if hasattr(speaker, "speak"):
                speaker.speak(text)
            else:
                speaker(text)
        return behavior

    def show_camera(self):
        """显示 Kinect 彩色画面和当前姿态；按 q 或 Esc 退出。"""
        label_text = {
            POSE_STANDING: "standing",
            POSE_SITTING: "sitting/resting",
            POSE_LYING: "lying/sleeping",
            POSE_FALLEN: "fallen",
            POSE_WAVING: "waving",
            None: "no body",
        }

        print("相机画面已打开，按 q 或 Esc 退出")
        recent_joints = deque(maxlen=RECOGNIZE_FRAMES)
        try:
            while True:
                color_frame, joints = self._get_joints(return_color=True)
                if color_frame is None:
                    continue

                if color_frame.ndim == 3 and color_frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
                else:
                    display_frame = color_frame

                pose = self._classify_pose(joints) if joints is not None else None
                if joints is not None:
                    recent_joints.append(joints)
                    # 显示逻辑同 recognize：挥手只覆盖站/坐，不覆盖躺/摔倒。
                    if pose not in (POSE_LYING, POSE_FALLEN) and self._detect_wave_in_window(list(recent_joints)):
                        pose = POSE_WAVING

                cv2.putText(
                    display_frame,
                    "Behavior: " + label_text.get(pose, str(pose)),
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Kinect Behavior Detector", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()

    def close(self):
        """释放 Kinect 设备与骨架追踪器(分时独占，用完必须关)。"""
        if self._tracker_started and self.tracker is not None:
            try:
                self.tracker.shutdown()
            except Exception:
                pass
            try:
                self.tracker.destroyTracker()
            except Exception:
                pass
            self._tracker_started = False
            self.tracker = None

        if self._cameras_started and self.kinect is not None:
            try:
                self.kinect.device_stop_cameras()
            except Exception:
                pass
            self._cameras_started = False

        if self._device_opened and self.kinect is not None:
            try:
                self.kinect.device_close()
            except Exception:
                pass
            self._device_opened = False

        self.kinect = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K4A 人体姿态/行为识别与地面标定工具")
    parser.add_argument("--k4abt", default="/lib/libk4abt.so", help="libk4abt.so 路径")
    parser.add_argument("--calibrate-ground", action="store_true", help="采集近/中/远脚踝点，拟合 ground_y=a*z+b")
    parser.add_argument("--calib-frames", type=int, default=45, help="每个标定点采样帧数")
    args = parser.parse_args()

    det = None
    try:
        det = BehaviorDetector(args.k4abt)
        if args.calibrate_ground:
            det.interactive_ground_calibration(frames=args.calib_frames)
        else:
            det.show_camera()
    finally:
        if det is not None:
            det.close()

