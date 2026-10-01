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

# ===== 【需现场需重点标定的参数！！！】 =====
# 地面在相机坐标系里的 Y 值(mm)。标定方法：让一人站直，读其两侧脚踝 Y 均值。
GROUND_Y = 1000.0
# 髋部离地高度阈值(米)：低于此值更像“摔倒贴地”，高于更像“躺床/沙发”。
HIP_GROUND = 0.25
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

# 对外输出只保留这 4 类；站立只是内部中间状态，不作为最终播报动作。
POSE_LYING = "躺下睡觉"
POSE_SITTING = "坐着休息"
POSE_FALLEN = "摔倒"
POSE_WAVING = "挥手"
POSE_STANDING = "站立"


class BehaviorDetector:
    def __init__(self, k4abt_lib_path, ground_y=GROUND_Y, hip_ground=HIP_GROUND):
        self.ground_y = ground_y
        self.hip_ground = hip_ground
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
        hip_y_ground = abs(self.ground_y - pelvis[1]) / 1000.0

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
        """窗口内挥手检测；优先级高于坐/站。"""
        raised_hits = Counter()
        xs = defaultdict(list)
        for j in frames_joints:
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
            # 短时丢跟踪不要立刻 nobody，避免语音/交互抖动。
            if self.last_behavior is not None and self.no_body_count <= NO_BODY_TOLERANCE:
                return self.last_behavior
            return None

        self.no_body_count = 0

        if self._detect_wave_in_window(joints_window):
            self.last_behavior = POSE_WAVING
            return POSE_WAVING

        if not pose_votes:
            return self.last_behavior

        # 多数投票；摔倒属于安全相关，达到 2 帧即可优先。
        counts = Counter(pose_votes)
        if counts[POSE_FALLEN] >= 2:
            result = POSE_FALLEN
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
                    if self._detect_wave_in_window(list(recent_joints)):
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
    # libk4abt.so 路径按实际环境填（通常与 libk4a.so 同目录）
    det = None
    try:
        det = BehaviorDetector("/lib/libk4abt.so")
        det.show_camera()
    finally:
        if det is not None:
            det.close()

