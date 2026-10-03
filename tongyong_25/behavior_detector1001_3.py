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

③ 髋部离地高度：用 z_transformer 将骨盆点的相机坐标转成离地高度，
   辅助区分“摔倒贴地” vs “躺床”。

结合 θ 和 h 区分：
    站立：θ 小 且 h 大
    坐：  θ 小 且 h 中
    躺：  θ 大 且 h 小（躺床/沙发，身体离地有一定高度）
    摔倒：θ 大 且 h 极小（整个人贴地）
"""

import math

import cv2
import numpy as np

import _k4abt
from pyKinectAzure import pyKinectAzure
from z_transformer import transform_camera_to_map_z


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

# 髋部离地高度阈值(米)：低于此值判“摔倒(贴地)”，高于判“躺(床/沙发)”。
HIP_GROUND = 0.6

# 躯干与竖直方向夹角阈值(度)：超过则视为躯干接近水平。
TRUNK_ANGLE_THRESHOLD = 60.0

# 髋-踝高度差阈值(米)：低于则视为“坐”(站立时整条腿竖直投影应更大)。
LEG_HEIGHT_SIT_THRESHOLD = 0.7

# 挥手判定：手腕高于同侧肩部的最小高度差(mm)。肩部比鼻子低得多，挥手时手无需抬过头顶。
WAVE_HEIGHT_DIFF = -100.0


class BehaviorDetector:
    def __init__(self, k4abt_lib_path, hip_ground=HIP_GROUND):
        self.hip_ground = hip_ground

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

            # 显示画面需要彩色图，Body Tracker 必须有深度图。
            # 只接收同时包含二者的 Capture，避免把缺少深度的帧送入跟踪器。
            self.kinect.config.synchronized_images_only = True
            self.kinect.device_start_cameras()
            self._cameras_started = True

            self.kinect.bodyTracker_start(k4abt_lib_path)  # 传入 libk4abt.so 路径
            self.tracker = self.kinect.body_tracker
            self._tracker_started = True
        except BaseException:
            # _k4a.VERIFY() 失败时会抛出 SystemExit，因此这里需要捕获
            # BaseException，清理已经成功打开的底层资源后再继续抛出。
            self.close()
            raise

    # ---------- 关节读取 ----------
    def _joint(self, body, name):
        """取某个关节的 3D 坐标(mm)，置信度过低返回 None。"""
        j = body.skeleton.joints[JOINT[name]]
        if j.confidence_level < _k4abt.K4ABT_JOINT_CONFIDENCE_MEDIUM:
            return None
        return np.array([j.position.v[0], j.position.v[1], j.position.v[2]], dtype=float)

    def _update(self, return_color=False, return_depth=False):
        """取一帧并刷新骨架；可同时返回彩色/深度图像副本。"""
        capture_acquired = False
        color_image_handle = None
        color_frame = None
        depth_image_handle = None
        depth_frame = None
        try:
            self.kinect.device_get_capture()
            capture_acquired = True

            if return_color:
                color_image_handle = self.kinect.capture_get_color_image()
                if bool(color_image_handle):
                    # 图像句柄稍后会释放，因此必须复制底层像素数据。
                    color_frame = self.kinect.image_convert_to_numpy(
                        color_image_handle
                    ).copy()

            if return_depth:
                depth_image_handle = self.kinect.capture_get_depth_image()
                if bool(depth_image_handle):
                    depth_frame = self.kinect.image_convert_to_numpy(
                        depth_image_handle
                    ).copy()

            self.kinect.bodyTracker_update()
            return color_frame, depth_frame
        finally:
            if color_image_handle is not None and bool(color_image_handle):
                self.kinect.image_release(color_image_handle)
            if depth_image_handle is not None and bool(depth_image_handle):
                self.kinect.image_release(depth_image_handle)
            # Body Tracker 已经处理完该 Capture 后，归还其底层句柄。
            if capture_acquired:
                self.kinect.capture_release()

    def _get_joints(self, return_color=False, return_depth=False):
        """取一帧；可同时返回彩色/深度画面和第一个有效人体的关节。"""
        frame_acquired = False
        try:
            color_frame, depth_frame = self._update(
                return_color=return_color, return_depth=return_depth
            )
            frame_acquired = True

            bodies = getattr(self.tracker, "bodiesNow", [])
            if not bodies:
                joints = None
            else:
                body = bodies[0]  # 居家场景每房间 1 人，取第一个

                # _joint() 返回的是独立的 NumPy 数组，因此释放 Body Frame
                # 后，下面的关节数据仍然有效。
                joints = {name: self._joint(body, name) for name in JOINT}

            if return_color and return_depth:
                return color_frame, depth_frame, joints
            if return_color:
                return color_frame, joints
            if return_depth:
                return depth_frame, joints
            return joints
        finally:
            if frame_acquired:
                self.tracker.release_frame()

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

        if theta > TRUNK_ANGLE_THRESHOLD:
            # 躯干接近水平：用 z_transformer 算离地高度，区分“摔倒” vs “躺”
            pelvis_m = pelvis / 1000.0  # mm → m
            height = transform_camera_to_map_z([pelvis_m[0], pelvis_m[1], pelvis_m[2]])
            if height < self.hip_ground:
                return "摔倒"
            return "躺"
        # 躯干竖直：靠腿投影长度分站/坐
        if h < LEG_HEIGHT_SIT_THRESHOLD:
            return "坐"
        return "站立"

    # ---------- 挥手检测(时序) ----------
    def _is_wrist_above_shoulder(self, j):
        """单帧判断是否有任一手腕明显高于同侧肩部。"""
        for wrist_name, shoulder_name in (
            ("WRIST_LEFT", "SHOULDER_LEFT"),
            ("WRIST_RIGHT", "SHOULDER_RIGHT"),
        ):
            wrist = j.get(wrist_name)
            shoulder = j.get(shoulder_name)
            if wrist is None or shoulder is None:
                continue
            if (shoulder[1] - wrist[1]) > WAVE_HEIGHT_DIFF:
                return True
        return False

    def detect_wave(self, frames=25, hit_threshold=3):
        """连续多帧检测手腕是否高于同侧肩部，判定挥手。返回 True/False。"""
        hit = 0
        for _ in range(frames):
            j = self._get_joints()
            if j is None:
                continue
            if self._is_wrist_above_shoulder(j):
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
        if pose in ("站立","坐")  and self.detect_wave():
            return "挥手"
        return pose

    def show_camera(self, show_depth=True):
        """
        显示 Kinect 彩色画面和当前静态姿态；按 q 或 Esc 退出。

        Args:
            show_depth: 是否同时显示伪彩色深度图（并排拼接）。
        """
        label_text = {
            "站立": "standing",
            "坐": "sitting",
            "躺": "lying",
            "摔倒": "fallen",
            "挥手": "waving",
            None: "no body",
        }

        print("相机画面已打开，按 q 或 Esc 退出")
        wave_hit = 0
        try:
            while True:
                if show_depth:
                    color_frame, depth_frame, joints = self._get_joints(
                        return_color=True, return_depth=True
                    )
                else:
                    color_frame, joints = self._get_joints(return_color=True)
                    depth_frame = None

                if color_frame is None:
                    continue

                # BGRA 彩色流转换为 OpenCV 使用的 BGR 三通道画面。
                if color_frame.ndim == 3 and color_frame.shape[2] == 4:
                    display_frame = cv2.cvtColor(
                        color_frame, cv2.COLOR_BGRA2BGR
                    )
                else:
                    display_frame = color_frame

                pose = self._classify_pose(joints) if joints is not None else None
                if (
                    pose in ("站立","坐")
                    and joints is not None
                    and self._is_wrist_above_shoulder(joints)
                ):
                    wave_hit += 1
                    if wave_hit >= 3:
                        pose = "挥手"
                else:
                    wave_hit = 0

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

                # ---------- 深度图处理与显示 ----------
                if show_depth and depth_frame is not None:
                    # 深度图是 16-bit(mm)，归一化到 8-bit 后转伪彩色
                    depth_8u = cv2.convertScaleAbs(depth_frame, alpha=0.03)
                    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)

                    # 深度图分辨率通常与彩色图不同，缩放到同高再并排
                    h_color = display_frame.shape[0]
                    h_depth = depth_color.shape[0]
                    if h_depth != h_color:
                        scale = h_color / h_depth
                        depth_color = cv2.resize(
                            depth_color,
                            (int(depth_color.shape[1] * scale), h_color),
                        )

                    # 在深度图上写标题
                    cv2.putText(
                        depth_color,
                        "Depth",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # 并排拼接
                    display_frame = np.hstack((display_frame, depth_color))

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