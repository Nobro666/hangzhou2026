#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
货架抓取。主流程里直接调用，不跑本文件的入口。

    import task2_pick_final
    ok = kinova.catch_huojia()
    ok = kinova.catch_huojia(target="lays_stax_can", layer="mid")

头顶相机只提供一个初步层（low / mid / high）。按这个层去检测，
该层腕部相机 2 次都没认出列表中的物品，就换层：先向上，上面没有了再向下。
画面偏下才在当前层抓。框中心高于抓取线就去上一层。左边框返回 "move_left" 给上层。
抓取带和更靠上的物体同时出现时，优先抓偏下的那个。

返回：
    True         已抓取并回到 home
    False        没抓到，手臂已回 home
    "move_left"  物体在画面左边。手臂停在当前检测位
"""

from __future__ import print_function

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.append(r"/home/zq/catkin_ws/src/cmoon/src")

import rospy

from catch_ground.src.catch import KinovaRobot
from catch_ground.src.realsense_yolo11 import RealSenseYolo11Detector


# =============================================================================
# 调参区：人手改的位姿均为 [x, y, z, tx, ty, tz]，角度单位是度
# =============================================================================

DEFAULT_TARGET = None  # None：不指定，列表中任一物品都抓
DEFAULT_LAYER = "mid"  # 头顶相机给出的初步层
SHELF_TARGETS = [
    "cola_bottle",
    "sprite_bottle",
    "orange_fanta_bottle",
    "baima_dish_soap",
    "lays_stax_can",
    "safeguard_pump_bottle",
    "blue_bowl",
    "spoon",
    "black_green_box",
    "oreo_long_box",
    "green_shampoo_bottle",
]
# 每一层腕部相机最多认这么多次，仍没有结果就换层
DETECT_TRIES = 2
# 框左边碰到这个像素内，返回左移。640x480 下可手调。
BORDER_MARGIN_PX = 40
# 框中心不低于这条线才在当前层抓。640x480 下，画面正中 y=240 去上一层，
# 雪碧框 y=264~480、中心约 372 留在当前层。其他高度按比例换算。
GRASP_CENTER_Y = 320
# 返回给上层代码：物体在左边框，底盘应左移。不要用 if kinova.catch_huojia() 判断成功。
MOVE_LEFT = "move_left"

EULER_GRASP = [81.040, 83.972, 11.606]
EULER_DETECT = [81.040, 83.972, 5.606]

LAYER_ORDER = ["low", "mid", "high"]


def _pose(x, y, z, tx, ty, tz):
    return [float(x), float(y), float(z), float(tx), float(ty), float(tz)]


HOME_POSE = _pose(
    0.2104809731245041,
    -0.25873029232025146,
    0.5095799565315247,
    *EULER_GRASP,
)

SHELF_LAYERS = {
    "low": {
        "name": "低层",
        "detect_pose": _pose(0.319639, -0.023217, 0.431487, *EULER_DETECT),
    },
    "mid": {
        "name": "中层",
        "detect_pose": _pose(0.354045, -0.054974, 0.640461, *EULER_DETECT),
    },
    "high": {
        "name": "高层",
        "detect_pose": _pose(0.366699, -0.085689, 0.861343, *EULER_DETECT),
    },
}

LIFT_DZ = 0.03

HAND_EYE_T = np.array([
    0.020816677729270594,
    0.06788506740074149,
    -0.10464690917725247,
], dtype=np.float64)
HAND_EYE_R = np.array([
    [-0.00846062, 0.99974337, -0.02101445],
    [-0.03355864, 0.02071949, 0.99922196],
    [0.99940094, 0.00915925, 0.03337473],
], dtype=np.float64)

FINGER_OPEN = [7, 7, 7]
FINGER_CLOSE = [85, 85, 85]
FINGER_TIGHT = [100, 100, 100]

MODEL_CANDIDATES = [
    str(Path(__file__).resolve().parent / "best.pt"),
]


def _first_existing(paths):
    for p in paths:
        if Path(p).is_file():
            return p
    return paths[0]


def get_layer_cfg(layer_key):
    if layer_key not in SHELF_LAYERS:
        raise ValueError(
            "未知层 '{}', 可选: {}".format(layer_key, list(SHELF_LAYERS.keys()))
        )
    return SHELF_LAYERS[layer_key]


def layer_above(layer_key):
    idx = LAYER_ORDER.index(layer_key)
    if idx + 1 < len(LAYER_ORDER):
        return LAYER_ORDER[idx + 1]
    return None


def layer_below(layer_key):
    idx = LAYER_ORDER.index(layer_key)
    if idx > 0:
        return LAYER_ORDER[idx - 1]
    return None


def border_side(item):
    """左边框返回 left。框中心在抓取线以下返回 None，当前层抓。再靠上返回 top，去上一层。"""
    if not item.box or not item.image_wh:
        return None
    x1, y1, x2, y2 = item.box
    width, height = item.image_wh
    if x1 <= BORDER_MARGIN_PX:
        return "left"
    center_y = (float(y1) + float(y2)) / 2.0
    grasp_y = height * (float(GRASP_CENTER_Y) / 480.0)
    if center_y >= grasp_y:
        return None
    return "top"


def _prefer(items, target_name):
    """指定了名字且在这组里，就用它；否则用置信度最高的。"""
    if target_name:
        named = [item for item in items if item.name == target_name]
        if named:
            return max(named, key=lambda item: item.conf or 0)
    return max(items, key=lambda item: item.conf or 0)


def search_order(hint_layer):
    """初步层 → 先向上 → 上面没有了再向下。"""
    get_layer_cfg(hint_layer)
    idx = LAYER_ORDER.index(hint_layer)
    order = []
    for i in range(idx, len(LAYER_ORDER)):
        order.append(LAYER_ORDER[i])
    for i in range(idx - 1, -1, -1):
        order.append(LAYER_ORDER[i])
    return order


def _assert_mdeg(pose, name):
    if len(pose) != 6:
        raise ValueError("{} 必须是 6 个数 [x,y,z,tx,ty,tz]，当前长度={}".format(name, len(pose)))


def arm_mdeg(kinova, pose, tag=""):
    _assert_mdeg(pose, tag or "pose")
    if tag:
        print("{} {}".format(tag, pose))
    kinova.arm_run(unit="mdeg", pose_target=list(pose))


def camera_to_ee(cam_x, cam_y, cam_z):
    p_cam = np.array([cam_x, cam_z, cam_y], dtype=np.float64)
    return HAND_EYE_R.dot(p_cam) + HAND_EYE_T


def camera_point_to_tool_pose(cam_x, cam_y, cam_z, detect_pose):
    """相机点加到该层检测位上。角度保持检测位。

    现场手调：X += p_ee[0]-0.09，Y += p_ee[2]+0.22，Z += p_ee[1]-0.29。
    """
    p_ee = camera_to_ee(cam_x, cam_y, cam_z)
    pose = [
        float(detect_pose[0]) + float(p_ee[0] - 0.09),
        float(detect_pose[1]) + float(p_ee[2] + 0.22),
        float(detect_pose[2]) + float(p_ee[1] - 0.29),
        float(detect_pose[3]),
        float(detect_pose[4]),
        float(detect_pose[5]),
    ]
    return p_ee.tolist(), pose


def build_grasp_pose_from_camera(result, detect_pose):
    p_ee, grasp = camera_point_to_tool_pose(result.x, result.y, result.z, detect_pose)
    print("相对末端 = {}".format(p_ee))
    print("前伸位姿 = {}".format(grasp))
    lift = list(grasp)
    lift[2] = grasp[2] + LIFT_DZ
    return grasp, lift


def _fresh_pipeline(detector):
    """detect_targets 每次都会 enable_stream 并在结束时 stop，下一层要换一套管道。"""
    detector.pipeline = rs.pipeline()
    detector.config = rs.config()


def _detect_items(detector):
    """始终看整张列表，才能同时判断中心物体和边框物体。"""
    _fresh_pipeline(detector)
    return detector.detect_all_targets(
        target_items=list(SHELF_TARGETS),
        max_retry=DETECT_TRIES,
        show_window=False,
    )


def detect_on_layer(kinova, detector, layer_key):
    """移到该层检测位，返回这一帧里列表中的全部物体。"""
    cfg = get_layer_cfg(layer_key)
    detect_pose = list(cfg["detect_pose"])
    print("到{}检测位 z={:.3f}".format(cfg["name"], detect_pose[2]))
    arm_mdeg(kinova, detect_pose, tag=cfg["name"])
    found = _detect_items(detector)
    if not found:
        print("{} 未识别到列表中的物品".format(cfg["name"]))
    return found, detect_pose


def grasp_detected(kinova, result, detect_pose):
    """开爪 → 前伸 → 合爪 → 抬起 → home。"""
    grasp_pose, lift_pose = build_grasp_pose_from_camera(result, detect_pose)
    print("开爪")
    kinova.finger_run(finger_target=FINGER_OPEN)
    time.sleep(0.5)
    print("前伸")
    arm_mdeg(kinova, grasp_pose, tag="前伸")
    time.sleep(0.5)
    print("合爪")
    kinova.finger_run(finger_target=FINGER_CLOSE)
    kinova.finger_run(finger_target=FINGER_TIGHT)
    time.sleep(0.5)
    print("抬起")
    arm_mdeg(kinova, lift_pose, tag="抬起")
    print("收回 home")
    arm_mdeg(kinova, HOME_POSE)
    return True


def pick_on_shelf(kinova, target_name, hint_layer):
    """底盘已在货架点位。hint_layer 是头顶相机的初步层。

    返回 True：已经抓住并回到 home。
    返回 False：没抓到，手臂已回 home。上层可去下一个点位。
    返回 MOVE_LEFT：物体在左边框。手臂停在当前检测位，上层应让底盘左移。
    """
    pending = search_order(hint_layer)
    print("初步层={} 尝试顺序={}".format(hint_layer, pending))
    weights = _first_existing(MODEL_CANDIDATES)
    if target_name:
        print("腕部相机优先目标={}，中心没有它时也抓列表中的其他物品。模型={}".format(target_name, weights))
    else:
        print("未指定目标，中心认出列表中任一物品即抓取。模型={}".format(weights))
    detector = RealSenseYolo11Detector(weights=Path(weights))

    seen = set()
    while pending:
        layer_key = pending.pop(0)
        if layer_key in seen:
            continue
        seen.add(layer_key)
        found, detect_pose = detect_on_layer(kinova, detector, layer_key)
        centers = []
        borders = []
        for item in found:
            side = border_side(item)
            if side is None and item.has_pose:
                centers.append(item)
            elif side is not None:
                borders.append(item)
        if centers:
            chosen = _prefer(centers, target_name)
            if borders:
                print("同时有中心和边框物体，优先抓取中心的 {}".format(chosen.name))
            else:
                print("抓取画面中心的 {}".format(chosen.name))
            print(
                "检测到 {} 相机=({:.3f},{:.3f},{:.3f})".format(
                    chosen.name, chosen.x, chosen.y, chosen.z
                )
            )
            return grasp_detected(kinova, chosen, detect_pose)
        if not borders:
            continue
        chosen = _prefer(borders, target_name)
        side = border_side(chosen)
        print("{} 位于{}边框，不抓取".format(chosen.name, {"top": "上", "bottom": "下", "left": "左"}[side]))
        if side == "left":
            print("靠左，手臂停在当前检测位，返回左移")
            return MOVE_LEFT
        nxt = layer_above(layer_key) if side == "top" else layer_below(layer_key)
        if nxt and nxt not in seen:
            print("改去{}层".format(get_layer_cfg(nxt)["name"]))
            pending.insert(0, nxt)
        else:
            print("该方向没有更远的层，按原顺序继续")

    print("没有可抓的中心物体")
    arm_mdeg(kinova, HOME_POSE, tag="home")
    return False


def catch_huojia(self, target=None, layer="mid"):
    """底盘已停在货架前。主流程调用 kinova.catch_huojia()。"""
    return pick_on_shelf(self, target, layer)


KinovaRobot.catch_huojia = catch_huojia


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="货架腕部相机换层抓取")
    parser.add_argument(
        "--layer",
        type=str,
        default=DEFAULT_LAYER,
        choices=list(SHELF_LAYERS.keys()),
        help="头顶相机初步层：low / mid / high",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="指定则优先抓这一个；不传则抓列表中任一物品",
    )
    opt = parser.parse_args()

    rospy.init_node("task2_pick_final", anonymous=True)
    kinova = KinovaRobot("j2n6s300")
    if not hasattr(kinova, "client_arm") or kinova.client_arm is None:
        print("机械臂未连接，请先启动 kinova_bringup")
    else:
        ok = kinova.catch_huojia(target=opt.target, layer=opt.layer)
        if ok == MOVE_LEFT:
            print("左移")
        elif ok is True:
            print("完成")
        else:
            print("失败")
