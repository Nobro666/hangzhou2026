#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""测试三张正脸注册、常规识别以及躺姿旋转兜底识别。"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from face_detect_tongyong import Detector


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FACE_DIR = PROJECT_DIR / "face"


def parse_args():
    parser = argparse.ArgumentParser(description="测试K4A三张正脸注册及已知人脸识别")
    parser.add_argument(
        "--face-dir",
        default=str(DEFAULT_FACE_DIR),
        help="注册照片目录，默认使用项目中的face目录",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=2.0,
        help="注册结束到开始识别之间的等待时间",
    )
    parser.add_argument(
        "--lying-wait-seconds",
        type=float,
        default=8.0,
        help="常规识别结束后，留给测试者摆成躺姿的时间",
    )
    return parser.parse_args()


def close_camera(detector):
    """尽力释放K4A和OpenCV资源，便于下一阶段重新打开相机。"""
    try:
        detector.close_k4a()
    except Exception as error:
        print(f"关闭K4A时发生异常：{error}")
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    face_dir = Path(args.face_dir).expanduser().resolve()
    face_dir.mkdir(parents=True, exist_ok=True)

    print("==============人脸注册与识别测试开始==============")
    print(f"人脸目录：{face_dir}")
    print("本测试没有语音播报，请保持正脸朝向K4A相机")

    detector = Detector(str(face_dir))
    registered_id = None

    try:
        print("\n[阶段1] 连续采集三张正脸并注册")
        registered_id = detector.register_new_face()
        if registered_id is None:
            print("人脸注册失败：首次画面未检测到有效人脸")
            return 1

        registered_folder = face_dir / str(registered_id)
        saved_images = sorted(registered_folder.glob("*.jpg"))
        print(f"注册成功：人脸ID={registered_id}")
        print(f"实际保存照片数量：{len(saved_images)}")
        for image_path in saved_images:
            print(f"  - {image_path.name}")

        # 主动关闭再重新打开相机，确保识别使用注册完成后的新画面，
        # 同时验证连续两次使用K4A时资源能够正常释放。
        close_camera(detector)

        wait_seconds = max(0.0, args.wait_seconds)
        print(
            f"\n[阶段2] {wait_seconds:.1f}秒后重新拍照识别人脸，"
            "可以在此期间轻微调整站姿，但请保持正脸"
        )
        time.sleep(wait_seconds)

        detector.detect_known_faces()
        recognized_id = detector.detect_result

        print("\n==============常规识别结果==============")
        print(f"本次注册ID：{registered_id}")
        print(f"本次识别ID：{recognized_id}")

        if recognized_id is None:
            print("常规识别失败：没有识别到已注册人脸")
            normal_test_ok = False
        elif recognized_id != registered_id:
            print("常规识别失败：识别ID与本次注册ID不一致")
            normal_test_ok = False
        else:
            print("常规识别成功：识别结果与本次注册人脸一致")
            normal_test_ok = True

        close_camera(detector)

        lying_wait_seconds = max(0.0, args.lying_wait_seconds)
        print(
            f"\n[阶段3] 请在{lying_wait_seconds:.1f}秒内摆成躺姿，"
            "并让脸部位于K4A画面的下半部分"
        )
        time.sleep(lying_wait_seconds)

        # 重新拍摄躺姿画面，并直接调用主程序所用的兜底算法：
        # 画面下部重叠区域裁剪、放大，再尝试原方向、左右90度和180度。
        lying_image_path = detector.take_photo(detector.device)
        lying_id, lying_distance = detector._detect_lying_face_fallback(
            lying_image_path,
            threshold=0.6,
        )

        print("\n==============躺姿兜底结果==============")
        print(f"本次注册ID：{registered_id}")
        print(f"躺姿识别ID：{lying_id}")
        print(f"躺姿最佳差异：{lying_distance}")

        if lying_id is None:
            print("躺姿兜底失败：没有匹配到已注册人脸")
            lying_test_ok = False
        elif lying_id != registered_id:
            print("躺姿兜底失败：识别ID与本次注册ID不一致")
            lying_test_ok = False
        else:
            print("躺姿兜底成功：识别结果与本次注册人脸一致")
            lying_test_ok = True

        print("\n==============总体测试结果==============")
        print(f"常规识别：{'通过' if normal_test_ok else '失败'}")
        print(f"躺姿兜底：{'通过' if lying_test_ok else '失败'}")
        if normal_test_ok and lying_test_ok:
            print("全部测试通过")
            return 0
        return 2
    except (Exception, SystemExit) as error:
        print(f"测试过程发生异常：{error}")
        return 4
    finally:
        close_camera(detector)


if __name__ == "__main__":
    sys.exit(main())
