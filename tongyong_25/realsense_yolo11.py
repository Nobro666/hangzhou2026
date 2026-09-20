#!/usr/bin/env python
# coding: UTF-8

"""
  @Auther: Wang Zhe
  @Date: 2025-9-24
  @Version: 1.0
"""

import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from numpy import random
import time
import pyrealsense2 as rs
from ultralytics import YOLO
from ultralytics.utils import ops
from typing import Union


class YoloResult:
    def __init__(self, name: str, x: float, y: float, z: float, angle: float):
        self.name = name
        self.x = x  
        self.y = y  
        self.z = z  
        self.angle = angle  

    def __str__(self):
        return f"物品:{self.name} | 坐标(X,Y,Z):({self.x:.3f},{self.y:.3f},{self.z:.3f}) | 角度:{self.angle:.1f}°"

class RealSenseYolo11Detector:
    def __init__(self, weights: Path = Path("weights"), 
                 imgsz: int = 640, conf_thres: float = 0.2, iou_thres: float = 0.45):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(weights)).to(self.device)

        print("=== 调试:model.names 信息 ===")
        print(f"类型：{type(self.model.names)}")
        print(f"内容：{self.model.names}")
        print("=============================")

        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)  
        
        self.colors = {name: [random.randint(0, 255) for _ in range(3)] for name in self.model.names.values()}

    def _calc_item_angle(self, item_img: np.ndarray) -> float:
        """计算物品最小外接矩形角度"""
        if item_img.size == 0:
            return 0.0
        gray = cv2.cvtColor(item_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 过滤小轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        if not valid_contours:
            return 0.0
        
        # 计算最小外接矩形角度
        all_points = np.vstack(valid_contours)
        rect = cv2.minAreaRect(all_points)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90
        return round(90 - angle, 1)  # 转换为与垂直轴的角度

    def _get_realsense_data(self) -> tuple:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        color_img = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        return color_img, depth_frame, depth_intrin

    def detect_targets(self, target_items: list, max_retry: int = 5, show_window: bool = True) -> Union[YoloResult, None]:
        # 配置RealSense流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)

        retry_count = 0
        try:
            while retry_count < max_retry:
                retry_count += 1
                print(f"第{retry_count}次识别...")

                # 获取RealSense数据
                color_img, depth_frame, depth_intrin = self._get_realsense_data()
                if color_img is None:
                    time.sleep(0.5)
                    continue

                display_img = color_img.copy()

                name_to_idx = {name: idx for idx, name in self.model.names.items()}
                target_idxs = [name_to_idx[item] for item in target_items if item in name_to_idx]
                results = self.model(
                    color_img, imgsz=self.imgsz, conf=self.conf_thres, 
                    iou=self.iou_thres, classes=target_idxs if target_idxs else None,
                    device=self.device, verbose=False
                )
                found_target = None
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()  # 边界框（x1,y1,x2,y2）
                        cls_name = self.model.names[int(box.cls[0])]
                        conf = box.conf[0]
                        

                        print(f"检测到物品：{cls_name} | 置信度：{conf:.2f} | 边界框：{[round(x) for x in xyxy]}")

                        x1, y1, x2, y2 = map(int, xyxy)
                        color = self.colors.get(cls_name, [0, 255, 0])  
                        
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                        label = f"{cls_name} {conf:.2f}"
                        cv2.putText(display_img, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        if cls_name not in target_items or conf < self.conf_thres:
                            continue

                        if cls_name in ["Handwash", "Shampoo"]:
                            center_x = int((x2 - x1)/2 + x1)
                            center_y = int((y2 - y1)*2/3 + y1)
                        else:
                            center_x = int((x1 + x2)/2)
                            center_y = int((y1 + y2)/2)

                        # 绘制中心点
                        cv2.circle(display_img, (center_x, center_y), 5, [0, 0, 255], -1)  
                        
                        depth = depth_frame.get_distance(center_x, center_y)
                        if depth < 0.1 or depth > 5.0:  # 过滤异常深度
                            continue
                        cam_x, cam_y, cam_z = rs.rs2_deproject_pixel_to_point(depth_intrin, [center_x, center_y], depth)

                        coord_text = f"X:{cam_x:.2f}m, Y:{cam_y:.2f}m, Z:{cam_z:.2f}m"
                        cv2.putText(display_img, coord_text, (x1, y2 + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 2)

                        item_img = color_img[y1:y2, x1:x2]
                        item_angle = self._calc_item_angle(item_img)
                        
                        angle_text = f"Angle:{item_angle:.1f}°"
                        cv2.putText(display_img, angle_text, (x1, y2 + 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255], 2)

                        found_target = YoloResult(cls_name, cam_x, cam_y, cam_z, item_angle)

                if show_window:
                    cv2.putText(display_img, f"Retry: {retry_count}/{max_retry}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2)

                    target_text = f"Targets: {', '.join(target_items)}"
                    cv2.putText(display_img, target_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 255, 0], 2)
                    
                    cv2.imshow("YOLO Detection (Press 'q' to quit)", display_img)
                    
                    # 检查按键，按q退出
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("用户手动退出检测")
                        return None

                if found_target:
                    if show_window:
                        # 额外显示1秒找到的结果
                        cv2.imshow("YOLO Detection (Found Target)", display_img)
                        cv2.waitKey(1000)
                    return found_target

                time.sleep(0.8)  

            print(f"已重试{max_retry}次，未识别到指定物品")
            return None

        finally:
            # 确保资源释放
            self.pipeline.stop()
            if show_window:
                cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = RealSenseYolo11Detector(weights=Path('home/cqr/catkin_ws/src/test_1/model/yolo11m.pt'))
    result = detector.detect_targets(target_items=["bottle","spoon","orange"])
    if result:
        print(f"坐标:X={result.x:.3f}, Y={result.y:.3f}, Z={result.z:.3f}")
        print(f"角度：{result.angle:.1f}°")
    else:
        print("未检测到目标物品")
