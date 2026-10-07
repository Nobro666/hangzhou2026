#!/usr/bin/env python3
"""
多相机目标检测封装"""

import numpy as np
import cv2
import torch
import time
import os
from abc import ABC, abstractmethod
from ultralytics import YOLO
import pyrealsense2 as rs
import pykinect_azure as pykinect
from pykinect_azure import (
    K4A_CALIBRATION_TYPE_COLOR, 
    K4A_CALIBRATION_TYPE_DEPTH,
    K4A_FRAMES_PER_SECOND_30,
    K4A_WIRED_SYNC_MODE_STANDALONE
)
import sys
sys.path.append('/home/zq/catkin_ws/src/cmoon/src')
from base_controller import Base
import time

class YoloResult:
    def __init__(self, name, box, x, y, conf) -> None:
        self.name = name
        self.box = box
        self.x = x
        self.y = y
        self.conf = conf

    def __str__(self):
        return f'name:{self.name},box:{self.box},x:{self.x},y:{self.y}'

class Camera(ABC):
    """Abstract base class for all camera types"""
    
    @abstractmethod
    def get_frame(self):
        """Get color frame from camera"""
        pass
        
    @abstractmethod
    def get_depth(self):
        """Get depth frame from camera"""
        pass

    def get_rgbd(self):
        """Get color and depth frames; camera subclasses should synchronize them."""
        ret, color_frame = self.get_frame()
        retd, depth_image = self.get_depth()
        return ret, color_frame, retd, depth_image
        
    @abstractmethod
    def get_calibration(self):
        """Get camera calibration matrix"""
        pass
        
    @abstractmethod
    def release(self):
        """Release camera resources"""
        pass

class KinectCamera(Camera):
    """Implementation for Azure Kinect camera"""
    
    def __init__(self):
        self.K = np.array([915.0828247070312, 0.000000, 961.7936401367188,
                          0.000000, 914.6190185546875, 555.453369140625,
                          0.000000, 0.000000, 1.000000]).reshape(3,3)
    def open_camera(self):
        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P  # 1080P:1920x1080, 720P:1280x720
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = K4A_FRAMES_PER_SECOND_30
        device_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE
        device_config.synchronized_images_only = True
        self.device = pykinect.start_device(config=device_config)

    def get_frame(self):
        capture = self.device.update()
        ret, color_frame = capture.get_color_image()
        return ret, color_frame
        
    def get_depth(self):
        capture = self.device.update()
        ret, depth_image = capture.get_transformed_depth_image()
        return ret, depth_image

    def get_rgbd(self):
        """从同一个K4A Capture取得彩色图和对齐到彩色相机的深度图。"""
        capture = self.device.update()
        ret, color_frame = capture.get_color_image()
        retd, depth_image = capture.get_transformed_depth_image()
        return ret, color_frame, retd, depth_image
        
    def get_calibration(self):
        return self.K
        
    def release(self):
        self.device.stop_cameras()
        self.device.close()

class RealSenseCamera(Camera):
    """Implementation for Intel RealSense camera"""
    
    def __init__(self):
        pass
    
    def open_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        profile = self.pipeline.get_active_profile()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([intr.fx, 0, intr.ppx,
                          0, intr.fy, intr.ppy,
                          0, 0, 1]).reshape(3,3)
        
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())
        
    def get_depth(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame:
            return False, None
        return True, np.asanyarray(depth_frame.get_data())

    def get_rgbd(self):
        """从同一组对齐后的RealSense帧取得彩色图和深度图。"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return False, None, False, None
        return (
            True,
            np.asanyarray(color_frame.get_data()),
            True,
            np.asanyarray(depth_frame.get_data()),
        )
        
    def get_calibration(self):
        return self.K
        
    def release(self):
        self.pipeline.stop()

class WebCamera(Camera):
    """Implementation for standard web camera"""
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.K = np.array([600, 0, 320,
                          0, 600, 240,
                          0, 0, 1]).reshape(3,3)
        
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
        
    def get_depth(self):
        return False, None
        
    def get_calibration(self):
        return self.K
        
    def release(self):
        self.cap.release()

class ItemsDetector:
    """Core detection class with YOLO model"""
    
    def __init__(self, model_path='/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/model/yolo11m.pt'):
        self.model = YOLO(model_path)
        print(f"内容：{self.model.names}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # 仅保存达到候选条件的画面，便于比赛后复查误检目标。
        self.candidate_log_dir = os.environ.get(
            "TRASH_DETECTION_LOG_DIR",
            os.path.expanduser("~/.ros/trash_detection_candidates"),
        )
        try:
            os.makedirs(self.candidate_log_dir, exist_ok=True)
        except OSError as error:
            print(f"创建垃圾检测日志目录失败：{error}")
            self.candidate_log_dir = None

    def get_target_distance(self, depth_image, x, y, box=None):
        """获取目标距离；优先使用检测框中央区域的有效深度中位数。"""
        if depth_image is None:
            return None
        try:
            height, width = depth_image.shape[:2]
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                box_width = max(1, x2 - x1)
                box_height = max(1, y2 - y1)

                # 使用框中央50%的区域，减少背景、地板和框边缘噪声。
                roi_x1 = max(0, x1 + box_width // 4)
                roi_x2 = min(width, x2 - box_width // 4)
                roi_y1 = max(0, y1 + box_height // 4)
                roi_y2 = min(height, y2 - box_height // 4)
                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                    roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
                    valid_depth = roi[np.isfinite(roi) & (roi > 0)]
                    if valid_depth.size:
                        return float(np.median(valid_depth)) * 0.001

            if 0 <= y < height and 0 <= x < width:
                distance = float(depth_image[int(y), int(x)]) * 0.001
                return distance if distance > 0 else None
            return None
        except (IndexError, TypeError, ValueError):
            return None

    def boxes_are_close(self, previous_box, current_box, width, height):
        """判断连续两帧的检测框是否属于画面中的相近位置。"""
        if previous_box is None:
            return False

        def box_center(box):
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        previous_center = box_center(previous_box)
        current_center = box_center(current_box)
        center_distance = np.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        max_center_distance = np.hypot(width, height) * 0.08

        previous_area = max(1.0, (
            (previous_box[2] - previous_box[0]) *
            (previous_box[3] - previous_box[1])
        ))
        current_area = max(1.0, (
            (current_box[2] - current_box[0]) *
            (current_box[3] - current_box[1])
        ))
        area_ratio = current_area / previous_area
        return center_distance <= max_center_distance and 0.5 <= area_ratio <= 2.0

    def save_candidate_detection(self, frame, result, distance, streak, status):
        """保存候选检测框、置信度、深度和连续确认次数。"""
        if self.candidate_log_dir is None:
            return

        annotated = frame.copy()
        x1, y1, x2, y2 = map(int, result.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        distance_text = "invalid" if distance is None else f"{distance:.2f}m"
        label = (
            f"{result.name} conf={result.conf:.2f} "
            f"depth={distance_text} streak={streak} {status}"
        )
        cv2.putText(
            annotated,
            label,
            (max(0, x1), max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{timestamp}_{time.time_ns() % 1000000000:09d}_"
            f"{result.name}_{result.conf:.2f}_{status}.jpg"
        )
        path = os.path.join(self.candidate_log_dir, filename)
        if cv2.imwrite(path, annotated):
            print(f"已保存垃圾候选检测画面：{path}")
        else:
            print(f"保存垃圾候选检测画面失败：{path}")

    def detect(self, camera, target='person', max_distance=None, depth=True,
               timeout=8):
        """
        检测指定目标并返回其三维坐标，参考detect_people.py的返回格式
        参数:
            camera: 相机对象
            target: 目标物品名称
            max_distance: 最大检测距离(米)，None表示不限制
            depth: 是否使用深度信息
            timeout: 本次检测的最大等待时间（秒）
        返回:
            (has_target, 3d_coords)
            has_target: 布尔值，表示是否检测到目标
            3d_coords: 三维坐标元组(x, y, z)，若未检测到则为(0, 0, 0)
        """
        start_time = time.time()
        required_consecutive_frames = 3
        consecutive_frames = 0
        previous_box = None
        confirmed_points = []
        while time.time() - start_time < timeout:
            if depth:
                ret, color_frame, retd, depth_image = camera.get_rgbd()
                if not ret or not retd:
                    continue
            else:
                ret, color_frame = camera.get_frame()
                if not ret:
                    continue
                depth_image = None

            # 执行检测
            self.color_frame = color_frame
            yoloresults = self.pred()
            
            K = camera.get_calibration()
            height, width = color_frame.shape[:2]

            candidates = [
                result for result in yoloresults
                if result.name == target
                and result.conf >= 0.5
                and self.judge_range(result.x, width, 1.0)
            ]
            if not candidates:
                consecutive_frames = 0
                previous_box = None
                confirmed_points = []
                continue

            # 同一帧有多个同类目标时，先跟踪置信度最高的检测框。
            result = max(candidates, key=lambda item: item.conf)
            if depth:
                distance = self.get_target_distance(
                    depth_image,
                    result.x,
                    result.y,
                    box=result.box,
                )
                if not distance or (
                        max_distance is not None and distance > max_distance):
                    self.save_candidate_detection(
                        color_frame,
                        result,
                        distance,
                        0,
                        "invalid_depth",
                    )
                    consecutive_frames = 0
                    previous_box = None
                    confirmed_points = []
                    continue
            else:
                distance = 0.0

            if self.boxes_are_close(previous_box, result.box, width, height):
                consecutive_frames += 1
            else:
                consecutive_frames = 1
                confirmed_points = []
            previous_box = list(result.box)

            z = distance
            point_image = np.array([result.x, result.y, 1])
            point_3d = z * np.linalg.inv(K).dot(point_image)
            confirmed_points.append(point_3d)
            if len(confirmed_points) > required_consecutive_frames:
                confirmed_points.pop(0)

            status = (
                "confirmed"
                if consecutive_frames >= required_consecutive_frames
                else "candidate"
            )
            self.save_candidate_detection(
                color_frame,
                result,
                distance,
                consecutive_frames,
                status,
            )

            if consecutive_frames >= required_consecutive_frames:
                stable_point = np.median(np.asarray(confirmed_points), axis=0)
                print(
                    f"{target}连续{required_consecutive_frames}帧确认成功，"
                    f"置信度={result.conf:.2f}，距离={distance:.2f}m"
                )
                return (
                    True,
                    (stable_point[0], stable_point[1], stable_point[2]),
                )

            if cv2.waitKey(10) in [ord('q'), 27]:
                break

        return (False, (0.0, 0.0, 0.0))
    
    def get_object_classes_sorted(self, camera, range=0.8, visualize=True):
        """
        获取所有识别到的物品种类，按从左到右顺序排列，支持可视化
        
        参数:
            camera: 相机设备对象
            range: 有效检测范围（0-1）
            visualize: 是否显示可视化结果
            
        返回:
            排序后的物品种类列表（字符串列表），按从左到右顺序排列
        """
        classes = []
        ret, color_frame = camera.get_frame()
        if not ret:
            return classes
        
        # 执行检测
        self.color_frame = color_frame.copy()
        yoloresults = self.pred()
        
        height, width = color_frame.shape[:2]
        
        # 筛选有效目标并收集信息
        valid_objects = []
        for result in yoloresults:
            if self.judge_range(result.x, width, range) and result.conf > 0.5:
                valid_objects.append({
                    'class': result.name,
                    'x': result.x,
                    'y': result.y,
                    'box': result.box,
                    'conf': result.conf
                })
        
        # 按x坐标（水平位置）从左到右排序
        valid_objects.sort(key=lambda x: x['x'])
        
        # 提取排序后的种类列表
        classes = [obj['class'].capitalize() for obj in valid_objects]
        
        # 可视化处理
        if visualize:
            self.visualize_sorted_objects(color_frame, valid_objects, width, range)
            
        return classes   
    
    def visualize_sorted_objects(self, frame, objects, width, range):
        """可视化排序后的目标对象"""
        # 绘制有效范围线
        height = frame.shape[0]
        cv2.line(frame, 
                (int(width * 0.5 * (1 - range)), 0),
                (int(width * 0.5 * (1 - range)), height),
                (0, 255, 0), 2)
        cv2.line(frame,
                (int(width * 0.5 * (1 + range)), 0),
                (int(width * 0.5 * (1 + range)), height),
                (0, 255, 0), 2)
        
        # 绘制每个目标
        for idx, obj in enumerate(objects):
            # 绘制边界框
            x1, y1, x2, y2 = obj['box']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            # 绘制类别和置信度
            label = f"{obj['class']} ({obj['conf']:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 绘制排序编号（从左到右）
            cv2.putText(frame, f"#{idx+1}", (int(x1)+5, int(y1)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 显示排序结果文本
        result_text = "排序: " + ", ".join([f"{i+1}.{obj['class']}" for i, obj in enumerate(objects)])
        cv2.putText(frame, result_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 保存图像到当前文件夹
        cv2.imwrite("sorted_objects_visualization.jpg", frame)
        
        # 显示图像并保持2秒后自动关闭
        cv2.imshow('Sorted Objects', frame)
        cv2.waitKey(2000)  # 等待2000毫秒（2秒）
        cv2.destroyWindow('Sorted Objects')
    
    def pred(self):
        """Run YOLO prediction on frame"""
        results = self.model(
            self.color_frame,
            imgsz=1280,
            conf=0.15,
            verbose=False,
        )
        self.color_frame = results[0].plot()
        model_names = results[0].names
        
        yoloresults = []
        for result in results[0]:
            box = result.boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = box.cls[0]
            if model_names[int(cls)] == "Water":
                center_y = y1/4 + y2/4*3
            else :
                center_y = (y1 + y2) / 2
            center_x = (x1 + x2) / 2
            yoloresult = YoloResult(model_names[int(cls)], [x1,y1,x2,y2], center_x, center_y, conf)
            yoloresults.append(yoloresult)
                
        return yoloresults
        
    def show(self, frame, width, range):
        """Display frame with detection results"""
        height = frame.shape[0]
        cv2.line(frame, 
                (int(width * 0.5 * (1 - range)), 0),
                (int(width * 0.5 * (1 - range)), height),
                (0, 255, 0), 2, 4)
        cv2.line(frame,
                (int(width * 0.5 * (1 + range)), 0),
                (int(width * 0.5 * (1 + range)), height),
                (0, 255, 0), 2, 4)
        cv2.imshow('yolo', frame)
        
    def judge(self, pattern, yolors, target=None, resolution=640, range=0.8):
        """Determine if detection meets criteria"""
        if pattern == "realtime":
            return cv2.waitKey(1) & 0xFF == ord('q')
        elif pattern == 'find':
            for yolor in yolors:
                if target == yolor.name:
                    return (self.judge_range(yolor.x, resolution, range) 
                            and (yolor.conf > 0.5))
        else:
            for yolor in yolors:
                if self.judge_range(yolor.x, resolution, range):
                    return True
        return False
        
    def judge_range(self, x, resolution, range):
        """Check if point is within target range"""
        left = resolution * 0.5 * (1 - range)
        right = resolution * 0.5 * (1 + range)
        return left <= x <= right
        
    def world(self, camera, depth_image, yolors):
        """Calculate 3D world coordinates"""
        K = camera.get_calibration()
        x = yolors.x
        y = yolors.y
        z = depth_image[int(y), int(x)] * 0.001
        point_image = np.array([x, y, 1])
        point = z * np.linalg.inv(K).dot(point_image)
        return point

if __name__ == "__main__":
    # 初始化相机和检测器
    camera = KinectCamera()
    camera.open_camera()
    detector = ItemsDetector()
    
    # 示例：检测"cup"目标，最大距离5米
    has_target, coords = detector.detect(camera, target='cup', max_distance=5.0)
    print(f"是否检测到目标: {has_target}")
    if has_target:
        print(f"三维坐标: x={coords[0]:.2f}m, y={coords[1]:.2f}m, z={coords[2]:.2f}m")
    
    # 获取按从左到右排序的物品种类列表，并显示可视化结果
    object_classes = detector.get_object_classes_sorted(camera, range=0.8, visualize=True)
    print("识别到的物品（从左到右）:", object_classes)
    print("物品数量为:", len(object_classes))  
    
    camera.release()
    cv2.destroyAllWindows()
