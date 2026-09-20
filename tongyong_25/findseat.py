"""
Data: 2026/4/3
Author: Wang Zhe
描述：发现座位的占用状态和人员信息
"""
import numpy as np
import cv2
import torch
import time
import sys
import rospy
from ultralytics import YOLO
from abc import ABC, abstractmethod

from camera_to_map import CoordinateConverter
from face_detect import Detector as FaceDetector


# 五个座位的地图坐标范围: (x_min, x_max, y_min, y_max)
SEAT_MAP_COORDS = {
    1: (0.05,0.8,1.0,1.7),
    2: (0.85,1.5,0.8,1.4),
    3: (1.59,2.2,1.0,1.8),
    4: (1.35,1.94,1.70,2.36),
    5: (0.44,1.2,1.70,2.31),
}

POSE_MODEL_PATH = "/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/model/yolo11x-pose.pt"
FACE_PHOTO_PATH = '/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/face' # 人脸照片保存路径
MAX_DETECT_DISTANCE = 5.0  # 最大检测距离（米）

class YoloResult:
    def __init__(self, name, x, y, conf) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.conf = conf
        self.distance = None

class Camera(ABC):
    @abstractmethod
    def get_frame(self):
        pass
        
    @abstractmethod
    def get_depth(self):
        pass
        
    @abstractmethod
    def get_calibration(self):
        pass
        
    @abstractmethod
    def release(self):
        pass


class KinectCamera(Camera):
    def __init__(self):
        self.K_kinect = np.array([915.0828247070312, 0.0, 961.7936401367188,
                                  0.0, 914.6190185546875, 555.453369140625,
                                  0.0, 0.0, 1.0]).reshape(3, 3)

    def open_camera(self):
        import pykinect_azure as pykinect
        from pykinect_azure import (
            K4A_FRAMES_PER_SECOND_30,
            K4A_WIRED_SYNC_MODE_STANDALONE
        )
        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device_config.camera_fps = K4A_FRAMES_PER_SECOND_30
        device_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE
        device_config.synchronized_images_only = True
        self.device = pykinect.start_device(config=device_config)

    def get_frame(self):
        capture = self.device.update()
        return capture.get_color_image()

    def get_depth(self):
        capture = self.device.update()
        return capture.get_transformed_depth_image()

    def get_calibration(self):
        return self.K_kinect

    def release(self):
        self.device.stop_cameras()
        self.device.close()

class SeatDetector:
    def __init__(self):
        rospy.init_node("seat_detection_node", anonymous=True) #单独运行该文件时放开注释

        # 初始化相机
        self.camera = KinectCamera()
        # 初始化YOLO Pose模型
        self.pose_model = YOLO(POSE_MODEL_PATH)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pose_model.to(self.device)
        # 初始化坐标转换
        self.coord_converter = CoordinateConverter()
        # 初始化人脸检测
        self.face_detector = FaceDetector(FACE_PHOTO_PATH, device='k4a')
        # 相机内参
        self.K = self.camera.get_calibration()

    def get_3d_coords(self, depth_image, x, y):
        if depth_image is None or x < 0 or y < 0:
            return None
        try:
            z = depth_image[int(y), int(x)] * 0.001
            if z <= 0 or z > MAX_DETECT_DISTANCE:
                return None
            point_image = np.array([x, y, 1])
            point_3d = z * np.linalg.inv(self.K).dot(point_image)
            return (point_3d[0], point_3d[1], point_3d[2])
        except Exception as e:
            rospy.logwarn(f"计算3D坐标失败: {e}")
            return None

    def detect_nose_keypoints(self, color_frame):
        nose_points = []
        results = self.pose_model(color_frame)
        
        for result in results:
            keypoints = result.keypoints
            if keypoints is None or len(keypoints) == 0:
                continue
            for person_kps in keypoints.xy:
                nose_x, nose_y = person_kps[0]  # 鼻子像素坐标
                if nose_x > 0 and nose_y > 0:  # 有效坐标
                    nose_points.append((float(nose_x), float(nose_y)))
        return nose_points

    def judge_seat_occupation(self, map_coords):
        """
        判断地图坐标属于哪个座位
        返回：座位编号 / None
        """
        if not map_coords:
            return None
        x, y, _ = map_coords 
        for seat_id, (x_min, x_max, y_min, y_max) in SEAT_MAP_COORDS.items():
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return seat_id
        return None

    def recognize_person(self, color_frame):
        try:
            # 保存临时照片
            temp_photo_path = f"{FACE_PHOTO_PATH}/temp_seat_face.jpg"
            cv2.imwrite(temp_photo_path, color_frame)
            
            face_id = self.face_detector.detect_known_faces(temp_photo_path)
            return face_id if face_id != 0 else "未知人员"
        except Exception as e:
            rospy.logerr(f"人员识别失败: {e}")
            return "识别失败"

    def detect_seats(self, timeout=10):
        """
        返回五个座位的占用状态和人员信息
        返回格式：{座位编号: {"occupied": 布尔值, "person": 人员ID/未知/None}}
        """
        start_time = time.time()
        # 初始化座位状态
        seat_status = {
            1: {"occupied": False, "person": None},
            2: {"occupied": False, "person": None},
            3: {"occupied": False, "person": None},
            4: {"occupied": False, "person": None},
            5: {"occupied": False, "person": None},
        }

        while time.time() - start_time < timeout:
            ret, color_frame = self.camera.get_frame()
            retd, depth_image = self.camera.get_depth()
            if not ret or not retd or color_frame is None or depth_image is None:
                time.sleep(0.1)
                continue

            nose_points = self.detect_nose_keypoints(color_frame)
            if not nose_points:
                continue

            occupied_seats = set() 
            for (nose_x, nose_y) in nose_points:
                camera_3d = self.get_3d_coords(depth_image, nose_x, nose_y)
                if not camera_3d:
                    continue

                map_3d = self.coord_converter.get_map_coords(camera_3d)
                if not map_3d:
                    continue

                seat_id = self.judge_seat_occupation(map_3d)
                if not seat_id or seat_id in occupied_seats:
                    continue

                seat_status[seat_id]["occupied"] = True
                seat_status[seat_id]["person"] = self.recognize_person(color_frame)
                occupied_seats.add(seat_id) 

            break

        return seat_status

    def release(self):
        self.camera.release()
        self.face_detector.close_k4a()
        cv2.destroyAllWindows()
        if rospy.core.is_initialized():
            rospy.signal_shutdown("座位检测完成，关闭ROS节点")


if __name__ == "__main__":
    seat_detector = SeatDetector()
    try:
        seat_detector.camera.open_camera()
        print("开始检测座位状态...")
        seat_status = seat_detector.detect_seats(timeout=10)
        
        print("\n===== 座位检测结果 =====")
        for seat_id in sorted(seat_status.keys()):
            status = seat_status[seat_id]
            if status["occupied"]:
                print(f"座位{seat_id}: 有人 | 人员身份: {status['person']}")
            else:
                print(f"座位{seat_id}: 空座位")
    except Exception as e:
        print(f"检测过程出错: {e}")
    finally:
        # 释放资源
        seat_detector.release()
        print("\n资源已释放，检测结束")