import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO
from abc import ABC, abstractmethod

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
                                  0.0, 0.0, 1.0]).reshape(3,3)
    
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

class PersonDetector:
    def __init__(self, model_path='./model/yolo11m.pt'):
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def get_target_distance(self, depth_image, x, y):
        if depth_image is None:
            return None
        try:
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                distance = depth_image[int(y), int(x)] * 0.001
                return distance if distance > 0 else None
            return None
        except:
            return None

    def detect_person(self, camera, max_distance):
        """
        检测指定距离内是否有人，并返回人的三维坐标
        参数:
            camera: 相机对象
            max_distance: 最大检测距离(米)
        返回:
            (has_person, 3d_coords)
            has_person: 布尔值，表示是否检测到指定距离内的人
            3d_coords: 三维坐标元组(x, y, z)，若未检测到则为(0, 0, 0)
        """
        start_time = time.time()
        timeout = 5  # 超时时间(秒)
        
        while time.time() - start_time < timeout:
            ret, color_frame = camera.get_frame()
            retd, depth_image = camera.get_depth()
            
            if not ret or not retd:
                continue

            results = self.model(color_frame)
            K = camera.get_calibration()
            height, width = color_frame.shape[:2]

            for result in results[0]:
                box = result.boxes
                cls = box.cls[0]
                if result.names[int(cls)] != 'person':
                    continue

                # 计算中心点
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                conf = box.conf[0]

                if conf < 0.5:
                    continue

                # 计算距离
                distance = self.get_target_distance(depth_image, center_x, center_y)
                if not distance or distance > max_distance:
                    continue

                # 计算三维坐标
                z = distance
                point_image = np.array([center_x, center_y, 1])
                point_3d = z * np.linalg.inv(K).dot(point_image)
                return (True, (point_3d[0], point_3d[1], point_3d[2]))

            if cv2.waitKey(10) == ord('q'):
                break

        return (False, (0.0, 0.0, 0.0))

if __name__ == "__main__":
    # 示例使用
    camera = KinectCamera()
    camera.open_camera()
    detector = PersonDetector()
    
    # 检测5米内是否有人
    has_person, coords = detector.detect_person(camera, max_distance=5.0)
    
    print(f"是否检测到人: {has_person}")
    if has_person:
        print(f"三维坐标: x={coords[0]:.2f}m, y={coords[1]:.2f}m, z={coords[2]:.2f}m")
    
    camera.release()
    cv2.destroyAllWindows()