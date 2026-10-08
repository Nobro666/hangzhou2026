import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO
from abc import ABC, abstractmethod
from pathlib import Path

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
        self.device = None
    
    def open_camera(self):
        if self.device is not None:
            return

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

    def get_rgbd(self):
        """从同一个 Capture 获取彩色图和对齐后的深度图。"""
        capture = self.device.update()
        ret, color_image = capture.get_color_image()
        retd, depth_image = capture.get_transformed_depth_image()
        return ret, color_image, retd, depth_image
    
    def get_calibration(self):
        return self.K_kinect
        
    def release(self):
        if self.device is None:
            return

        device = self.device
        self.device = None
        try:
            device.stop_cameras()
        finally:
            device.close()

class PersonDetector:
    def __init__(self, model_path=None):
        project_dir = Path(__file__).resolve().parent
        if model_path is None:
            candidates = [
                project_dir / 'model' / 'yolo11m.pt',
                project_dir / 'catch_ground' / 'src' / 'model' / 'yolo11m.pt',
            ]
            model_file = next(
                (path for path in candidates if path.is_file()),
                candidates[0],
            )
        else:
            model_file = Path(model_path).expanduser()
            if not model_file.is_absolute():
                model_file = project_dir / model_file

        if not model_file.is_file():
            raise FileNotFoundError(
                f'未找到人物检测模型：{model_file}。'
                '请将 yolo11m.pt 放入项目 model 目录。'
            )

        self.model_path = str(model_file)
        self.model = YOLO(self.model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f'人物检测模型：{self.model_path}')

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

    def detect_person(self, camera, max_distance, timeout=5.0,
                      candidate_filter=None, return_confidence=False):
        """
        检测指定距离内是否有人，并返回人的三维坐标
        参数:
            camera: 相机对象
            max_distance: 最大检测距离(米)
            timeout: 单个方向的最长检测时间(秒)
            candidate_filter: 可选候选过滤函数，接收相机三维坐标并返回
                True/False。返回False时继续检查同一帧其他人物和后续帧。
            return_confidence: 为True时额外返回检测置信度。
        返回:
            (has_person, 3d_coords)
            has_person: 布尔值，表示是否检测到指定距离内的人
            3d_coords: 三维坐标元组(x, y, z)，若未检测到则为(0, 0, 0)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            ret, color_frame, retd, depth_image = camera.get_rgbd()
            
            if not ret or not retd:
                continue

            results = self.model(color_frame, verbose=False)
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
                conf = float(box.conf[0])

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
                person_coords = (
                    point_3d[0],
                    point_3d[1],
                    point_3d[2],
                )

                if candidate_filter is not None:
                    try:
                        if not candidate_filter(person_coords):
                            # 当前候选不在允许区域，继续检查本帧中的
                            # 其他人物；本帧都无效时继续读取后续帧。
                            continue
                    except Exception as error:
                        print(f"人物候选位置过滤发生异常：{error}")
                        continue

                if return_confidence:
                    return (True, person_coords, conf)
                return (True, person_coords)

            if cv2.waitKey(10) == ord('q'):
                break

        if return_confidence:
            return (False, (0.0, 0.0, 0.0), 0.0)
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
