import cv2
import os
from pyKinectAzure import pyKinectAzure, _k4a
from deepface import DeepFace
import numpy as np
import shutil
import time
import torch
# from ultralytics import YOLO
from datetime import datetime

"""
10月13日修改
detect_known_faces返回值变为人脸标号，0表示未检测到
main函数调用方法:
    photo_path = '/home/zq/catkin_ws/src/cmoon/src/jujia24/photo_test'  # 替换为你想要保存照片的路径
    camera = FaceDetector(photo_path)   # 创建FaceDetector类的实例
    camera.register_new_face()          # 注册人脸
    camera.detect_known_faces()         # 识别人脸
识别结果保存位置：
    camera.detect_result
"""

"""
2026.9.30修改
添加左侧、右侧照片采集，提高识别稳定性

注册前检查是否已经注册，避免同一个人产生多个 ID

添加语音提示回调，注册侧脸前播报“请向左转一点”“请向右转一点”

阈值0.65—>0.5，降低误识别率

重复注册同一个人脸时，覆盖原有照片

"""

class Detector:
    def __init__(self, photopath, device = 'k4a'):
        self.photopath = photopath  # 定义照片保存的路径
        # self.delete_all_faces()
        # 检查照片保存路径是否存在，如果不存在则创建
        if not os.path.exists(self.photopath):
            os.makedirs(self.photopath)
        
        self.known_faces = {}  # 存储每个人脸编号对应的特征向量
        self.face_id_counter = 0  # 人脸编号计数器，用于生成新的人脸编号
        self.face_folders = {}  # 存储每个人脸编号对应的文件夹路径
        self.detect_result = None
        self.depth = 0
        self.device = device
        self.k4a = pyKinectAzure()
        self.K_kinect = np.array([915.0828247070312, 0.0, 961.7936401367188, 
                                  0.0, 914.6190185546875, 555.453369140625, 
                                  0.0, 0.0, 1.0]).reshape(3,3)  # 1080P
        self.update_known_faces()
        self.is_open = 0
        # self.open_k4a()


        #加载模型
        # self.model = YOLO("/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/model/yolo11x-pose.pt")
        # self.yolodevice = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(self.yolodevice)
        # print(f"Using device: {self.yolodevice}")

   


    def get_person_3d_coords(self, depth_image, x, y):
        """计算人相对于相机的三维坐标（以鼻子为参考点）"""
        if x < 1 or y < 1:
            return (0.0, 0.0, 0.0)  # 无效坐标
        
        # 获取深度值（米）
        z = depth_image[int(y), int(x)] * 0.001
        self.depth = z
        print(f"z:{z}")
        # 获取相机内参
        K = self.K_kinect
        
        # 计算三维坐标
        point_image = np.array([x, y, 1])
        point_3d = z * np.linalg.inv(K).dot(point_image)
        
        return [point_3d[0], point_3d[1], point_3d[2]]  # (x, y, z)


#----------------------------------------------------------------------------------------------------------
# 人脸检测部分


    def detect_faces(self, img_path):
        try:
            """检测图片中的人脸，并返回最中心的人脸"""

            faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", align=True, enforce_detection=True)
            # face = DeepFace.extract_faces(img_path, detector_backend="opencv", align=True, enforce_detection=True)
            img = cv2.imread(img_path)

            center_face = None
            min_distance = float('inf')
            min_depth = float('inf')
            face_depth = 0
            if len(faces) > 0:
                img_height, img_width = img.shape[:2]  # 获取图像尺寸
                img_center_x, img_center_y = img_width / 2, img_height / 2  # 计算图像中心点

                for result in faces:
                    facial_area = result["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_center_x, face_center_y = x + w / 2, y + h / 2  # 计算人脸中心点
                    # 获取深度
                    depth_image = self.k4a.transform_depth_to_color(self.depth_image_handle, self.color_image_handle) 
                    face_depth = depth_image[int(y+h/2), int(x+w/2)] * 0.001
                    # print(f"距离:{face_depth} m")

                    # 计算人脸中心点到图像中心点的距离
                    # distance = ((img_center_x - face_center_x) ** 2 + (img_center_y - face_center_y) ** 2) ** 0.5

                    # 找到最中心的人脸
                    if face_depth < min_depth and face_depth != 0:
                        min_depth = face_depth
                        center_face = result

                if center_face is not None:
                    print("识别到人脸")
                    facial_area = center_face["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绘制矩形框



                    img_resized = self.resize_image(img, 0.5)
                    cv2.imshow("face_detect", img_resized)
                    cv2.waitKey(2000)
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return center_face
        except Exception as e:
            print("未识别到人脸")
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return None



    def detect_two_faces(self, img_path):
        try:
            """检测图片中的人脸，并返回最中心的两张人脸"""

            faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", align=True, enforce_detection=True)
            # face = DeepFace.extract_faces(img_path, detector_backend="opencv", align=True, enforce_detection=True)
            # print(faces)
            img = cv2.imread(img_path)

            center_face = []
            face_distances = []
            min_distance = float('inf')
            face_depth = 0
            if len(faces) > 0:
                img_height, img_width = img.shape[:2]  # 获取图像尺寸
                img_center_x, img_center_y = img_width / 2, img_height / 2  # 计算图像中心点

                for result in faces:
                    facial_area = result["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_center_x, face_center_y = x + w / 2, y + h / 2  # 计算人脸中心点

                    depth_image = self.k4a.transform_depth_to_color(self.depth_image_handle, self.color_image_handle) 
                    base_3d_point = self.get_person_3d_coords(depth_image, int(face_center_x), int(face_center_y))
                    face_depth = depth_image[int(y+h/2), int(x+w/2)] * 0.001
                    # print(f"距离:{face_depth} m")

                    # 计算人脸中心点到图像中心点的距离
                    distance = face_depth
                    if distance == 0:
                        distance = 99

                    # 将结果和距离存入列表
                    face_distances.append({
                        "face": result,
                        "distance": distance,
                        "base_point": base_3d_point
                    })

                # 升序排序
                face_distances.sort(key=lambda item: item['distance'])
                top_two_faces_data = face_distances[:2]
                final_center_data = [
                (item['face'], item['base_point']) 
                for item in top_two_faces_data
                ]

                if len(final_center_data) > 0:
                    print(f"识别到 {len(faces)} 张人脸, 选取最近的 {len(final_center_data)} 张")
                    # 画图
                    for face_result, base_point in final_center_data:
                        facial_area = face_result["facial_area"]
                        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                        # face_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绘制矩形框

                  

                    img_resized = self.resize_image(img, 0.5)
                    cv2.imshow("face_detect", img_resized)
                    cv2.waitKey(2000)
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return final_center_data
        except Exception as e:
            # print("未识别到人脸")
            print(e)
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return []

    def save_face_image(self, face_img, person_id):
        # 保存人脸图像到对应的文件夹
        # 检查是否存在编号对应的文件夹，如果不存在则创建文件夹
        folder_path = os.path.join(self.photopath, str(person_id))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 构建文件名，例如 person_1_face_1.jpg
        file_name = f"person_{person_id}_face_{len(os.listdir(folder_path)) + 1}.jpg"
        file_path = os.path.join(folder_path, file_name)

        # 保存人脸图像
        cv2.imwrite(file_path, face_img)
        # print("正在更新特征向量平均值...")
        # self.update_known_faces()
        # print(f"已保存图像到: {file_path}")

    def take_photo(self, device='camera'):
        """摄像头拍照保存"""
        if device == 'k4a' or device == 'kinect':
            self.open_k4a()

            # 定义保存照片的路径
            path = self.photopath + '/photo.jpg'
            # 循环捕获图像直到获取到图像
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                depth_image_handle = self.k4a.capture_get_depth_image()
                if color_image_handle:
                    self.color_image_handle = color_image_handle
                    self.depth_image_handle = depth_image_handle
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    # color_image = color_image[0:1080,800:1120]
                    cv2.imwrite(path, color_image)
                    break

        elif device == 'laptop':   # 调用笔记本摄像头
            # 使用OpenCV打开笔记本内置摄像头
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open video device.")
                return None
            path = self.photopath + '/photo.jpg'
            # 创建显示窗口
            # cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
 
            # while True:
                # print(111)
            ret, frame = cap.read()
            if ret:
                # print(222)
                cv2.imwrite(path, frame)
                # cv2.imshow('Camera Preview', frame)
            else:
                print("Error: Could not read frame from camera.")
           
            # time.sleep(1)
            cap.release()
            cv2.destroyAllWindows()
        else:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.open(0)
            flag, frame = cap.read()

            cv2.imwrite(path, frame)
            cap.release()
        # 返回保存图像的路径
        return path
    
    def resize_image(self, img, scale_factor):
        """缩放图像"""
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized
    
    # 计算给定文件夹中所有人脸的平均特征向量
    def calculate_average_embedding(self, folder_path, model_name='VGG-Face'):
        embeddings = []
        face_count = 0

        # 遍历文件夹中的所有图像文件
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                try:
                    # 提取单个人脸的特征向量
                    embeddings_result = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        enforce_detection=False,
                        detector_backend="retinaface",
                        align=True
                    )
                    embeddings.append(embeddings_result[0]["embedding"])
                    face_count += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

        if face_count == 0:
            return None

        # 计算平均特征向量
        average_embedding = np.mean(embeddings, axis=0)
        return average_embedding

    def update_known_faces(self):
        # 遍历已知的人脸文件夹，为每个人脸文件夹的特征向量计算平均值
        if self.face_folders == {}:
            print("程序中未先注册人脸，直接遍历文件夹来计算特征向量平均值")
            for folder_name in os.listdir(self.photopath):
                folder_path = os.path.join(self.photopath, folder_name)
                if os.path.isdir(folder_path) and folder_name.isdigit():  # 确保是数字编号的文件夹
                    person_id = int(folder_name)
                    self.face_folders[person_id] = folder_path

        for person_id, folder_path in self.face_folders.items():
            # 计算每个人脸文件夹中的特征向量平均值
            average_embedding = self.calculate_average_embedding(folder_path)
            if average_embedding is not None:
                self.known_faces[person_id] = average_embedding
            else:
                print(f"文件夹中未找到：{folder_path}")

    def compare_faces(self, new_image_path, model_name='VGG-Face'):
        # 提取新图像的特征向量，并与现有的比较
        # 未被调用
        new_image_embedding = DeepFace.represent(
            img_path=new_image_path,
            model_name=model_name,
            enforce_detection=False,
            detector_backend="retinaface",
            align=True
        )[0]["embedding"]
        
        # 初始化最佳匹配和最低距离
        best_match = None
        best_distance = float('inf')

        # 计算新图像与已知人脸的相似度
        for person_id, known_embedding in self.known_faces.items():
            distance = 1 - np.dot(known_embedding, new_image_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(new_image_embedding))
            if distance < best_distance:
                best_distance = distance
                best_match = person_id

        # 阈值判断
        threshold = 0.65
        if best_match and best_distance < threshold:
            print(f"检测到主人: {best_match}, 距离 {best_distance}")
        else:
            print("未找到主人")


    def _find_registered_face(self, face, threshold=0.5):
        """将检测到的人脸与已注册特征比较，返回已有 ID 和距离。"""
        if not self.known_faces:
            return None, float('inf')

        new_image_embedding = DeepFace.represent(
            img_path=face["face"],
            model_name='VGG-Face',
            enforce_detection=False,
            detector_backend="retinaface",
            align=True
        )[0]["embedding"]

        best_match = None
        best_distance = float('inf')
        for person_id, known_embedding in self.known_faces.items():
            distance = 1 - np.dot(
                known_embedding,
                new_image_embedding
            ) / (
                np.linalg.norm(known_embedding)
                * np.linalg.norm(new_image_embedding)
            )
            print(f"注册前人脸匹配差异: ID={person_id}, distance={distance}")
            if distance < best_distance:
                best_distance = distance
                best_match = person_id

        if best_match is not None and best_distance < threshold:
            return best_match, best_distance
        return None, best_distance

    # ===== 2026-10-01 修改：增加左转、右转语音提示回调 START =====
    def _capture_face_for_registration(self, person_id, prompt,
                                       prompt_callback=None,
                                       max_attempts=3):
        """按指定朝向采集一张人脸，避免未检测到时无限等待。"""
        for attempt in range(1, max_attempts + 1):
            print(f"{prompt}（第{attempt}/{max_attempts}次）")
            if prompt_callback is not None:
                # 回调由主流程提供，负责播报并等待播报结束。
                prompt_callback(prompt)
                # 播报完成后留出短暂时间，让主人稳定头部姿态。
                time.sleep(1)
            else:
                # 未提供语音回调时保持原有等待逻辑。
                time.sleep(3 if attempt == 1 else 2)

            img_path = self.take_photo(self.device)
            face = self.detect_faces(img_path)
            if face is None:
                print("未检测到人脸")
                continue

            img = cv2.imread(img_path)
            facial_area = face["facial_area"]
            x = facial_area["x"]
            y = facial_area["y"]
            w = facial_area["w"]
            h = facial_area["h"]
            face_img = img[y:y+h, x:x+w]
            self.save_face_image(face_img, person_id)
            return True

        print(f"{prompt}的人脸采集失败，跳过该角度")
        return False

    def _save_front_face_for_registration(self, person_id, img_path, face):
        """保存首次检测到的正面人脸照片。"""
        img = cv2.imread(img_path)
        facial_area = face["facial_area"]
        x = facial_area["x"]
        y = facial_area["y"]
        w = facial_area["w"]
        h = facial_area["h"]
        face_img = img[y:y+h, x:x+w]
        self.save_face_image(face_img, person_id)
        print(f"注册新人脸: {person_id}，正面照片采集成功")

    def _register_front_only(self, person_id, img_path, face):
        """方案一：只保存一张正面人脸照片。"""
        self._save_front_face_for_registration(person_id, img_path, face)
        return 1

    def _register_front_and_sides(self, person_id, img_path, face,
                                  prompt_callback=None):
        """方案二：保存正面、左侧和右侧人脸照片。"""
        self._save_front_face_for_registration(person_id, img_path, face)
        saved_face_count = 1

        if self._capture_face_for_registration(
            person_id,
            "请向左转一点",
            prompt_callback=prompt_callback
        ):
            saved_face_count += 1
        if self._capture_face_for_registration(
            person_id,
            "请向右转一点",
            prompt_callback=prompt_callback
        ):
            saved_face_count += 1

        return saved_face_count

    def register_new_face(self, img_path=None, prompt_callback=None):
        """注册新人脸；通过注释调用行选择单张或三张采集方案。"""
        print("开始注册人脸")
        img_path = self.take_photo(self.device)
        face = self.detect_faces(img_path)
        if face is None:
            return None

        # 创建新 ID 前先检查是否已经注册。
        # 已注册的人脸复用原 ID，并用本次采集结果覆盖原照片。
        existing_person_id, best_distance = self._find_registered_face(face)
        if existing_person_id is not None:
            print(
                f"该人脸已经注册，已有ID: {existing_person_id}，"
                f"距离: {best_distance}"
            )
            new_person_id = existing_person_id
            new_folder_path = os.path.join(
                self.photopath,
                str(new_person_id)
            )
            if os.path.exists(new_folder_path):
                shutil.rmtree(new_folder_path)
            os.makedirs(new_folder_path)
            self.known_faces.pop(new_person_id, None)
            print(f"正在覆盖人脸ID {new_person_id} 的原有照片")
        else:
            max_id = 0
            for folder_name in os.listdir(self.photopath):
                if folder_name.isdigit() and int(folder_name) > max_id:
                    max_id = int(folder_name)
            self.face_id_counter = max_id + 1
            new_person_id = self.face_id_counter
            new_folder_path = os.path.join(
                self.photopath,
                str(new_person_id)
            )
            os.makedirs(new_folder_path, exist_ok=True)

        self.face_folders[new_person_id] = new_folder_path

        # --------------------------------------------------------------
        # 人脸注册采集方案：两种方案只能启用一种。
        # --------------------------------------------------------------

        # 方案一：只拍一张正脸。
        # saved_face_count = self._register_front_only(
        #     new_person_id,
        #     img_path,
        #     face
        # )

        # 方案二：拍正面、左侧和右侧三张照片。
        saved_face_count = self._register_front_and_sides(
            new_person_id,
            img_path,
            face,
            prompt_callback=prompt_callback
        )
        # ===== 2026-10-01 修改：增加左转、右转语音提示回调 END =====

        self.detect_result = new_person_id
        print(
            f"人脸ID {new_person_id} 共保存 "
            f"{saved_face_count} 张注册照片"
        )
        print("正在更新特征向量平均值...")
        self.update_known_faces()
        return new_person_id





    def detect_known_faces(self, img_path = None):
        # 检测已知人脸，返回识别的人脸编号，未找到则是0
        print("正在检测已知人脸")
        if img_path == None:
            img_path = self.take_photo(self.device)

        face_data  = self.detect_two_faces(img_path)

        self.detect_result = None
        result = [[0,0,0],[0,0,0]]  # 第一个主人坐标，第二个客人坐标
        result_coords = [0, 0, 0]
        # self.close_k4a()

        if len(face_data) == 0:
            print("未识别到人脸, 跳过该进程")
            return result  # 返回空结果列表
        
        unregistered_face_coords = None

        # 提取新图像的特征向量
        for items in face_data:
            new_image_embedding = DeepFace.represent(
                img_path=items[0]["face"],
                model_name='VGG-Face',
                enforce_detection=False,
                detector_backend="retinaface",
                align=True
            )[0]["embedding"]
            
            
            best_match = None
            best_distance = float('inf')
            # self.update_known_faces()

            # 尝试匹配已知人脸
            for person_id, known_embedding in self.known_faces.items():
                distance = 1 - np.dot(known_embedding, new_image_embedding) / (np.linalg.norm(known_embedding) * np.linalg.norm(new_image_embedding))
                print(f"人脸匹配差异:{distance}")
                if distance < best_distance:
                    best_distance = distance
                    best_match = person_id
            

            # 检查匹配结果
            if best_match is not None and best_distance < 0.5:
                print(f"检测到主人：{best_match}，距离{best_distance}")
                self.detect_result = best_match     # 将检测结果保存在 detect_result 里
                result[0] = items[1]
            else:
                result[1] = items[1]
                print(f"best_distance:{best_distance}， 最短距离大于0.5，不是已知人脸")
                # break
                # 保存已识别人脸的图片
                # img = cv2.imread(img_path)

                # facial_area = face["facial_area"]
                # x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                # face_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
                # self.save_face_image(face_img, best_match)     

        return result  # 返回识别的人脸编号，未找到则是0
    
    def delete_all_faces(self):
        # 删除整个文件夹及其内容
        if os.path.exists(self.photopath):
            shutil.rmtree(self.photopath)
            print(f"已删除文件夹及其内容: {self.photopath}")
        else:
            print("文件夹不存在，无法删除。")


    def open_k4a(self):
        if not self.is_open:
            device_opened = False
            try:
                # self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'
                self.k4a.device_open()
                device_opened = True
                device_config = self.k4a.config
                device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
                # device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
                self.k4a.device_start_cameras(device_config)
                self.is_open = 1
                time.sleep(1)
                print("开k4a")
            except (Exception, SystemExit) as error:
                # _k4a.VERIFY() 失败时会抛出 SystemExit。
                # 若设备已经打开但相机启动失败，必须关闭句柄，
                # 否则下一次打开会出现 LIBUSB_ERROR_BUSY。
                if device_opened:
                    try:
                        self.k4a.device_close()
                    except Exception as close_error:
                        print(f"清理K4A设备句柄失败: {close_error}")
                self.is_open = 0
                raise RuntimeError("打开K4A相机失败") from error
        else:
            print("k4a已经开启")

    def close_k4a(self):
        if self.is_open:
            self.k4a.device_stop_cameras()
            self.k4a.device_close()
            self.is_open = 0
            print("关k4a")
        else:
            print("k4a已经关闭")




# 使用Kinect摄像头拍照并识别人脸
if __name__ == "__main__":
    photo_path = '/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/face'  # 替换为你想要保存照片的路径
    camera = Detector(photo_path)   # 创建FaceDetector类的实例
    camera.register_new_face()          # 注册人脸
    # time.sleep(1)
    # camera.register_new_face()          # 注册人脸
    # time.sleep(1)
    # camera.register_new_face()          # 注册人脸
    # # time.sleep(5)
    # start_time = time.time()
    # face_num = 1        
    # camera.speak("开始注册人脸")
    # while(face_num <= 3):
    #     time.sleep(2)
    #     if camera.register_new_face():
    #         face_num += 1
    #         if face_num <= 3:
    #             camera.speak("开始注册下一位")
    #         time.sleep(2)
    #     else:
    #         camera.speak("未找到人脸，两秒后将再试一次")
    #         time.sleep(2)
    # camera.speak("注册完毕")
    # print(f"time:{time.time()-start_time}")
    # print(camera.detect_known_faces())
    # print(camera.detect_known_faces())
    # print(camera.detect_known_faces())
    camera.close_k4a()
    # coordinates,depth = camera.person_detect()
    # if coordinates is not None:
    #     camera.register_new_face()
    #     print(f"距离为{depth}")

    # result, color_frame = camera.pose_detect()
    # cv2.imshow("Nose depth (press Q to continue)", color_frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

    # action_type = camera.wave()
    # print(action_type)

    
