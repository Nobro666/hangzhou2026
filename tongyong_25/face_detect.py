import cv2
import os
from pyKinectAzure import pyKinectAzure, _k4a
from deepface import DeepFace
import numpy as np
import shutil
import time
import torch
from ultralytics import YOLO
from datetime import datetime
import sys
sys.path.append(r"/home/zq/catkin_ws/src/cmoon/src")

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


        # 加载模型
        self.model = YOLO("/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/model/yolo11x-pose.pt")
        self.yolodevice = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.yolodevice)
        print(f"Using device: {self.yolodevice}")


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

    # 检测人    
    def person_detect(self):
        if self.device == 'k4a' or self.device == 'kinect':
            self.open_k4a()

            # 捕获图像
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                depth_image_handle = self.k4a.capture_get_depth_image()
                if color_image_handle:
                    self.color_image_handle = color_image_handle
                    self.depth_image_handle = depth_image_handle
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    break
            results = []
            # 检测
            
            results = self.model(color_image)  
            try:
                keypoints = results[0].keypoints[0][0]
                if keypoints.shape[0] > 0:                 
                    # nose = keypoints[0][0]
                    # print(nose)
                    # 定义保存照片的路径
                    path = self.photopath + '/photo.jpg'
                    x,y = keypoints.xy[0][0]
                    # 有鼻子就拿深度
                    depth_image = self.k4a.transform_depth_to_color(self.depth_image_handle, self.color_image_handle) 

                    base_3d_point = self.get_person_3d_coords(depth_image, int(x), int(y))
                    # 保存图片
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    # color_image = color_image[0:1080,800:1120]
                    cv2.imwrite(path, color_image)  
                    # ------ 新增：显示并标注 ------
                    # cv2.circle(color_image, (int(x), int(y)), 8, (0, 0, 255), -1)
                    # # cv2.putText(color_image, f"{person_depth} mm", (int(x) + 10, int(y) - 10),
                    # #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # cv2.imshow("Nose depth (press Q to continue)", color_image)
                    # if cv2.waitKey(0) & 0xFF == ord('q'):
                    #     cv2.destroyAllWindows()
                    # --------------------------------
                    # camera.k4a.device_stop_cameras()
                    # camera.k4a.device_close()
                    # self.close_k4a()
                    return base_3d_point, path
                else:
                    print('未检测到鼻子')
                    # self.close_k4a()
                    return None, 0
            except:
                print('未检测到人')
                return None, 0

#----------------------------------------------------------------------------------------------------------
# 人脸检测部分

    def detect_faces_2(self, img_path):
        try:
            """检测图片中的人脸，并返回最中心的人脸"""

            faces = DeepFace.extract_faces(img_path, detector_backend="retinaface", align=True, enforce_detection=True)
            # face = DeepFace.extract_faces(img_path, detector_backend="opencv", align=True, enforce_detection=True)
            img = cv2.imread(img_path)

            center_face = None
            min_distance = float('inf')
            face_depth = 0
            if len(faces) > 0:
                img_height, img_width = img.shape[:2]  # 获取图像尺寸
                img_center_x, img_center_y = img_width / 2, img_height / 2  # 计算图像中心点

                for result in faces:
                    facial_area = result["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_center_x, face_center_y = x + w / 2, y + h / 2  # 计算人脸中心点

                    # 计算人脸中心点到图像中心点的距离
                    distance = ((img_center_x - face_center_x) ** 2 + (img_center_y - face_center_y) ** 2) ** 0.5

                    # 找到最中心的人脸
                    if distance < min_distance:
                        min_distance = distance
                        center_face = result

                if center_face is not None:
                    print("识别到人脸")
                    facial_area = center_face["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绘制矩形框

                    # 获取深度
                    depth_image = self.k4a.transform_depth_to_color(self.depth_image_handle, self.color_image_handle) 
                    face_depth = depth_image[int(y+h/2), int(x+w/2)] * 0.001
                    print(f"距离:{face_depth} m")

                    img_resized = self.resize_image(img, 0.5)
                    cv2.imshow("face_detect", img_resized)
                    cv2.waitKey(2000)
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return center_face, face_depth
        except Exception as e:
            print("未识别到人脸")
            # self.k4a.device_stop_cameras()
            # self.k4a.device_close()
            return None, 0

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
        print("正在更新特征向量平均值...")
        self.update_known_faces()
        print(f"已保存图像到: {file_path}")

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
        threshold = 0.5
        if best_match and best_distance < threshold:
            print(f"检测到主人: {best_match}, 距离 {best_distance}")
        else:
            print("未找到主人")

    def register_new_face(self, img_path = None):
        # 注册新人脸
        print("正在注册人脸")
        if img_path == None:
            img_path = self.photopath
        img_path = self.take_photo(self.device)
        face = self.detect_faces(img_path)
        reg_num = 0
        if face:
            if self.detect_known_faces(img_path, face) == 0: # 判断是否注册过
                
                for folder_name in os.listdir(self.photopath):  # 在注册过的人脸文件夹数字之后再建立新文件夹
                    if folder_name.isdigit():  # 确保是数字编号的文件夹
                        self.face_id_counter = int(folder_name)

                self.face_id_counter += 1
                new_person_id = self.face_id_counter

                new_folder_path = os.path.join(self.photopath, str(new_person_id))
                self.face_folders[new_person_id] = new_folder_path
                os.makedirs(new_folder_path, exist_ok=True)
                img = cv2.imread(img_path)

                facial_area = face["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                face_img = img[y:y+h, x:x+w]    # 裁剪人脸区域

                self.save_face_image(face_img, new_person_id)
                print(f"注册新人脸: {new_person_id}")
                self.detect_result = new_person_id

                while(reg_num < 2):
                    if reg_num == 0:
                        print("请向左转一点")
                    else:
                        print("请向右转一点")
                    time.sleep(3)
                    img_path = self.take_photo(self.device)
                    face = self.detect_faces(img_path)
                    if face:
                        facial_area = face["facial_area"]
                        x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                        img = cv2.imread(img_path)
                        face_img = img[y:y+h, x:x+w]    # 裁剪人脸区域
                        self.save_face_image(face_img, new_person_id)
                        reg_num += 1
                    else:
                        print("未检测到人脸，两秒后将再试一次")
                        time.sleep(2)
                # self.close_k4a()

                return 1
            else:
                print("已经注册过该人脸")
                # self.close_k4a()
                return 0
        else:
            return 0

    def detect_known_faces(self, img_path = None, face = None):
        # 检测已知人脸，返回识别的人脸编号，未找到则是0
        print("正在检测已知人脸")
        if img_path == None:
            img_path = self.take_photo(self.device)
        if face == None:
            start_time = time.time()
            face = self.detect_faces(img_path)
            print(f"time : {time.time()-start_time}")
        self.detect_result = None
        result = 0
        # self.close_k4a()

        if face is None:
            print("未识别到人脸, 跳过该进程")
            return result  # 返回空结果列表
        
        # 提取新图像的特征向量
        
        new_image_embedding = DeepFace.represent(
            img_path=face["face"],
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
            if distance < best_distance:
                best_distance = distance
                best_match = person_id
        

        # 检查匹配结果
        if best_match is not None and best_distance < 0.5:
            print(f"检测到主人：{best_match}，距离{best_distance}")
            self.detect_result = best_match     # 将检测结果保存在 detect_result 里
            result = best_match

            # 保存已识别人脸的图片
            img = cv2.imread(img_path)

            facial_area = face["facial_area"]
            x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
            # face_img = img[y:y+h, x:x+w]  # 裁剪人脸区域
            # self.save_face_image(face_img, best_match)
            
        else:
            print(f"best_distance:{best_distance}， 最短距离大于0.6，不是已知人脸")

        return result  # 返回识别的人脸编号，未找到则是0
    
    def delete_all_faces(self):
        # 删除整个文件夹及其内容
        if os.path.exists(self.photopath):
            shutil.rmtree(self.photopath)
            print(f"已删除文件夹及其内容: {self.photopath}")
        else:
            print("文件夹不存在，无法删除。")

#----------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------
# 姿态检测部分
    def pose_detect(self, key="feet"):
        """
            检测人体关键点，并根据传入参数返回对应的关键点信息
        """
        try:
            if self.device == 'k4a' or self.device == 'kinect':
                self.open_k4a()
                # 捕获图像
                while True:
                    self.k4a.device_get_capture()
                    color_image_handle = self.k4a.capture_get_color_image()
                    depth_image_handle = self.k4a.capture_get_depth_image()
                    if color_image_handle:
                        self.color_image_handle = color_image_handle
                        self.depth_image_handle = depth_image_handle
                        color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                        break

            results = self.model(color_image)
            # color_frame = results[0].plot()
            color_frame = color_image.copy()
            keypoint_rs = results[0].keypoints
            keypoint = keypoint_rs.xy
            results = []
            if keypoint_rs.shape[1] > 0:
                color_frame = self.draw_skeleton(color_frame, keypoint[0])
                if key == "hand":
                    left_wrist = keypoint[0][9] # 索引 9 是左腕
                    right_wrist = keypoint[0][10]
                    print(f"left_wrist: {left_wrist}")
                    print(f"right_wrist: {right_wrist}")
                    results = [left_wrist,right_wrist]
                elif key == "foot":
                    left_foot = keypoint[0][15]
                    right_foot = keypoint[0][16]
                    print(f"left_foot: {left_foot}")
                    print(f"right_foot: {right_foot}")
                    results = [left_foot,right_foot]
                #     xyn = result.keypoints.xyn  # normalized
                #     kpts = result.keypoints.data  # x, y, visibility (if available)
                #     print(f"xy: {xy}")
                #     print(f"xyn: {xyn}")
                #     print(f"kpts: {kpts}")
            # self.close_k4a()
            return results, color_frame
        except:
            # self.close_k4a()
            return None, None
    
    def draw_skeleton(self, image, keypoints):
        """
        在图像上绘制人体骨架
        image: 原始图像
        keypoints: 人体关键点列表（shape: [17, 2]，每个元素为(x, y)坐标）
        """


        # 定义骨架连接（每个元组代表两个关键点的索引，构成一条骨骼线）
        self.skeleton = [
            (0, 1), (0, 2),  # 鼻子-左眼，鼻子-右眼
            (1, 3), (2, 4),  # 左眼-左耳，右眼-右耳
            (5, 6),  # 左肩-右肩
            (5, 7), (7, 9),  # 左肩-左肘-左腕
            (6, 8), (8, 10), # 右肩-右肘-右腕
            (11, 12), # 左髋-右髋
            (5, 11), (6, 12), # 左肩-左髋，右肩-右髋
            (11, 13), (13, 15), # 左髋-左膝-左脚踝
            (12, 14), (14, 16)  # 右髋-右膝-右脚踝
        ]
        # 定义骨架线条颜色（BGR格式）
        self.skeleton_color = (0, 255, 255)  # 黄色
        # 定义关键点颜色和大小
        self.keypoint_color = (0, 0, 255)    # 红色
        self.keypoint_radius = 3

        # 绘制关键点
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            if x > 0 and y > 0:  # 过滤无效关键点
                cv2.circle(image, (x, y), self.keypoint_radius, self.keypoint_color, -1)
        
        # 绘制骨骼连接
        for (i, j) in self.skeleton:
            kp1 = keypoints[i]
            kp2 = keypoints[j]
            x1, y1 = int(kp1[0]), int(kp1[1])
            x2, y2 = int(kp2[0]), int(kp2[1])
            # 只绘制有效关键点之间的连接
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(image, (x1, y1), (x2, y2), self.skeleton_color, 2)
        
        return image
    
    def detect_pose_type(self, person_kps, width, x1, x2, y1 ,y2):
        """
        检测人体姿势类型
        返回: (pose_type, distance)
        pose_type: 0-站立不挥手, 1-挥手, 2-躺下
        """
        # 获取关键节点
        left_wrist = person_kps[9]
        right_wrist = person_kps[10]
        nose = person_kps[0]
        left_hip = person_kps[11]
        right_hip = person_kps[12]
        left_ankle = person_kps[15]
        right_ankle = person_kps[16]


        # 检测是否躺下
        people_height = abs(y1 - y2)
        people_width = abs(x1 - x2)

        # 计算肩宽
        # shoulder_width = math.sqrt((abs(person_kps[6].tolist()[0] - person_kps[5].tolist()[0])) ** 2 + (abs(person_kps[6].tolist()[1] - person_kps[5].tolist()[1])) ** 2)

        # print(f"height:{height},shoulder_width:{shoulder_width}")
        print(f"people_height:{people_height},people_width:{people_width}")


        # 如果宽高比大于0.8，判断为躺下
        # if (shoulder_width / height) > 0.3:
        if people_width > people_height * 1.2:
            return 2  # 躺下
        
        # 先检测是否挥手（保持原有逻辑）
        x, y = nose.tolist()
        range_val = 0.5
        if (left_wrist.tolist()[1] > 1 and left_wrist.tolist()[1] < y and (width * 0.5 * (1 - range_val) < x < width * 0.5 * (1 + range_val))):
            return 1  # 挥手
        
        if (right_wrist.tolist()[1] > 1 and right_wrist.tolist()[1] < y and (width * 0.5 * (1 - range_val) < x < width * 0.5 * (1 + range_val))):
            return 1  # 挥手
        # 否则判断为站立不挥手
        return 0

    def wave(self):
        """
        扩展原有挥手检测功能，同时检测站立不挥手和躺下动作
        返回: (action_flag, action_type, distance, pose_image, color_image)
        action_type: 0-站立不挥手, 1-挥手, 2-躺下
        """
        try:
            if self.device == 'k4a' or self.device == 'kinect':
                    self.open_k4a()
            # 捕获图像
            while True:
                self.k4a.device_get_capture()
                color_image_handle = self.k4a.capture_get_color_image()
                depth_image_handle = self.k4a.capture_get_depth_image()
                if color_image_handle:
                    self.color_image_handle = color_image_handle
                    self.depth_image_handle = depth_image_handle
                    color_image = self.k4a.image_convert_to_numpy(color_image_handle)
                    break

            cv2.namedWindow('find_wave', cv2.WINDOW_NORMAL)
            action_flag = False
            action_type = -1  # 默认未检测到有效动作
            distance = 0.0
            results = self.model(color_image)
            pose_image = results[0].plot()
            height, width = color_image.shape[:2]

            keypoint_rs = results[0].keypoints
            
            if keypoint_rs.shape[1] > 0:
                for person_idx, person_kps in enumerate(keypoint_rs.xy):
                    # 检测姿势类型和距离

                    x1, y1, x2, y2 = map(int, results[0].boxes[person_idx].xyxy[0].tolist())
                    print(f"x1:{x1},y1:{y1},x2:{x2},y2:{y2}")
                    pose_type = self.detect_pose_type(person_kps, width, x1, x2, y1, y2)
                    
                    if pose_type != -1:
                        action_flag = True
                        action_type = pose_type
                        
                        # 绘制对应标签
                        color_image = self.draw_skeleton(color_image, person_kps)
                        x1, y1, x2, y2 = map(int, results[0].boxes[person_idx].xyxy[0].tolist())
                        labels = ["Standing", "Waving", "Lying"]
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(color_image, 
                                    f'{labels[pose_type]}', 
                                    (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 1)
                        
                        # 保存图像
                        now = datetime.now().strftime('%H:%M:%S')
                        # cv2.imwrite(f"{labels[pose_type]}{now}.jpg", color_image)
                        cv2.imshow('find_wave', color_image)
                        # if cv2.waitKey(0) & 0xFF == ord('q'):
                        #     cv2.destroyAllWindows()
                        break
            # self.close_k4a()        
            return action_type 
        except:
            # self.close_k4a()
            return None

#----------------------------------------------------------------------------------------------------------
    def open_k4a(self):
        if not self.is_open:
            # self.modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so'         
            self.k4a.device_open()
            device_config = self.k4a.config
            device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
            # device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
            self.k4a.device_start_cameras(device_config)
            self.is_open = 1
            time.sleep(1)
            print("开k4a")
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
    # camera.register_new_face()          # 注册人脸
    # time.sleep(1)
    camera.detect_known_faces()          # 注册人脸
    # time.sleep(1)
    # # camera.register_new_face()          # 注册人脸
    # # time.sleep(5)
    # start_time = time.time()
    # print(111)
    # print(camera.detect_known_faces())         # 识别人脸
    # print(camera.person_detect())   # 找人
    # camera.open_k4a()
    # time.sleep(1)
    camera.close_k4a()
    # time.sleep(1)
    # camera.open_k4a()
    # time.sleep(1)
    # print(f"time:{time.time()-start_time}")
    # camera.close_k4a()
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

    

