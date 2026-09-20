#!/usr/bin/env python3
# coding: UTF-8

# from cmoon.src.pykinect_azure.k4a import device
import rospy
import cv2
from aip import AipBodyAnalysis, AipFace
import base64
import os
import numpy as np
import torch
from typing import *
from pathlib import Path
from numpy import dtype, random
import argparse

from PIL import Image
import imutils
from base_controller import Base

# import sys
# sys.path.insert(1, '../')
import pykinect_azure as pykinect

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

# import pykinect_azure as pykinect
from geometry_msgs.msg import Point

cv2_image = TypeVar("image opened by cv2")


class Detector(object):
    def __init__(self):

        # self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'
        self.photopath = r'/home/zq/dashgo_ws/src/cmoon/photo'
        # self.photopath = '/home/sundawn/catkin_ws/src/cmoon/photo'

        self.image1_path = self.photopath + '/photo.jpg'
        self.base = Base()

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def take_photo(self,device='camera'):
        """电脑摄像头拍照保存"""
        if device == 'k4a' or device == 'kinect':
            print("using k4a")


            # Initialize the library, if the library is not found, add the library path as argument
            pykinect.initialize_libraries()

            # Modify camera configuration
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            # print(device_config)

            # Start device
            device = pykinect.start_device(config=device_config)
            cv2.namedWindow('Face',cv2.WINDOW_NORMAL)
            # image1_path = self.photopath + '/photo.jpg'
            while True:

                # Get capture
                capture = device.update()

                # Get the color image from the capture
                ret, color_image = capture.get_color_image()
                if not ret:
                    continue
                # resolution = [1280, 720]
                # resolution = [1980, 1080]
                cv2.imwrite(self.image1_path, color_image)
                if 'photo.jpg' in os.listdir(self.photopath):
                    capture.release_handle()
                    break
            
            device.stop_cameras()
            device.close()
            faceimg = cv2.imread(self.image1_path)
            cv2.imshow("Face",faceimg)
            cv2.waitKey(3000)
            cv2.destroyAllWindows() 

        else:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.open(0)
            flag, frame = cap.read()
            # print(self.photopath)
            # path = self.photopath + '/photo.jpg'
            cv2.imwrite(self.image1_path, frame)
            
            cv2.namedWindow("Face", cv2.WINDOW_NORMAL)  # 建立图像对象
            faceimg = cv2.imread(self.image1_path)
            cv2.imshow("Face",faceimg)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            cap.release()

        return self.image1_path

    def get_attr(self, *key):
        return

    def detect(self, attributes=None, device='camera', mode=None, *keys):
        """电脑摄像头拍照检测"""
        path = self.take_photo(device)
        print("saving path:",path)
        result = self.get_attr(path, attributes)
        return result

class YoloResult:
    """
    Yolo检测结果数据类型

    :param name: 标签名称
    :param box: 矩形框左上右下xyxy坐标
    :param x: 矩形框中心点x坐标
    :param y: 矩形框中心点y坐标
    :param conf: 置信度
    :param distance: 中心点深度值（暂未接入）
    """

    def __init__(self, name: str, box: List[int], x: int, y: int, isVertical: bool, conf: float, distance: float = None):
        self.name = name
        self.box = box
        self.x = x
        self.y = y
        self.conf = conf
        self.distance = distance
        self.isVertical = isVertical

    def __str__(self):
        return f'name:{self.name}; box:{self.box}; x:{self.x}; y:{self.y}; isvertical:{self.isVertical}; conf:{self.conf:.2f}'


class ObjectDetector:
    """
    Object Detection YoloV7

    :param weights: .pt模型路径
    :param imgsz: 图片大小
    :param conf_thres: 置信度阈值
    :param iou_thres: iou阈值
    """

    def __init__(self, deviceInput, weights: Path = Path("weights", "yolov7.pt"), imgsz: int = 640,
                 conf_thres: float = 0.5, iou_thres: float = 0.45):
        # self.weights = weights
        self.photo_path = Path(Path.cwd(), "images")
        self.device = select_device()
        self.save_dir = Path(Path.cwd(), "results")
        # self.half = self.device.type != 'cpu'
        self.half = False
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = None
        self.stride = None
        self.names = None
        self.colors = None
        self.deviceInput = deviceInput
        self.K = np.array([[614.86962890625, 0.0, 635.5834350585938],
                           [0.0, 614.7677612304688, 364.99200439453125],
                           [0.0, 0.0, 1.0]], dtype = np.float32)
        self.D = np.array([0.6654,-2.73789,0.000626466, -0.0004694879], dtype = np.float32)
        #  1.48309,0.541851, -2.565876, 1.41643

        self.K = np.linalg.inv(self.K)

        self.base = Base()
        self.weights = r'/home/zq/catkin_ws/src/cmoon/src/weights/yolov7.pt'
        print(f"weights:{self.weights}\nUsing device:{self.device}")
        self.pub = rospy.Publisher('/yolo_result', Point, queue_size=10)
        self.point = Point()


    def load_model(self) -> torch.nn.Module:
        """加载模型"""
        model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        # model = TracedModel(model, self.device, self.imgsz)
        if self.half:
            model.half()
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        return model

    def process_img(self, img0: cv2_image) -> np.ndarray:
        """
        图像预处理

        :param img0: opencv读取的原始图像
        :return: 处理过的图像,直接传入模型
        """
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def process_result(self, pred: list, img: cv2_image, img0: cv2_image, t0, t1, t2, t3) -> List[YoloResult]:
        """
        处理检测结果

        :param pred: 模型输出结果
        :param img: 经process_img处理过的图像
        :param img0: opencv读取的原始图像
        :param t0: 开始时间
        :param t1: 图片预处理后时间
        :param t2: 推理时间
        :param t3: nms时间
        :return: 处理过的检测结果(YoloResult)
        """
        results = []
        names = []
        self.four = []
        for i, det in enumerate(pred):  # detections per image
            print("detdetdetdetdetdetdetdetdet:")
            # print(det)
            s = ''

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                print(det)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    label = f'{self.names[int(cls)]} {conf:.2f}'

                    # plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)

                    name = self.names[int(cls)]
                    box = [xyxy[:][0].item(), xyxy[:][1].item(), xyxy[:][2].item(), xyxy[:][3].item()]
                    print("+++++++++++++++++++++")
                    print(xyxy)
                    print("+++++++++++++++++++++")
                    self.four.append(box)
                    center = self.xyxy2cnt(box)
                    isVertical = self.verticalJudge(box)
                    result = YoloResult(name, box, center[0], center[1], isVertical, float(conf))
                    results.append(result)
                    names.append(name)
                # if len(results):
                #     for classes in results:
                #         names.append(self.names[str(classes[0])])
                #         classes[0] = self.names[classes[0]]

            # Print time (inference + NMS)
            # print(f'{s}fps:{(1 / (t3 - t0)):.2f}. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        return names, results

    def pred(self, model, img0: cv2_image) -> List[YoloResult]:
        """模型推理

        :param model: 调用load_model加载的模型
        :param img0: opencv读取的原始图像
        :return: 处理过的检测结果
        """
        t0 = time_synchronized()
        img = self.process_img(img0)
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            print("sssssssssssssssssss7777777777777")
            print(img.mode())
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
        # print(pred,"09090909090909090909090909")
        t3 = time_synchronized()
        name, results = self.process_result(pred, img, img0, t0, t1, t2, t3)
        return name, results

    def verticalJudge(self, xyxy: List[int]) -> bool:
        """
        判断物品是横放还是竖放

        :param xyxy: [x,y,x,y]
        :return: flag
        """
        if (xyxy[2] - xyxy[0]) > (xyxy[3] - xyxy[1]): 
            flag = False
        else:
            flag = True
        return flag

    def xyxy2cnt(self, xyxy: List[int]) -> List[int]:
        """
        xyxy坐标转中心点坐标

        :param xyxy: [x,y,x,y]
        :return: [x,y]
        """
        center = [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
        self.c = center
        return center

    def show(self, img: cv2_image, results: List[YoloResult], resolution: List[int] = (640, 480),
             range: float = 1.0) -> NoReturn:
        """
        opencv显示图片

        :param img: opencv读取的原始图片
        :param results: List of YoloResult(每个YoloResult代表一个物品)
        :param resolution: 图片分辨率[宽,高]
        :param range: 画面中心多大范围内的检测结果被采用
        :return: None
        """
        cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
        width = resolution[0]
        height = resolution[1]
        cv2.resizeWindow('yolo', width, height)
        cv2.line(img, (int(width * 0.5 * (1 - range)), 0), (int(width * 0.5 * (1 - range)), height),
                 (0, 255, 0), 1, 4)
        cv2.line(img, (int(width * 0.5 * (1 + range)), 0), (int(width * 0.5 * (1 + range)), height),
                 (0, 255, 0), 1, 4)
        for item in results:
            cv2.circle(img, (int(item.x), int(item.y)), 1, (0, 0, 255), 8)
        cv2.imshow("yolo", img)
        cv2.waitKey(1)

    def in_range(self, xs: List[int], width: int = 640, range: float = 1.0) -> bool:
        """
        判断物品中心点是否在设定的画面范围内

        :param xs: 一张图像上检测到的物品中心x坐标
        :param width: 图片宽度
        :param range: 画面中心多大范围内的检测结果被采用
        :return:
        """
        left = width * 0.1 * (1 - range)#改前是两个0.5
        right = width * 1.0 * (1 + range)
        return any([left <= x <= right for x in xs])

    def judge(self, mode: str, results: List[YoloResult], width: int = 640, range: float = 1.0,
              find: str = None) -> bool:
        """
        判断是否退出检测

        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param results: List of YoloResult(每个YoloResult代表一个物品)
        :param width: 图片宽度
        :param range: 画面中心多大范围内的检测结果被采用
        :param find: 需要寻找的物品名称
        :return: True/False
        """

        items = {item.name: item.x for item in results}
        
        if find is not None:
            mode = "find"
        print("mode +" , mode)
        if mode == 'realtime':
            flag = cv2.waitKey(1) & 0xFF == ord('q')
        elif mode == 'find':
            flag = find in items and self.in_range([items[find]], width, range)
        elif mode == 'detect':
            flag = cv2.waitKey(1) and items != [] and self.in_range([item.x for item in results], width, range)
            print("else - 3")
        print( "judge :", flag)
        return flag


    def transform_(self, n_coorindate_cls, depth_to_color_image):
        target_coorindate = list()  # 存放目标点三维坐标 二维列表
        for name, coorindate_cls in n_coorindate_cls:
            # print("coor:",coorindate_cls)
            x1, y1, x2, y2 = coorindate_cls[0], coorindate_cls[1], coorindate_cls[2], coorindate_cls[3]
            center_x = round((x2 - x1) / 2 + x1)
            center_y = round((y2 - y1) / 2 + y1)
            # target_coorindate = list()  # 存放目标点三维坐标 二维列表
            # print("coor:%s,%s" % (center_x, center_y))
            dep = depth_to_color_image[center_y][center_x].item()
            print("depth1:",dep)

            if (dep >= 100):  # 大于10cmv
                # image_coorindate = np.array([center_x, center_y , dep // 2]).reshape(3, 1)  # 图像坐标
                # print(center_x,",",center_y)
                # point = np.array([center_x, center_y], dtype = np.float32)
                # camera_point = cv2.undistortPoints(point, self.K, self.D)

                # center_x_new = camera_point[0][0][0]
                # center_y_new = camera_point[0][0][1]
                # print("centerX_new : " ,center_x_new)
                
                # image_coorindate = np.array([center_x_new, center_y_new, dep]).reshape(3,1)

                image_coorindate = np.array([(center_x - 635.5834) * dep / 614.86962,
                                             (center_y - 364.9920) * dep / 614.76776,
                                              dep]).reshape(3, 1)  # 图像坐标
                
                # image_coorindate = np.array([center_x, center_y, dep]).reshape(3, 1)  # 相机坐标
                # camera_coorindate = np.dot(self.K, image_coorindate)  # 相机坐标
                # target_coorindate.append([name, camera_coorindate[0].item() / 1000,
                #                           camera_coorindate[1].item() / 1000,
                #                           camera_coorindate[2].item() / 1000,
                #                           ])

                target_coorindate.append([name, image_coorindate[0].item(),
                image_coorindate[1].item(),
                image_coorindate[2].item(),
                ])
                # self.point.result = name
                # self.point.x = camera_coorindate[0].item() / 1000
                # self.point.y = camera_coorindate[1].item() / 1000
                # self.point.z = camera_coorindate[2].item() / 1000
                self.point.x = image_coorindate[0].item() / 1000
                self.point.y = image_coorindate[1].item() / 1000
                self.point.z = image_coorindate[2].item() / 1000
                # print("point:",target_coorindate)
                self.pub.publish(self.point)

        return target_coorindate

    def run(self, device: str = "cam", mode: str = "realtime", range: float = 1.0, nosave: bool = False,
            find: str = None, classes: str = None, rotate=1.0, depth=False) -> List[YoloResult]:
        """
        运行检测

        :param device: 使用的设备(azure kinect/电脑摄像头)
        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param range: 画面中心多大范围内的检测结果被采用
        :param nosave: 是否保存图片到本地
        :param find: 需要寻找的物体名称
        :param classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
        :return: List of YoloResult(每个YoloResult代表一个物品)
        """
        _find = find
        model = self.load_model()
        if classes is not None:
            self.classes = [self.names.index(name) for name in classes.split(',')]
        results = []
        if device == "k4a" or device == "kinect":
            print("using k4a")

            # # Initialize the library, if the library is not found, add the library path as argument
            # pykinect.initialize_libraries()

            # # Modify camera configuration
            # device_config = pykinect.default_configuration
            # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P

            # # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            # device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
            # # print(device_config)

            # # Start device
            # device = pykinect.start_device(config=device_config)
            # print ("device start!")

            while True:
                if rotate:
                    self.base.rotate(rotate)
                    # print("rotate")
                    pass

                # Get capture
                capture = self.deviceInput.update()
                # Get the color image from the capture
                ret1, color_image = capture.get_color_image()
                retd, depth_image = capture.get_transformed_depth_image()
                img100 = color_image

                # 如果没收到返回值，则跳过这个循环
                if not ret1:
                    continue
                self.save_dir_2 = '/home/zq/catkin_ws/src/cmoon/src/results'
                img0 = color_image

                cv2.imwrite(str(Path(self.save_dir_2, "result_2.jpg")), img0)
                # img_depth = depth_image
                resolution = [1280, 720]
                # resolution = [1980, 1080]
                
                name, results = self.pred(model, img0)#错误在这!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if(len(self.four)==0):
                    continue
                lredg = min(self.four[0][0],1280-self.four[0][2])
                lredg = min(lredg,81)
                lredg-=1
                self.img3 = img100[int(self.four[0][1]-80):int(self.four[0][3]),int(self.four[0][0]-lredg):int(self.four[0][2]+lredg)]
                
                for item in results:
                    print(item)
                
                # self.show(img0, results, resolution, range)
               # self.show(img_depth, results, resolution, range)
                
                if results != list():
                    x = self.c[0]
                    y = self.c[1]
                    print("depth2:",depth_image[y][x])
                # print('show')
                
                if retd and name != list() and depth is True:
                    # retd,depth_image = capture.get_transformed_depth_image()
                    # print("depth:",depth_image)
                    
                    trans_result = self.transform_(zip(name, self.four), depth_image)
                    print("result_point:", trans_result)
                print("!----------------------------------------------------!")
                
                if self.judge(mode, results, resolution[0], range, find = _find):
                    print("111111111111111111111=======1111111111111111")
                    if not nosave:
                        print("here")
                        savename = "result"+str(name)+".jpg"
                        cv2.imwrite("/home/zq/catkin_ws/src/cmoon/src/results/"+savename, img0)
                    break
                
            cv2.destroyAllWindows()
            if rotate:
                self.base.stop()
                print('base_stop_k4a')

        # return results
        return trans_result



    def detect(self, device: str = "cam", mode: str = "find", range: float = 1.0, nosave: bool = False,
               find: str = None, classes: str = None, rotate=0.5, depth=False) -> List[YoloResult]:
        """
         运行检测

        :param device: 使用的设备(azure kinect/电脑摄像头)
        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param range: 画面中心多大范围内的检测结果被采用
        :param nosave: 是否保存图片到本地
        :param find: 需要寻找的物体名称
        :param classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
        :return: List of YoloResult(每个YoloResult代表一个物品)
        """
        depth_ = depth
        # classes = "scissors"
        with torch.no_grad():  # 推理时减少显存占用
            results = self.run(device, mode, range, nosave, find, classes,rotate, depth=depth_)
        box_four = self.four
        model = self.load_model()
        rotate = self.rotating_test(self.img3,model,box_four)
        return results,box_four,rotate
    
    def rotating_test(self,img,model,box0):
        # for i in range(0,4):
        #     box0[0][i] = int (box0[0][i])
        rot = 0
        hmax = 0
        self.iou_thres = 0.2
        self.conf_thres = 0.2
        width = 25
        height = 120
        bod = 50
        box0[0][2] =box0[0][2]-box0[0][0]+2*bod
        box0[0][3] =box0[0][3]-box0[0][1]+2*bod
        box0[0][0] = 0
        box0[0][1] = 0
        center = [1,1]
        center[0] = int((box0[0][0]+box0[0][2])/2)
        center[1] = int((box0[0][1]+box0[0][3])/2)
        # box = [center[0]-width/2,center[1]-height/2,center[0]+width/2,center[1]+height/2]
        # box = [center[0]-width/2,center[1]-height/2,center[0]+width/2,center[1]+height/2]
        img = cv2.copyMakeBorder(img,bod,bod,bod,bod,cv2.BORDER_CONSTANT,value=[230,240,253])

        for i in range(0,18):
            deg = i*10
            img_rot = imutils.rotate(img,angle=deg)
            # img_test_area = self.get_inbox(img_rot,box)
            # img_test_area = cv2.copyMakeBorder(img_test_area,bod,bod,bod,bod,cv2.BORDER_CONSTANT,value=[255,255,255])
        
            name1,resultex = self.pred(model,img_rot)
            if(len(resultex) == 0):
                continue
            dh =self.four[0][3]-self.four[0][1]
            if(dh>hmax):
                rot = deg
                hmax = dh
            print(dh)
            print(deg)
            print(name1)
            print(resultex[0].conf)
            # cv2.imshow("rotate",img_rot)
            # cv2.waitKey(0)
        # l = 12
        # r = 22
        # st = 0
        # ed = 36
        # fh = []
        # for i in range(0,36):
        #     fh.append(0)
        # resultl = []
        # while(len(resultl) == 0):
        #     l-=1
        #     img_rotl = imutils.rotate(img,angle=5*l)
        #     namel,resultl = self.pred(model,img_rotl)
        #     fh[l] =  self.four[0][3]-self.four[0][1]
        # resultr = []
        # while(len(resultr) == 0):
        #     r+=1
        #     img_rotr = imutils.rotate(img,angle=5*r)
        #     namer,resultr = self.pred(model,img_rotr)
        #     fh[r] =  self.four[0][3]-self.four[0][1]
        # if(fh[l]<fh[r]):
        #     ed = r+1
        # else:
        #     st = l

        # # if(fh[l]>fh[r]):
        # #     dh = fh[l]
        # #     rot = 5*l
        # # else:
        # #     dh = fh[r]
        # #     rot = 5*r

        # for i in range(st,ed):
        #     if(i == l or i == r):
        #         if(fh[i]>hmax):
        #             rot = 5*i
        #             hmax = fh[i]
        #         continue
        #     deg = i*5
        #     img_rot = imutils.rotate(img,angle=deg)
        #     name1,resultex = self.pred(model,img_rot)
        #     if(len(resultex) == 0):
        #         continue
        #     dh =self.four[0][3]-self.four[0][1]
        #     if(dh>hmax):
        #         rot = deg
        #         hmax = dh
        #     print(dh)
        #     print(deg)
        #     print(name1)
        #     print(resultex[0].conf)

                
            
            # cv2.imshow("rotate",img_rot)
            # cv2.waitKey(0)

        

        return rot
    def get_inbox(self,img,box):

        img = img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
        return img



if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        pykinect.initialize_libraries()
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device = pykinect.start_device(config=device_config)
        # bodyTracker = pykinect.start_body_tracker()
        print("Camera Started!")

        # # 物体识别
        yoloclass = "scissors,book,bottle"
        yolo = ObjectDetector(deviceInput=device)
        result = yolo.detect(device='kinect', mode='realtime', find="bottle", classes=yoloclass,rotate=0, range=0.5,depth=True)
        for object in result:
            print("name:{},box:{},x:{},y:{}".format(object.name, object.box, object.x, object.y))
        
        device.stop_cameras()
        device.close()

    except rospy.ROSInterruptException:
        pass