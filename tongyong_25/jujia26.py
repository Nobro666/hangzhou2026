#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 居家生活机器人赛项主程序

任务流程：
1. 自主进场
2. 注册三位主人
3. 巡游四个房间
4. 识别主人身份及行为
5. 根据行为完成人机交互
6. 寻找并清理三个垃圾
7. 从出口自主离场


待完善功能：
3.识别主人行为   recognize_behavior()
4.根据行为完成人机交互   interact_with_human()
5.寻找并清理垃圾   find_and_clean_garbage()

函数复杂可新增文件
"""



from summer_tts_speaker import SummerTTSSpeaker
from speech_2026 import CompetitionVoiceService
from face_to_person import facetoPerson
from goal_calculator import calculate_facing_goal
import sys
sys.path.append(r"/home/zq/catkin_ws/src/cmoon/src")
import rospy
import os
import cv2
from navigator import Navigator  # 导航模块
from pathlib import Path
from base_controller import Base  # 底盘运动模块
from std_msgs.msg import String  # std_msgs中包含消息类型string，发布的消息类型为String，从String.data中可获得信息，
import datetime
import argparse
import time
from catch_ground.src.catch import KinovaRobot
from catch_ground.src.realsense_yolo11 import RealSenseYolo11Detector
from detect_people import KinectCamera, PersonDetector
# from catch_ground.src.detector_items import ItemsDetector
from catch_ground.src.detector_items_c import ItemsDetector
from camera_to_map import CoordinateConverter
from position_last_second import SmartGoalFinder
# from face_detect import Detector #单人版
from face_detect_tongyong import Detector #双人版
import subprocess
from geometry_msgs.msg import PoseWithCovarianceStamped


# 储存导航路径点
LOCATION = {  
    "chu":[[0.15546364296360587,0.057071174642044135,0.138],[0.0,0.0,-0.056701600114551796,0.9983911701054099]],
    "start":[[1.205113775169155,-0.01452724161048334,0.13799999999999998],[0.0,0.0,-0.06591762098512825,0.9978250684582247]],
    "room0":[[2.650209716268067,-0.2643584489602559,0.13800000000000004],[0.0,0.0,-0.09297985252935967,0.9956679903580403]],
    "room1":[[3.9809536318790713,-0.15082740142472903,0.138],[0.0,0.0,0.030418679119681522,0.9995372449091698]],
    "room2":[[6.433993107228845,-0.5246430951369397,0.13800000000000004],[0.0,0.0,-0.009768701490203735,0.9999522850972417]],
    "room3":[[7.157761356010285,-0.8594503723082786,0.13800000000000007],[0.0,0.0,-0.09600218141232529,0.9953811235723103]],
    "over":[[7.847181964561788,-1.0953564410234704,0.138],[0.0,0.0,-0.06472131177523377,0.9979033779891182]],
    "switch":[],
    "trash_can":[]
}

# 主人要求关键词
target_keywords = ["开","关"]

# 主人名字
target_name = ["张三","李四","赵二"]

# ===== 2026-09-20 修改：统一定义行为识别结果，便于后续分发动作 =====
ACTION_SIT = "坐下"
ACTION_LIE = "躺下"
ACTION_FALL = "摔倒"
ACTION_WAVE = "挥手"
ACTION_UNKNOWN = "未知行为"

# 房间内分段旋转搜索参数：检测 6 个方向，每次左转 60°，完成一整圈。
ROOM_SCAN_VIEW_COUNT = 6
ROOM_SCAN_STEP_DEGREES = 60.0
ROOM_SCAN_DETECT_TIMEOUT = 2.0
ROOM_SCAN_TURN_TIMEOUT = 8.0
ROOM_SCAN_SETTLE_SECONDS = 0.5


class Controller:
    def __init__(self, name):
        print("==============开始初始化==============")
        rospy.init_node(name, anonymous=True)  # 初始化ros节点
        # rospy.Subscriber('/start_signal', String, self.control)  # 创建订阅者订阅recognizer发出的地点作为启动信号
        self.base = Base()  # 实例化移动底盘模块
        self.location = LOCATION
        self.navigator = Navigator(self.location)
        self.transpoint = CoordinateConverter()
        self.goalpoint = SmartGoalFinder()
        self.ftp = facetoPerson()
        print("==============导航初始化完成==============")

        self.kinova = KinovaRobot("j2n6s300")
        print("==============机械臂初始化完成==============")
    
        self.detector = RealSenseYolo11Detector(weights=Path("/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/model/yolo11m.pt"))
        self.camera = KinectCamera()
        self.people_detector = PersonDetector()
        self.items_detector = ItemsDetector()
        self.photo_path = '/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/face'  # 替换为你想要保存照片的路径
        self.face = Detector(self.photo_path)
        print("==============视觉初始化完成==============")

        self.speak = SummerTTSSpeaker()
        self.voice = CompetitionVoiceService(
            owner_names=target_name,
            speaker=self.speak,
        )
        print("==============语音初始化完成==============")

        # 保存人脸 ID、主人编号和姓名
        self.person_info = {}
        # 保存已经识别过的主人，避免重复处理
        self.recognized_owner_ids = set()
        # 保存巡游识别结果
        self.owner_observations = {}
      
        self.publish_initial_pose()
        time.sleep(1)
        self.control()
        print("==============全部初始化完成==============")
    
    def execute_command(self,command):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        return output
    
   
    def grip_object_coor(self,list):
        """
        返回传入相机坐标 返回目标点
        """
        robot_pose = self.goalpoint.get_robot_pose()
        map_object = self.transpoint.get_map_coords(list)
        goal_object = calculate_facing_goal(robot_pose,map_object,0.65)
        return goal_object
    

    def publish_initial_pose(self):
        """
        等待指定时间后，发布一个固定的初始位姿到 /initialpose 话题。
        """
        # 初始化ROS节点
        #rospy.init_node('auto_initial_pose_publisher', anonymous=True)

        # --- 用户配置区域 ---
        # 1. 设置比赛开始前的等待时间（秒）
        #    这个时间应该足够长，以确保比赛已经开始，门已经打开。
        wait_duration = 5.0 

        # 2. 设置你的机器人在 'map' 坐标系下的初始坐标 (单位：米)
        pos_x = 0.15546364296360587  # <-- 在这里填入你的X坐标
        pos_y = 0.057071174642044135  # <-- 在这里填入你的Y坐标
        
        # 3. 设置你的机器人的初始朝向 (四元数)
        quat_x = 0.0  # <-- 在这里填入你的四元数X
        quat_y = 0.0  # <-- 在这里填入你的四元数Y
        quat_z = -0.056701600114551796  # <-- 在这里填入你的四元数Z
        quat_w = 0.9983911701054099  # <-- 在这里填入你的四元数W
        # --- 配置结束 ---

        # 创建一个发布者，发布到 /initialpose 话题
        # amcl 节点会订阅这个话题来获取初始位姿
        pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)

        # 等待，直到发布者准备好
        rospy.sleep(1.0)

        # 打印等待信息
        rospy.loginfo("等待 {} 秒后自动初始化机器人位姿...".format(wait_duration))
        rospy.sleep(wait_duration)

        # 创建 PoseWithCovarianceStamped 消息
        initial_pose_msg = PoseWithCovarianceStamped()
        
        # 填充消息头
        initial_pose_msg.header.stamp = rospy.Time.now()
        initial_pose_msg.header.frame_id = "map"  # 确保这个是你的地图坐标系名称

        # 填充位姿信息
        initial_pose_msg.pose.pose.position.x = pos_x
        initial_pose_msg.pose.pose.position.y = pos_y
        initial_pose_msg.pose.pose.position.z = 0 # 2D导航通常z为0

        initial_pose_msg.pose.pose.orientation.x = quat_x
        initial_pose_msg.pose.pose.orientation.y = quat_y
        initial_pose_msg.pose.pose.orientation.z = quat_z
        initial_pose_msg.pose.pose.orientation.w = quat_w

        # 填充协方差矩阵 (表示位姿的不确定性)
        # 对于一个确定的初始位置，可以设置较小的值
        initial_pose_msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685]

        # 发布消息
        rospy.loginfo("正在发布初始位姿到 /initialpose ...")
        pub.publish(initial_pose_msg)
        
        # 再次发布几次以确保 amcl 能够接收到
        rospy.sleep(0.5)
        pub.publish(initial_pose_msg)

        rospy.loginfo("初始位姿发布成功！节点将退出。")
    



    def register(self, id):
        """注册一位主人，并绑定人脸 ID、主人编号和姓名。"""
        owner_index = id
        person_id = None
        max_attempts = 3
        face_attempts = 0

        self.voice.say(
            f"请主人{owner_index}站在我面前，保持静止",
            wait=True,
        )

        while (person_id is None and
               face_attempts < max_attempts and
               not rospy.is_shutdown()):
            face_attempts += 1
            self.voice.say("开始人脸注册，请看向我", wait=True)
            print(f">>> 正在进行人脸注册...（第{face_attempts}/{max_attempts}次）")
            try:
                person_id = self.face.register_new_face(
                    prompt_callback=lambda prompt: self.voice.say(
                        prompt, wait=True
                    )
                )
            except Exception as error:
                print(f"人脸注册发生异常: {error}")
                person_id = None
            finally:
                # 人脸模块和房间人物检测使用不同的 K4A 封装。
                # 每次注册尝试后都释放相机，避免巡游时重复打开设备。
                try:
                    self.face.close_k4a()
                except Exception as close_error:
                    print(f"关闭人脸相机发生异常: {close_error}")

            if person_id is None:
                print("未检测到有效人脸")
                if face_attempts < max_attempts:
                    self.voice.say("注册失败，请再试一次", wait=True)

        if person_id is None:
            if rospy.is_shutdown():
                print("ROS 已关闭，人脸注册终止")
            else:
                print(f"人脸检测已达到最大尝试次数（{max_attempts}次）")
                self.voice.say("人脸注册失败", wait=True)
            return None

        print(f"人脸注册成功，ID: {person_id}")
        self.voice.say("人脸注册成功", wait=True)

        # --------------------------------------------------------------
        # 阶段 2：询问并识别主人姓名
        # --------------------------------------------------------------

        print(">>> [阶段2] 正在采集姓名...")

        try:
            person_name, name_score, raw_name = self.voice.ask_owner_name(
                retries=max_attempts,
                duration=5.0,
            )
            print(f"姓名识别原文: {raw_name}")
            print(f"姓名匹配结果: {person_name}，置信度: {name_score:.2%}")
        except Exception as error:
            print(f"姓名语音识别发生异常: {error}")
            person_name = None

        if person_name is None:
            if rospy.is_shutdown():
                print("ROS 已关闭，姓名采集终止")
            else:
                print(f"语音识别已达到最大尝试次数（{max_attempts}次）")
                self.voice.say("姓名识别失败", wait=True)
            return None

        # --------------------------------------------------------------
        # 阶段 3：绑定人脸 ID、主人编号和姓名
        # --------------------------------------------------------------

        self.person_info[person_id] = {
            "owner_index": owner_index,
            "name": person_name,}
        
        register_result = {
            "person_id": person_id,
            "owner_index": owner_index,
            "person_name": person_name,}
        print("--------------------------------")
        print("主人注册完成")
        print(f"主人顺序: {owner_index}")
        print(f"人脸 ID: {person_id}")
        print(f"主人姓名: {person_name}")
        print("--------------------------------")
        self.voice.announce_owner_registered(person_name, owner_index)
        return register_result
    


    def find_room(self):
        """
        巡游四个房间，寻找主人并识别行为。

        当前只实现主要任务流程：
        1. 导航到房间；
        2. 搜索人物；
        3. 接近人物；
        4. 识别主人；
        5. 调用行为识别接口；
        6. 保存识别结果。
        """

        self.voice.say("开始巡游房间", wait=True)
        room_results = []

        for room_index in range(4):
            room_name = "room" + str(room_index)

            print("--------------------------------")
            print(f">>> 正在巡游房间：{room_name}")
            print("--------------------------------")
            if not self.navigator.goto(room_name):
                print(f"无法到达{room_name}，跳过该房间")
                continue

            # 使用 detect_people.py
            person_result = self.find_owner_in_room(room_name)
            if person_result is None:
                print(f"{room_name}中没有找到主人")
                continue

            # ===== 2026-09-20 修改：接近主人时传入转换后的地图坐标 =====
            # 使用 camera_to_map.py 和 position_last_second.py
            approach_success = self.approach_owner(person_result["map_coords"])
            if not approach_success:
                print(f"无法接近{room_name}中的人物")
                continue

            # 使用 face_detect_tongyong.py
            owner_result = self.recognize_owner()
            if owner_result is None:
                print(f"{room_name}中的人物不是已注册主人")
                self.voice.say("没有识别出主人", wait=True)
                continue

            face_id = owner_result["face_id"]
            owner_index = owner_result["owner_index"]
            person_name = owner_result["person_name"]

            # 避免重复识别同一个主人
            if face_id in self.recognized_owner_ids:
                print(f"{person_name}已经完成识别，跳过")
                continue

            self.voice.announce_owner_recognized(
                person_name,
                owner_index,
            )

            # ===== 2026-09-20 修改：统一调用行为识别接口 =====
            behavior = self.recognize_behavior(face_id, person_name)

            print("--------------------------------")
            print("主人识别结果")
            print(f"房间：{room_name}")
            print(f"主人编号：{owner_index}")
            print(f"人脸ID：{face_id}")
            print(f"姓名：{person_name}")
            print(f"行为：{behavior}")
            print("--------------------------------")

            self.voice.announce_behavior(behavior, person_name)

            observation = {
                "room_name": room_name,
                "face_id": face_id,
                "owner_index": owner_index,
                "person_name": person_name,
                "camera_coords": person_result[
                    "camera_coords"
                ],
                "map_coords": person_result[
                    "map_coords"
                ],
                "behavior": behavior,
            }

            # ===== 2026-09-20 修改：识别行为后执行对应的人机交互 =====
            interaction_success = self.interact_with_human(observation)
            observation["interaction_success"] = interaction_success
            self.owner_observations[face_id] = observation

            # 只有交互任务完成后，才将该主人标记为已完成。
            if interaction_success:
                self.recognized_owner_ids.add(face_id)
            room_results.append(observation)

            # 三位主人都找到后结束巡游
            if len(self.recognized_owner_ids) >= 3:
                print("三位主人均已找到")
                break

        # ===== 2026-09-20 修改：完成所有房间后再返回，避免只巡游第一个房间 =====
        print("巡游房间结束")
        print(f"共完成主人任务数量：{len(self.recognized_owner_ids)}")
        return room_results

    def find_owner_in_room(self, room_name):
        """
        在当前房间分段旋转一整圈并检测人物。

        使用：
        - KinectCamera
        - PersonDetector.detect_person()
        - Base.turn()

        返回：
        {
            "room_name": 房间名称,
            "camera_coords": 相机三维坐标,
            "map_coords": 地图坐标
        }
        """

        has_person = False
        camera_coords = None
        try:
            self.camera.open_camera()

            for view_index in range(ROOM_SCAN_VIEW_COUNT):
                if rospy.is_shutdown():
                    break

                current_angle = view_index * ROOM_SCAN_STEP_DEGREES
                print(
                    f"{room_name}人物扫描方向 "
                    f"{view_index + 1}/{ROOM_SCAN_VIEW_COUNT}，"
                    f"相对起始方向约 {current_angle:.0f}°"
                )
                has_person, camera_coords = (
                    self.people_detector.detect_person(
                        self.camera,
                        max_distance=5.0,
                        timeout=ROOM_SCAN_DETECT_TIMEOUT,
                    )
                )
                if has_person:
                    break

                # 每个方向未检测到人物后左转一段。最后一次也转动，
                # 使完整扫描结束时总转角为 360°，恢复到起始方向。
                turn_success = self.base.turn(
                    ROOM_SCAN_STEP_DEGREES,
                    timeout=ROOM_SCAN_TURN_TIMEOUT,
                )
                if not turn_success:
                    print(f"{room_name}底盘转向失败，终止该房间扫描")
                    break
                rospy.sleep(ROOM_SCAN_SETTLE_SECONDS)
        except Exception as error:
            print(f"{room_name}人物检测发生异常：{error}")

        finally:
            try:
                self.camera.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

        if not has_person:
            return None
        print(f"{room_name}检测到人物，"f"相机坐标：{camera_coords}")

        # 调用 camera_to_map.py
        map_coords = (self.transpoint.get_map_coords(camera_coords))

        if map_coords is None:
            print(f"{room_name}人物地图坐标转换失败")
            return None
        print(f"{room_name}人物地图坐标："f"{map_coords}")
        return {
            "room_name": room_name,
            "camera_coords": camera_coords,
            "map_coords": map_coords,
        }
    

    def approach_owner(self, person_map):
        """
        根据人物地图坐标，导航到人物附近的安全位置。
        """
        person_goal = (self.goalpoint.find_best_goal(person_map))
        if person_goal is None:
            print("没有找到人物附近的安全导航点")
            return False

        self.location["current_person"] = person_goal
        try:
            return self.navigator.goto("current_person")
        except Exception as error:
            print(f"导航到人物附近失败：{error}")
            return False
        
        
    def recognize_owner(self):
        """
        识别机器人面前的主人。

        detect_known_faces() 执行后，
        匹配到的人脸 ID 保存在 self.face.detect_result。
        """

        face_id = None
        try:
            self.face.detect_known_faces()
            face_id = self.face.detect_result
        except Exception as error:
            print(f"主人身份识别发生异常：{error}")

        finally:
            try:
                self.face.close_k4a()
            except Exception:
                pass
            cv2.destroyAllWindows()

        if face_id is None:
            return None
        person_info = self.person_info.get(face_id)
        if person_info is None:
            print(f"识别到人脸ID={face_id}，""但注册信息中没有对应姓名")
            return None
        return {
            "face_id": face_id,
            "owner_index": person_info["owner_index"],
            "person_name": person_info["name"],
        }
    
    def recognize_behavior(self, face_id, person_name):
        """
        识别主人行为。

        后续需要调用 Kinect 骨架或 YOLO Pose 实现：
        1. 坐下或躺下；
        2. 摔倒；
        3. 挥手。
        """

        print(f">>> 准备识别{person_name}的行为，人脸ID={face_id}")
        # TODO 未实现：接入 Kinect 骨架识别或 YOLO Pose，并返回以下常量之一：
        # ACTION_SIT_OR_LIE、ACTION_FALL、ACTION_WAVE。
        return ACTION_UNKNOWN

    # ===== 2026-09-20 修改：新增行为分发及后续机械臂动作主流程 =====
    def interact_with_human(self, observation):
        """根据识别出的行为，分发对应的人机交互任务。"""
        behavior = observation["behavior"]
        person_name = observation["person_name"]

        if behavior == ACTION_SIT or behavior == ACTION_LIE:
            return self.handle_switch_behavior(person_name)
        if behavior == ACTION_FALL:
            return self.handle_fall_behavior(person_name)
        if behavior == ACTION_WAVE:
            return self.handle_wave_behavior(person_name)

        print(f"{person_name}的行为尚未识别，暂不执行交互动作")
        self.voice.say("暂时没有识别出你的行为", wait=True)
        return False

    def handle_switch_behavior(self, person_name):
        """主人坐下或躺下：询问需求、识别开关标记、执行机械臂动作。"""

        # 1. 询问并识别开关需求
        if rospy.is_shutdown():
            return False

        switch_command = self.voice.ask_raw_text(
            prompt=f"{person_name}，请告诉我需要操作哪个开关",
            duration=5.0,
            retries=3,
            repeat=True,
            repeat_template="你的需求是，{}",
            free_grammar=True,
        ).strip()

        if not switch_command:
            print("未获取到开关需求")
            return False

        print(f"主人要求：{switch_command}")

        # 2. 根据需求识别对应开关标记
        marker_position = None
        # TODO 未实现：在这里直接调用现有目标检测模块，
        # 根据 switch_command 找到对应开关标记并获取三维坐标。
        # 再将坐标转换到机械臂控制接口要求的坐标系。

        if marker_position is None:
            print("开关标记定位尚未实现或未找到目标")
            return False

        # 3. 执行机械臂动作
        # TODO 未实现：机械臂函数暂时保留接口，返回 False。
        return self.move_arm_above_switch_marker(marker_position)


    def handle_fall_behavior(self, person_name):
        """主人摔倒：定位人体、执行机械臂动作。"""

        self.voice.say(f"{person_name}，我来帮助你", wait=True)

        # 1. 获取摔倒主人的身体位置
        body_position = None
        # TODO 未实现：在这里直接调用人体检测或骨架识别模块，
        # 获取目标身体部位的三维坐标，并计算其上方的目标位置。
        # 再将坐标转换到机械臂控制接口要求的坐标系。

        if body_position is None:
            print("摔倒人体定位尚未实现或未找到目标")
            return False

        # 2. 执行机械臂动作
        # TODO 未实现：机械臂函数暂时保留接口，返回 False。
        return self.move_arm_above_person(body_position)


    def handle_wave_behavior(self, person_name):
        """主人挥手：询问需求、识别中文语音并复述。"""

        if rospy.is_shutdown():
            return False

        request_text = self.voice.ask_raw_text(
            prompt=f"{person_name}，请告诉我你的需求",
            duration=5.0,
            retries=3,
            repeat=True,
            repeat_template="你的需求是，{}",
            free_grammar=True,
        ).strip()

        if not request_text:
            print("连续三次未识别到主人需求")
            return False

        print(f"主人需求原文：{request_text}")
        return True




    def control(self):
        text = None
        self.kinova.close_finger()
        self.navigator.goto("chu")
        self.navigator.goto("start")
        self.voice.say("已到达入场点", wait=True)
        time.sleep(1)


        """---注册主人---"""
        register_results = []
        for i in range(3):
            result = self.register(i + 1)
            if result is not None:
                register_results.append(result)
                print(
                    f"主人{i + 1}注册成功："
                    f"人脸ID={result['person_id']}，"
                    f"姓名={result['person_name']}")
            else:
                print(f"主人{i + 1}注册失败")

        print("三位主人注册流程结束")
        print(f"成功注册人数: {len(register_results)}")
        print(f"主人信息表: {self.person_info}")



        """---巡游四个房间---"""
        # self.navigator.goto("room0")
        # self.navigator.goto("room1")
        # self.navigator.goto("room2")
        # self.navigator.goto("room3")
        room_results = self.find_room()
        print("巡游主人识别结果：")

        for result in room_results:
            print("--------------------------------")
            print(f"房间：{result['room_name']}")
            print(f"主人编号：{result['owner_index']}")
            print(f"人脸ID：{result['face_id']}")
            print(f"姓名：{result['person_name']}")
            print(f"行为：{result['behavior']}")
            print("--------------------------------")



        "---捡垃圾并投放垃圾桶---"
        #待实现



        """---自主离场---"""
        self.voice.say("开始自主离场", wait=True)
        if not self.navigator.goto("over"):
            print("自主离场导航失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, required=False, default = 1)
    parser.add_argument('--d', type=int, required=False, default = 1)
    opt = parser.parse_args()
    try:
        Controller('jujia26')  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题，直到节点已经关闭
    except rospy.ROSInterruptException:
        pass
