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



# ===== 2026-10-01 语音统一接口改造 START =====
# 主函数不再直接调用 vosk_speech_recognition.record_and_recognize，
# 也不再直接用 self.speak.speak 做业务播报；统一走 CompetitionVoiceService。
# CompetitionVoiceService 内部负责：播报 -> 等 /summer_tts_done -> 录音 -> Vosk 转文字 -> 可选复述。
from summer_tts_speaker import SummerTTSSpeaker
from speech_2026 import CompetitionVoiceService
# ===== 2026-10-01 语音统一接口改造 END =====
from face_to_person import facetoPerson
from goal_calculator import calculate_facing_goal
import sys
sys.path.append(r"/home/zq/catkin_ws/src/cmoon/src")
import rospy
import os
import re
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
    "chu":[[0.10528231323935666,-0.07509295495687326,0.138],[0.0,0.0,0.0053423071277963925,0.9999857297754565]],
    "start":[[0.10528231323935666,-0.07509295495687326,0.138],[0.0,0.0,0.0053423071277963925,0.9999857297754565]],
    "room0":[[1.2703725035133533,-0.35325321400655685,0.13800000000000004],[0.0,0.0,0.005915822422323317,0.9999825013694327]],
    "room1":[[1.9122668550954658,-0.3784482835162455,0.13799999999999998],[0.0,0.0,0.0033810126717031125,0.9999942843603226]],
    "room2":[[2.3588365454624434,-0.3901218696327931,0.13800000000000004],[0.0,0.0,-0.048092393938254326,0.998842891372456]],
    "room3":[[2.921499798866684,-0.3978212835564952,0.13800000000000004],[0.0,0.0,-0.02891080342808905,0.9995819953586311]],
    "over":[[4.079833045916185,-0.3118363099353716,0.13799999999999998],[0.0,0.0,0.012550201701563646,0.9999212431173018]]
}

# 主人要求关键词
target_keywords = ["开","关"]

# 主人名字
target_name = ["张三","李四","王五"]

# ===== 2026-09-20 修改：统一定义行为识别结果，便于后续分发动作 =====
ACTION_SIT = "坐下"
ACTION_LIE = "躺下"
ACTION_FALL = "摔倒"
ACTION_WAVE = "挥手"
ACTION_UNKNOWN = "未知行为"


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

        # ===== 2026-10-01 语音统一接口改造 START =====
        # self.speak 仅作为底层 ROS TTS 发布器传给 CompetitionVoiceService；
        # 后续主流程不要直接 self.speak.speak(...)，统一使用 self.voice。
        self.speak = SummerTTSSpeaker()
        # 固定播报：self.voice.say(..., wait=True)
        # 交互原话：self.voice.ask_raw_text(...)，内部会先等播报 done 再录音识别并复述。
        self.voice = CompetitionVoiceService(
            owner_names=target_name,
            speaker=self.speak,
        )
        # ===== 2026-10-01 语音统一接口改造 END =====
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

    def resolve_owner_name_from_target(self, raw_text):
        """
        将姓名识别原文约束到 target_name。

        规则：
        1. 先调用 CompetitionVoiceService.parser.parse_owner_name，内部包含：
           直接文本匹配、字符重叠匹配、拼音相似度匹配；
        2. 如果能匹配到 target_name 中已有姓名，返回标准姓名；
        3. 如果仍无法匹配，认为是现场新增姓名，追加到 target_name，
           并同步到 CompetitionVoiceService，后续保持同一标准写法。
        """
        raw = (raw_text or "").strip()
        if not raw:
            return "", False

        matched_name = None
        score = 0.0
        if hasattr(self, "voice") and hasattr(self.voice, "parser"):
            matched_name, score = self.voice.parser.parse_owner_name(raw)
        if matched_name:
            print(f"姓名匹配到 target_name：raw={raw} -> {matched_name}，score={score:.2f}")
            return matched_name, False

        cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", raw)
        for prefix in ("我叫", "我的名字叫", "名字叫", "姓名是", "我是", "叫"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = raw

        target_name.append(cleaned)
        if hasattr(self, "voice"):
            if cleaned not in self.voice.owner_names:
                self.voice.owner_names.append(cleaned)
            if hasattr(self.voice, "parser") and cleaned not in self.voice.parser.owner_names:
                self.voice.parser.owner_names.append(cleaned)
        print(f"新增主人姓名到 target_name：{cleaned}；当前 target_name={target_name}")
        return cleaned, True

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
        pos_x = 0.07749469922910823  # <-- 在这里填入你的X坐标
        pos_y = -0.5406383437823923  # <-- 在这里填入你的Y坐标
        
        # 3. 设置你的机器人的初始朝向 (四元数)
        quat_x = 0.0  # <-- 在这里填入你的四元数X
        quat_y = 0.0  # <-- 在这里填入你的四元数Y
        quat_z = 0.008212284556930885  # <-- 在这里填入你的四元数Z
        quat_w = 0.9999662786226123  # <-- 在这里填入你的四元数W
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
        """
        注册一位主人：
        1) 固定播报引导主人站位；
        2) 调用人脸注册；
        3) 通过 CompetitionVoiceService.ask_raw_text 询问姓名，只取识别原文，不做词条匹配；
        4) 绑定 face_id、owner_index、person_name。
        """
        owner_index = id
        person_id = None
        max_attempts = 3
        face_attempts = 0

        self.voice.say(f"请主人{owner_index}站在我面前，保持静止", wait=True)

        while (person_id is None and
               face_attempts < max_attempts and
               not rospy.is_shutdown()):
            face_attempts += 1
            self.voice.say("开始人脸注册，请看向我", wait=True)
            print(f">>> 正在进行人脸注册...（第{face_attempts}/{max_attempts}次）")
            try:
                person_id = self.face.register_new_face()
            except Exception as error:
                print(f"人脸注册发生异常: {error}")
                person_id = None
            finally:
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

        # ===== 2026-10-01 姓名注册逻辑修正 START =====
        # 姓名注册不能只保存识别原文：先与 target_name 匹配，保证使用给定名单中的标准写法；
        # 若识别出的姓名不在 target_name，则认为现场新增姓名，追加到 target_name 并同步到语音服务。
        print(">>> [阶段2] 正在采集姓名...（先与 target_name 匹配；不在列表则新增）")
        raw_name = self.voice.ask_raw_text(
            prompt="你叫什么名字？",
            duration=5,
            retries=max_attempts,
            repeat=False,
            free_grammar=True,
        ).strip()
        person_name, name_added = self.resolve_owner_name_from_target(raw_name)

        if person_name:
            if name_added:
                self.voice.say(f"好的，新增主人姓名，{person_name}", wait=True)
            else:
                self.voice.say(f"好的，{person_name}", wait=True)

        # ===== 2026-10-01 姓名注册逻辑修正 END =====
        if not person_name:
            if rospy.is_shutdown():
                print("ROS 已关闭，姓名采集终止")
            else:
                print(f"语音识别已达到最大尝试次数（{max_attempts}次）")
                self.voice.say("姓名识别失败，已达到最大尝试次数", wait=True)
            return None

        self.person_info[person_id] = {
            "owner_index": owner_index,
            "name": person_name,
        }

        register_result = {
            "person_id": person_id,
            "owner_index": owner_index,
            "person_name": person_name,
        }
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
            self.navigator.goto(room_name)
            time.sleep(1)

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

            self.voice.announce_owner_recognized(person_name, owner_index)

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
        在当前房间中检测人物。

        使用：
        - KinectCamera
        - PersonDetector.detect_person()

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
            has_person, camera_coords = (self.people_detector.detect_person(self.camera,max_distance=5.0))
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
            self.navigator.goto("current_person")
            return True
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

    # ===== 2026-10-01 交互原话逻辑改造 START =====
    def handle_switch_behavior(self, person_name):
        """
        主人坐下或躺下：只做“询问 -> 原话识别 -> 复述”。

        当前词条不完整，暂不在主流程里切割“开/关、颜色、编号”等字段，
        也不做关键词匹配；后续开关视觉定位和机械臂执行接好后，
        可直接使用返回的 switch_command 原文作为日志或上层大模型/规则解析输入。
        """
        if rospy.is_shutdown():
            return False

        switch_command = self.voice.ask_raw_text(
            prompt=f"{person_name}，请告诉我需要操作哪个开关",
            duration=5,
            retries=3,
            repeat=True,
            repeat_template="你的需求是，{}",
            free_grammar=True,
        ).strip()

        if not switch_command:
            print("未获取到开关需求")
            return False

        print(f"主人开关需求原文：{switch_command}")
        # 当前语音交互任务到“复述原话”为止；机械臂开关执行后续再接入。
        return True


    def handle_fall_behavior(self, person_name):
        """主人摔倒：当前先固定播报提示；人体定位和机械臂动作后续再接入。"""
        self.voice.say(f"{person_name}，请保持不动，我来帮助你", wait=True)
        print("摔倒人体定位和机械臂动作尚未接入，当前完成语音提示。")
        return True


    def handle_wave_behavior(self, person_name):
        """主人挥手：询问需求、识别中文原话并复述，不做意图/词条匹配。"""
        if rospy.is_shutdown():
            return False

        request_text = self.voice.ask_raw_text(
            prompt=f"{person_name}，请告诉我你的需求",
            duration=5,
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
    # ===== 2026-10-01 交互原话逻辑改造 END =====




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
        self.voice.say("开始巡游房间", wait=True)
        time.sleep(1)
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
        self.navigator.goto("over")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, required=False, default = 1)
    parser.add_argument('--d', type=int, required=False, default = 1)
    opt = parser.parse_args()
    try:
        Controller('reception')  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题，直到节点已经关闭
    except rospy.ROSInterruptException:
        pass
