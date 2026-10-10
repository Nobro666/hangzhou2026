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

优化思路：

地图范围过滤，人的坐标必须在场地坐标范围内
先在门口找一遍人和垃圾，先处理距离较近者，
再进入房间内部完成剩余任务

只有摔倒后的人机交互保留第二次靠近
底盘速度调快
未知人物未知动作随便蒙一个
添加超时直接自主离场，确保能拿到自主离场的分数

需要明确的地方：
问需求的人机交互问完还需不需要做别的

注意：
到场地后除了打需要导航到的点外，还要在赛场的四个角落各打一个点，确认赛场范围
"""



from summer_tts_speaker import SummerTTSSpeaker
from speech_2026 import CompetitionVoiceService
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
    "chu":[[0.13158331728377007,-0.006716505400027814,0.138],[0.0,0.0,0.0010637781280346942,0.9999994341878871]],
    "start":[[1.4442159164730417,0.12393709459396401,0.138],[0.0,0.0,0.15602370067869278,0.9877533117264278]],
    "room0":[[2.733324741708853,0.7250536048895461,0.138],[0.0,0.0,-0.005270137648095871,0.9999861127281569]],
    "room1_door":[],
    "room1":[[4.6003966923517154,1.0459468574130684,0.13800000000000004],[0.0,0.0,0.16525682855011448,0.986250566852845]],
    "room2_door":[],
    "room2":[[5.599555222316705,-1.3830625515214168,0.138],[0.0,0.0,0.12488564807055481,0.9921711419437664]],
    "room3_door":[],
    "room3":[[4.377265948267691,-2.533630478654654,0.138],[0.0,0.0,0.937220854044187,-0.34873639148314406]],
    "over":[[4.068038268894237,-4.397159735867198,0.138],[0.0,0.0,-0.7343875552222743,0.6787303726330884]],
    "switch":[[5.476704518261456,0.5897450893812475,0.13799999999999996],[0.0,0.0,0.06216837191063609,0.9980656759622488]],
    "trash_can":[[3.4399235099554977,-2.297665895918219,0.138],[0.0,0.0,0.9981220032837491,0.06125737964388872]],
    "bed":[]
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

# 房间内分段旋转搜索参数：检测 4 个方向，每次左转 90°，完成一整圈。
ROOM_SCAN_VIEW_COUNT = 4
ROOM_SCAN_STEP_DEGREES = 90.0
ROOM_SCAN_DETECT_TIMEOUT = 2.0
ROOM_SCAN_TURN_TIMEOUT = 8.0
ROOM_SCAN_SETTLE_SECONDS = 0.5

# 房间门口只按导航点预设朝向检测一次，不转动底盘。
ROOM_DOOR_POINTS = {
    "room0": "start",
    "room1": "room1_door",
    "room2": "room2_door",
    "room3": "room3_door",
}
DOOR_SCAN_DETECT_TIMEOUT = 2.0

# 人物有效区域（map坐标系，单位：米）。
# TODO：根据当前比赛地图填写矩形区域的最小/最大X、Y坐标。
# 参数未填写或范围无效时，人物候选会被拒绝并在终端提示。
PERSON_AREA_MIN_X = None
PERSON_AREA_MAX_X = None
PERSON_AREA_MIN_Y = None
PERSON_AREA_MAX_Y = None

# 从上述矩形边界向场内收缩的安全余量，单位：米。
# 不需要向内收缩时保持0.0；边界附近容易误收场外人员时可设为0.1～0.2。
PERSON_AREA_MARGIN = 0.0

# 人物识别需要保留较远距离，确保站立、坐下或躺下时脸部能够进入画面。
OWNER_GOAL_MAX_RADIUS = 1.8
OWNER_GOAL_MIN_RADIUS = 1.6

# 完成人脸和姿态识别后，使用同一个人物地图坐标再次规划更近的导航点。
OWNER_CLOSE_MAX_RADIUS = 1.1
OWNER_CLOSE_MIN_RADIUS = 0.8

# 垃圾抓取导航距离
TRASH_GOAL_MAX_RADIUS = 0.7
TRASH_GOAL_MIN_RADIUS = 0.5

# Azure Kinect Body Tracking动态库及行为识别采样帧数。
# 可通过环境变量覆盖动态库路径，便于不同机器人部署。
BEHAVIOR_K4ABT_LIB_PATH = os.environ.get("K4ABT_LIB_PATH","/lib/libk4abt.so")
BEHAVIOR_RECOGNIZE_FRAMES = 20

# 垃圾粗定位类别。应与Kinect检测模型和catch_ty.py中的RealSense
# 垃圾模型类别保持一致。
TRASH_TARGET_CLASSES = ["empty_bottle", "paper_ball"]
TRASH_TARGET_COUNT = 3
TRASH_SCAN_MAX_DISTANCE = 5.0
# 地图坐标系中地面垃圾允许的高度范围。超出范围的候选继续扫描，
# 不再生成导航目标；用于过滤桌椅、人体或背景上的误检。
TRASH_MAP_MIN_HEIGHT = -0.15
TRASH_MAP_MAX_HEIGHT = 0.45
TRASH_MODEL_PATH = os.environ.get(
    "TRASH_MODEL_PATH",
    "/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/model/best5.pt",
)


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
    
        # self.detector = RealSenseYolo11Detector(weights=Path("/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/model/yolo11m.pt"))
        self.camera = KinectCamera()
        self.people_detector = PersonDetector()
        # 垃圾模型按需加载，避免启动主程序时占用额外显存。
        self.items_detector = None
        self.photo_path = '/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/face'  # 替换为你想要保存照片的路径
        self.face = Detector(self.photo_path)
        print("==============视觉初始化完成==============")

        self.speak = SummerTTSSpeaker()
        self.voice = CompetitionVoiceService(owner_names=target_name, speaker=self.speak)
        print("==============语音初始化完成==============")

        # 保存人脸 ID、主人编号和姓名
        self.person_info = {}
        # 保存已经识别过的主人，避免重复处理
        self.recognized_owner_ids = set()
        # 保存巡游识别结果
        self.owner_observations = {}
        # 标记人物任务是否已经离开当前房间固定点位。没有检测到人物时
        # 保持 False，垃圾搜索前不再向同一个房间点重复导航。
        self.person_navigation_started = False
        # 保存通过地图区域检查的人物坐标，避免检测返回后重复转换。
        self.valid_person_map_coords = None
        # 保存已完成投放的垃圾数量及记录。
        self.cleaned_trash_count = 0
        self.trash_observations = []
        # 按开关机械臂按需初始化，避免主程序启动时重复等待Action Server。
        self.switch_arm = None
      
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

        先调用 CompetitionVoiceService 中的统一姓名解析器，依次执行
        直接文本、字符重叠和拼音相似度匹配。匹配成功时返回
        target_name 中的标准姓名；仍未匹配时，才将清理后的识别文本
        作为现场新增姓名，并同步加入语音服务的姓名列表。
        """
        raw = (raw_text or "").strip()
        if not raw:
            return "", False

        matched_name = None
        match_score = 0.0
        if hasattr(self, "voice") and hasattr(self.voice, "parser"):
            matched_name, match_score = (self.voice.parser.parse_owner_name(raw))
        if matched_name:
            print(
                f"姓名匹配到 target_name：raw={raw} -> "
                f"{matched_name}，score={match_score:.2f}"
            )
            return matched_name, False

        cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", raw)
        for prefix in (
            "我叫",
            "我的名字叫",
            "名字叫",
            "姓名是",
            "我是",
            "叫",
        ):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = raw

        for name in target_name:
            if name and (name in cleaned or cleaned in name):
                return name, False

        target_name.append(cleaned)

        if cleaned not in self.voice.owner_names:
            self.voice.owner_names.append(cleaned)
        if (hasattr(self.voice, "parser") and cleaned not in self.voice.parser.owner_names):
            self.voice.parser.owner_names.append(cleaned)

        print(
            f"新增主人姓名到 target_name：{cleaned}；"
            f"当前 target_name={target_name}"
        )
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
        pos_x = 0.4576881647866494  # <-- 在这里填入你的X坐标
        pos_y = -0.055454671844406035  # <-- 在这里填入你的Y坐标
        
        # 3. 设置你的机器人的初始朝向 (四元数)
        quat_x = 0.0  # <-- 在这里填入你的四元数X
        quat_y = 0.0  # <-- 在这里填入你的四元数Y
        quat_z = 0.013112045382736552  # <-- 在这里填入你的四元数Z
        quat_w = 0.9999140334378156  # <-- 在这里填入你的四元数W
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
        rospy.sleep(1.0)
        pub.publish(initial_pose_msg)

        rospy.loginfo("初始位姿发布成功！节点将退出。")
    



    def register(self, id):
        """注册一位主人，并绑定人脸 ID、主人编号和姓名。"""
        owner_index = id
        person_id = None
        max_attempts = 3
        face_attempts = 0

        self.voice.say(f"请主人{owner_index}站在我面前", wait=True)

        while (person_id is None and face_attempts < max_attempts and not rospy.is_shutdown()):
            face_attempts += 1
            print(f">>> 正在进行人脸注册...（第{face_attempts}/{max_attempts}次）")
            try:
                person_id = self.face.register_new_face(prompt_callback=lambda prompt: self.voice.say(prompt, wait=True))
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

        # ===== 2026-10-01 姓名注册逻辑修正 START =====
        print(
            ">>> [阶段2] 正在采集姓名..."
            "（先与 target_name 匹配；不在列表则新增）"
        )
        person_name = None
        name_added = False
        try:
            raw_name = self.voice.ask_raw_text(
                prompt="你叫什么名字？",
                duration=5,
                retries=max_attempts,
                repeat=False,
                free_grammar=True,
            ).strip()
            print(f"姓名识别原文: {raw_name}")
            person_name, name_added = (self.resolve_owner_name_from_target(raw_name))
        except Exception as error:
            print(f"姓名语音识别发生异常: {error}")

        if person_name:
                self.voice.say(f"好的，{person_name}",wait=True)
        # ===== 2026-10-01 姓名注册逻辑修正 END =====

        if not person_name:
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
        巡游四个房间。

        先在每个房间的门口朝预设方向各检测一次人物和垃圾；
        门口同时看到人物和垃圾时保存两组坐标，按距离由近到远
        依次处理。只看到一项时，先处理该项，再进入房间完成
        另一项任务；两者都没有时，进入房间完成两项任务。
        房间内保留原有的4方向分段旋转搜索。
        """
        self.voice.say("开始巡游房间", wait=True)
        room_results = []
        room_arrival_speech = {
            "room0": "到达客厅",
            "room1": "到达厨房",
            "room2": "到达卧室",
            "room3": "到达餐厅",
        }

        for room_index in range(4):
            room_name = "room" + str(room_index)
            print("--------------------------------")
            print(f">>> 正在巡游房间：{room_name}")
            print("--------------------------------")

            door_name = ROOM_DOOR_POINTS[room_name]
            door_scan_available = True
            door_person_done = False
            door_trash_done = False
            if room_name == "room0":
                # 主人注册在start点完成，此时不重复导航。
                print("room0使用当前start点进行门口检测")
            else:
                if not self.navigator.goto(door_name):
                    print(f"无法到达{door_name}，尝试直接进入{room_name}")
                    door_scan_available = False

            if door_scan_available:
                person_candidate, trash_candidate = (self.find_targets_at_door(room_name, door_name))
                door_candidates = []
                if person_candidate is not None:
                    door_candidates.append(("person", person_candidate["distance"], person_candidate))
                if trash_candidate is not None:
                    door_candidates.append(("trash", trash_candidate["distance"], trash_candidate))

                # 两者距离相同时垃圾优先
                door_candidates.sort(key=lambda item: (item[1], 0 if item[0] == "trash" else 1))
                if door_candidates:
                    order_text = " -> ".join(
                        f"{'人物' if kind == 'person' else '垃圾'}"
                        f"({distance:.2f}m)"
                        for kind, distance, _ in door_candidates
                    )
                    print(f"{door_name}按距离处理顺序：{order_text}")

                for candidate_kind, candidate_distance, candidate in door_candidates:
                    if candidate_kind == "person":
                        print(
                            f"{door_name}处理人物候选，"
                            f"距离={candidate_distance:.2f}m，"
                            f"置信度={candidate['confidence']:.2f}"
                        )
                        observation = self.people_room(room_name,person_result=candidate)
                        if observation is not None:
                            room_results.append(observation)
                            door_person_done = True
                    else:
                        print(
                            f"{door_name}处理垃圾候选，"
                            f"距离={candidate_distance:.2f}m，"
                            f"置信度={candidate['confidence']:.2f}"
                        )
                        trash_result = self.trash_room(room_name,trash_result=candidate)
                        door_trash_done = trash_result is not None

                if (len(self.recognized_owner_ids) >= 3 and
                        self.cleaned_trash_count >= TRASH_TARGET_COUNT):
                    print("三位主人任务和三个垃圾任务均已完成")
                    break

            if door_person_done and door_trash_done:
                print(f"{door_name}已完成人物和垃圾任务，进入下一个房间")
                continue
            if door_person_done:
                print(f"{door_name}已完成人物任务，进入{room_name}搜索垃圾")
            elif door_trash_done:
                print(f"{door_name}已完成垃圾任务，进入{room_name}搜索人物")
            else:
                print(f"{door_name}未完成任务，进入{room_name}搜索人物和垃圾")
            if not self.navigator.goto(room_name):
                print(f"无法到达{room_name}，跳过该房间")
                continue
            self.voice.say(room_arrival_speech[room_name], wait=True)

            interior_person_attempted = False
            if not door_person_done:
                interior_person_attempted = True
                observation = self.people_room(room_name)
                if observation is not None:
                    room_results.append(observation)

            if not door_trash_done:
                # 门口已完成人物任务时，机器人已重新导航到房间点，
                # 只需执行垃圾搜索。只有在房间内接近人物后才再次返回。
                if interior_person_attempted and self.person_navigation_started:
                    print(f"人物任务结束，返回{room_name}点位后再搜索垃圾")
                    if not self.navigator.goto(room_name):
                        print(f"无法返回{room_name}点位，跳过该房间的垃圾搜索")
                        continue
                elif interior_person_attempted:
                    print(f"{room_name}未接近人物，直接开始垃圾搜索")

                self.trash_room(room_name)

            if (len(self.recognized_owner_ids) >= 3 and self.cleaned_trash_count >= TRASH_TARGET_COUNT):
                print("三位主人任务和三个垃圾任务均已完成")
                break
        print("巡游房间结束")
        print(f"共完成主人任务数量：{len(self.recognized_owner_ids)}")
        print(f"共完成垃圾投放数量：{self.cleaned_trash_count}")
        return room_results


    def find_targets_at_door(self, room_name, door_name):
        """在房间门口的预设朝向检测一次人物和垃圾。"""
        person_result = None
        trash_result = None
        self.valid_person_map_coords = None

        try:
            if self.items_detector is None:
                self.items_detector = ItemsDetector(model_path=TRASH_MODEL_PATH)
            self.camera.open_camera()

            print(f"{door_name}单方向快速检测人物")
            has_person, person_coords, person_confidence = (
                self.people_detector.detect_person(
                    self.camera,
                    max_distance=5.0,
                    timeout=DOOR_SCAN_DETECT_TIMEOUT,
                    candidate_filter=self._person_candidate_in_person_area,
                    return_confidence=True,
                )
            )
            person_map_coords = self.valid_person_map_coords
            if has_person and person_map_coords is not None:
                person_distance = sum(float(value) ** 2 for value in person_coords) ** 0.5
                person_result = {
                    "room_name": room_name,
                    "camera_coords": person_coords,
                    "map_coords": person_map_coords,
                    "confidence": person_confidence,
                    "distance": person_distance,
                }

            print(f"{door_name}单方向快速检测垃圾")
            trash_name, trash_coords, trash_confidence = (
                self.items_detector.detect_targets(
                    self.camera,
                    target_items=TRASH_TARGET_CLASSES,
                    max_distance=TRASH_SCAN_MAX_DISTANCE,
                    timeout=DOOR_SCAN_DETECT_TIMEOUT,
                    return_confidence=True,
                )
            )
            if trash_name is not None:
                trash_map_coords = self.transpoint.get_map_coords(trash_coords)
                if trash_map_coords is None:
                    print(f"{door_name}的{trash_name}坐标转换失败")
                elif not (TRASH_MAP_MIN_HEIGHT <= trash_map_coords[2] <= TRASH_MAP_MAX_HEIGHT):
                    print(
                        f"忽略{door_name}的{trash_name}候选："
                        f"地图高度{trash_map_coords[2]:.3f}m不在地面范围内"
                    )
                else:
                    trash_distance = sum(float(value) ** 2 for value in trash_coords) ** 0.5
                    trash_result = {
                        "room_name": room_name,
                        "trash_name": trash_name,
                        "camera_coords": trash_coords,
                        "map_coords": trash_map_coords,
                        "confidence": trash_confidence,
                        "distance": trash_distance,
                    }
        except Exception as error:
            print(f"{door_name}门口快速检测发生异常：{error}")
        finally:
            try:
                self.camera.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

        return person_result, trash_result


    def people_room(self, room_name, person_result=None):
        """执行当前房间的人物搜索、身份/行为识别和人机交互。"""
        self.person_navigation_started = False
        if person_result is None:
            person_result = self.find_owner_in_room(room_name)
        if person_result is None:
            print(f"{room_name}中没有找到主人")
            return None

        # 从这里开始可能已经偏离房间固定点；即使接近导航中途失败，
        # 垃圾搜索前也应重新回到房间点位。
        self.person_navigation_started = True
        if not self.approach_owner(person_result["map_coords"]):
            print(f"无法接近{room_name}中的人物")
            return None

        owner_result = self.recognize_owner()
        owner_recognized = owner_result is not None
        if not owner_recognized:
            print(f"{room_name}中的人物不是已注册主人")
            self.voice.say("没有识别出主人", wait=True)
            face_id = None
            owner_index = None
            person_name = "未知人物"
        else:
            face_id = owner_result["face_id"]
            owner_index = owner_result["owner_index"]
            person_name = owner_result["person_name"]
            if face_id in self.recognized_owner_ids:
                print(f"{person_name}已经完成识别，跳过重复交互")
                return None
            self.voice.announce_owner_recognized(person_name, owner_index)

        behavior = self.recognize_behavior(face_id, person_name)
        self.voice.announce_behavior(behavior, person_name)

        observation = {
            "room_name": room_name,
            "face_id": face_id,
            "owner_index": owner_index,
            "person_name": person_name,
            "camera_coords": person_result["camera_coords"],
            "map_coords": person_result["map_coords"],
            "behavior": behavior,
        }
        observation_key = (face_id if face_id is not None else f"unknown_{room_name}")

        # 人脸和姿态识别均在第一次靠近的距离完成。
        # 只有摔倒救助动作需要在交互前第二次靠近主人；
        # 坐下、躺下和挥手直接在当前位置执行交互。
        if behavior == ACTION_FALL:
            if not self.approach_owner_closer(person_result["map_coords"]):
                print(f"无法进一步接近{room_name}中摔倒的主人，取消救助交互")
                observation["interaction_success"] = False
                self.owner_observations[observation_key] = observation
                return observation

        interaction_success = self.interact_with_human(observation)
        observation["interaction_success"] = interaction_success
        self.owner_observations[observation_key] = observation
        if interaction_success and owner_recognized:
            self.recognized_owner_ids.add(face_id)

        print("--------------------------------")
        print("主人识别结果")
        print(f"房间：{room_name}")
        print(f"主人编号：{owner_index}")
        print(f"人脸ID：{face_id}")
        print(f"姓名：{person_name}")
        print(f"行为：{behavior}")
        print("--------------------------------")
        return observation


    def trash_room(self, room_name, trash_result=None):
        """Kinect粗定位当前房间垃圾，接近后用RealSense抓取并投放。"""
        if self.cleaned_trash_count >= TRASH_TARGET_COUNT:
            print("三个垃圾均已投放，跳过后续垃圾搜索")
            return None
        
        if trash_result is None:
            trash_result = self.find_trash_in_room(room_name)
        else:
            self._announce_trash_found(trash_result, room_name)
        if trash_result is None:
            print(f"{room_name}中K4A没有找到垃圾")
            return None

        trash_goal = self.goalpoint.find_best_goal(
            trash_result["map_coords"],
            max_radius=TRASH_GOAL_MAX_RADIUS,
            min_radius=TRASH_GOAL_MIN_RADIUS,
        )
        if trash_goal is None:
            print(f"没有找到{room_name}垃圾附近的安全导航点")
            return None

        self.location["current_trash"] = trash_goal
        if not self.navigator.goto("current_trash"):
            print(f"无法接近{room_name}中的垃圾")
            return None

        try:
            # catch_ty中的方法与当前self.kinova使用相同的控制接口，
            # 直接复用实例，避免再次rospy.init_node和重复连接Action Server。
            from catch_ty import KinovaRobotGroud

            grasp_result = KinovaRobotGroud.catch_ground(
                self.kinova,
                weights_path=TRASH_MODEL_PATH,
                target_items=[trash_result["trash_name"]],
            )
            if not grasp_result:
                print(f"{room_name}垃圾抓取失败")
                return None

            if not self.navigator.goto("trash_can"):
                print("已执行垃圾抓取，但无法到达垃圾桶")
                return None

            put_result = KinovaRobotGroud.put_rubbish(self.kinova)
            if not put_result:
                print("到达垃圾桶，但垃圾投放失败")
                return None
        except (Exception, SystemExit) as error:
            print(f"{room_name}垃圾抓取或投放发生异常：{error}")
            return None

        self.cleaned_trash_count += 1
        trash_result["put_success"] = True
        self.trash_observations.append(trash_result)
        print(
            f"{room_name}垃圾已投放，"
            f"完成数量={self.cleaned_trash_count}/{TRASH_TARGET_COUNT}"
        )
        return trash_result


    def _announce_trash_found(self, trash_result, room_name):
        """统一播报并打印已确认的垃圾候选。"""
        found_name = trash_result["trash_name"]
        if found_name == "empty_bottle":
            self.voice.say("发现空瓶", wait=True)
        elif found_name == "paper_ball":
            self.voice.say("发现纸团", wait=True)
        else:
            self.voice.announce_trash_found(found_name, room_name)
        print(
            f"{room_name}发现垃圾：{found_name}，"
            f"相机坐标={trash_result['camera_coords']}，"
            f"地图坐标={trash_result['map_coords']}"
        )


    def find_trash_in_room(self, room_name):
        """使用头顶Kinect分段扫描垃圾并返回相机/地图坐标。"""
        found_name = None
        camera_coords = None
        map_coords = None
        try:
            if self.items_detector is None:
                self.items_detector = ItemsDetector(model_path=TRASH_MODEL_PATH)
            self.camera.open_camera()
            for view_index in range(ROOM_SCAN_VIEW_COUNT):
                if rospy.is_shutdown():
                    break

                print(
                    f"{room_name}垃圾扫描方向 "
                    f"{view_index + 1}/{ROOM_SCAN_VIEW_COUNT}"
                )
                target_name, coords = self.items_detector.detect_targets(
                    self.camera,
                    target_items=TRASH_TARGET_CLASSES,
                    max_distance=TRASH_SCAN_MAX_DISTANCE,
                    timeout=ROOM_SCAN_DETECT_TIMEOUT,
                )
                if target_name is not None:
                    candidate_map_coords = self.transpoint.get_map_coords(coords)
                    if candidate_map_coords is None:
                        print(
                            f"{room_name}的{target_name}候选坐标转换失败，"
                            "继续扫描"
                        )
                        continue

                    candidate_height = candidate_map_coords[2]
                    if not (
                        TRASH_MAP_MIN_HEIGHT <= candidate_height <=
                        TRASH_MAP_MAX_HEIGHT
                    ):
                        print(
                            f"忽略{room_name}的{target_name}候选："
                            f"地图高度{candidate_height:.3f}m不在地面范围"
                            f"[{TRASH_MAP_MIN_HEIGHT:.2f}, "
                            f"{TRASH_MAP_MAX_HEIGHT:.2f}]m内"
                        )
                        continue

                    found_name = target_name
                    camera_coords = coords
                    map_coords = candidate_map_coords
                if found_name is not None:
                    break

                if not self.base.turn(
                    ROOM_SCAN_STEP_DEGREES,
                    timeout=ROOM_SCAN_TURN_TIMEOUT,
                ):
                    print(f"{room_name}垃圾扫描转向失败")
                    break
                rospy.sleep(ROOM_SCAN_SETTLE_SECONDS)
        except Exception as error:
            print(f"{room_name}垃圾粗定位发生异常：{error}")
        finally:
            try:
                self.camera.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

        if found_name is None or camera_coords is None or map_coords is None:
            return None

        trash_result = {
            "room_name": room_name,
            "trash_name": found_name,
            "camera_coords": camera_coords,
            "map_coords": map_coords,
        }
        self._announce_trash_found(trash_result, room_name)
        return trash_result

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
        self.valid_person_map_coords = None
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
                        candidate_filter=self._person_candidate_in_person_area,
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

        # 候选过滤回调中已经完成坐标转换，直接复用转换结果。
        map_coords = self.valid_person_map_coords

        if map_coords is None:
            print(f"{room_name}人物地图坐标转换失败")
            return None
        print(f"{room_name}人物地图坐标："f"{map_coords}")
        return {
            "room_name": room_name,
            "camera_coords": camera_coords,
            "map_coords": map_coords,
        }

    def _person_candidate_in_person_area(self, camera_coords):
        """只接受预设矩形比赛区域内的人物候选。"""
        map_coords = self.transpoint.get_map_coords(camera_coords)
        if map_coords is None:
            print(f"忽略人物候选：相机坐标{camera_coords}转换失败")
            return False

        bounds = (
            PERSON_AREA_MIN_X,
            PERSON_AREA_MAX_X,
            PERSON_AREA_MIN_Y,
            PERSON_AREA_MAX_Y,
        )
        if any(value is None for value in bounds):
            rospy.logwarn_throttle(
                5.0,
                "人物有效区域参数尚未填写，忽略人物候选",
            )
            return False

        min_x, max_x, min_y, max_y = map(float, bounds)
        margin = max(0.0, float(PERSON_AREA_MARGIN))
        valid_min_x = min_x + margin
        valid_max_x = max_x - margin
        valid_min_y = min_y + margin
        valid_max_y = max_y - margin
        if valid_min_x >= valid_max_x or valid_min_y >= valid_max_y:
            rospy.logerr_throttle(
                5.0,
                "人物有效区域参数无效，请检查最小值、最大值和边界余量",
            )
            return False

        person_x = float(map_coords[0])
        person_y = float(map_coords[1])
        if not (
            valid_min_x <= person_x <= valid_max_x
            and valid_min_y <= person_y <= valid_max_y
        ):
            print(
                "忽略场外人物候选："
                f"相机坐标={camera_coords}，地图坐标={map_coords}，"
                f"有效范围=X[{valid_min_x:.2f}, {valid_max_x:.2f}]，"
                f"Y[{valid_min_y:.2f}, {valid_max_y:.2f}]"
            )
            return False

        self.valid_person_map_coords = map_coords
        print(
            "人物候选位于比赛区域内："
            f"相机坐标={camera_coords}，地图坐标={map_coords}"
        )
        return True
    

    def approach_owner(self, person_map):
        """
        根据人物地图坐标，导航到人物附近的安全位置。
        """
        person_goal = self.goalpoint.find_best_goal(
            person_map,
            max_radius=OWNER_GOAL_MAX_RADIUS,
            min_radius=OWNER_GOAL_MIN_RADIUS,
        )
        if person_goal is None:
            print("没有找到人物附近的安全导航点")
            return False

        self.location["current_person"] = person_goal
        try:
            return self.navigator.goto("current_person")
        except Exception as error:
            print(f"导航到人物附近失败：{error}")
            return False

    def approach_owner_closer(self, person_map):
        """
        人脸和姿态识别完成后，根据同一个人物地图坐标再次靠近。
        """
        person_close_goal = self.goalpoint.find_best_goal(
            person_map,
            max_radius=OWNER_CLOSE_MAX_RADIUS,
            min_radius=OWNER_CLOSE_MIN_RADIUS,
        )
        if person_close_goal is None:
            print("没有找到人物附近更近的安全导航点")
            return False

        self.location["current_person_close"] = person_close_goal
        try:
            print(
                "准备第二次接近主人，"
                f"目标距离范围：{OWNER_CLOSE_MIN_RADIUS}"
                f"～{OWNER_CLOSE_MAX_RADIUS}米"
            )
            return self.navigator.goto("current_person_close")
        except Exception as error:
            print(f"第二次接近主人失败：{error}")
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
        使用 behavior_detector.py 的 Azure Kinect 骨架识别器判断主人行为。

        人物搜索和人脸识别均会占用同一台Kinect，因此在这里按需创建
        BehaviorDetector，并在本次识别结束后立即释放设备。
        """

        print(f">>> 准备识别{person_name}的行为，人脸ID={face_id}")
        behavior_detector = None
        try:
            # 延迟导入，避免程序启动阶段因Body Tracking运行库问题
            # 影响导航、注册等不依赖姿态识别的功能。
            from behavior_detector import (
                BehaviorDetector,
                POSE_FALLEN,
                POSE_LYING,
                POSE_SITTING,
                POSE_STANDING,
                POSE_WAVING,
            )

            print(
                "正在启动Azure Kinect姿态识别，"
                f"采样帧数={BEHAVIOR_RECOGNIZE_FRAMES}"
            )
            behavior_detector = BehaviorDetector(BEHAVIOR_K4ABT_LIB_PATH)
            detected_behavior = behavior_detector.recognize(frames=BEHAVIOR_RECOGNIZE_FRAMES)
            print(f"behavior_detector原始结果：{detected_behavior}")

            behavior_mapping = {
                POSE_SITTING: ACTION_SIT,
                POSE_LYING: ACTION_LIE,
                POSE_FALLEN: ACTION_FALL,
                POSE_WAVING: ACTION_WAVE,
            }
            behavior = behavior_mapping.get(detected_behavior, ACTION_UNKNOWN)
            if detected_behavior == POSE_STANDING:
                print("检测到主人站立，但站立不属于当前任务行为")
            elif detected_behavior is None:
                print("姿态识别期间没有获得足够的有效骨架")
            print(f"转换后的任务行为：{behavior}")
            return behavior
        except (Exception, SystemExit) as error:
            print(f"姿态识别发生异常：{error}")
            return ACTION_UNKNOWN
        finally:
            if behavior_detector is not None:
                try:
                    behavior_detector.close()
                    print("姿态识别器和Kinect已释放")
                except Exception as close_error:
                    print(f"释放姿态识别器失败：{close_error}")
            cv2.destroyAllWindows()

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
        # 优先匹配完整动词，不能只判断单个“开、关”。
        if any(word in switch_command for word in ("关闭", "关掉", "关上", "关灯")):
            switch_action = "close"
        elif any(word in switch_command for word in ("打开", "开启", "开灯")):
            switch_action = "open"
        elif switch_command == "关":
            switch_action = "close"
        elif switch_command == "开":
            switch_action = "open"
        else:
            print(f"无法判断开关需求：{switch_command}")
            self.voice.say("没有听清需要打开还是关闭", wait=True)
            return False

        try:
            import switch as switch_controller
            # self.navigator.goto("switch")
            print(
                f"开始执行开关动作："
                f"{'打开' if switch_action == 'open' else '关闭'}"
            )
            success = switch_controller.go_to(switch_action)
            if success:
                self.voice.say("开关操作已完成", wait=True)
            else:
                self.voice.say("开关操作失败", wait=True)
            return success
        except (Exception, SystemExit) as error:
            print(f"执行开关动作发生异常：{error}")
            return False


    def handle_fall_behavior(self, person_name):
        """主人摔倒：机械臂直接伸到预先标定的固定位置。"""
        try:
            from arm2people_final import TARGET_POSE

            self.voice.say(f"{person_name}，我来帮助你", wait=True)
            print(f"摔倒救助机械臂目标位姿：{TARGET_POSE}")
            self.kinova.arm_run(unit="mq", pose_target=TARGET_POSE)
            print("摔倒救助机械臂动作已执行")

            home_pose = list(self.kinova.homePositionMdeg)
            print(f"摔倒救助动作完成，机械臂开始回原位：{home_pose}")
            self.kinova.arm_run(unit="mdeg", pose_target=home_pose)
            print("机械臂已回到原位")

            return True
        except (Exception, SystemExit) as error:
            print(f"执行摔倒救助机械臂动作发生异常：{error}")
            return False


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
        for i in range(1):
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
        time.sleep(5)
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

        print("房间垃圾处理结果：")
        print(f"已完成投放数量：{self.cleaned_trash_count}")
        for result in self.trash_observations:
            print(
                f"{result['room_name']}：{result['trash_name']}，"
                f"地图坐标={result['map_coords']}"
            )

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
