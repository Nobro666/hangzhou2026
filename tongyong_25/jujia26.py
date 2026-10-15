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
先在门口找人；没有找到时进入房间内搜索。
垃圾前往固定抓取站位，由RealSense确认并抓取。

摔倒和挥手的人机交互需要第二次靠近
底盘速度调快
未知人物未知动作随便蒙一个
添加超时直接自主离场，确保能拿到自主离场的分数

需要明确的地方：
问需求的人机交互问完还需不需要做别的

注意&TODO：
到场地后除了打需要导航到的点外，还要在赛场的四个角落各打一个点，确认赛场范围
更换模型路径，添加发现果皮
记得把注册主人改回3个
记得填写 Switch0 和 switch2 的导航点坐标
调整离场时间
离场位置记得要同步填写在看门狗的文件开头
猜测姓名根据实际名单修改
开关位置两个房间两个，不区分开与关
记得松急停！！！！！！！！！

姿态识别需要调参
语音识别
机械臂根据实际位置调参：人体正上方；开关正上方；垃圾桶上方


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
from detect_people import KinectCamera, PersonDetector
from camera_to_map import CoordinateConverter
from position_last_second import SmartGoalFinder
# from face_detect import Detector #单人版
from face_detect_tongyong import Detector #双人版
import subprocess
from geometry_msgs.msg import PoseWithCovarianceStamped


# 储存导航路径点
LOCATION = {  
    "chu":[[0.6337276630383335,0.27157631732486287,0.138],[0.0,0.0,-0.01756594836124237,0.9998457068258934]],
    "start":[[3.1083250386123114,-0.81632613391356,0.13799999999999998],[0.0,0.0,0.516131573496784,0.8565093104220958]],
    "room0":[[5.8351309604451576,-0.15796936893040414,0.138],[0.0,0.0,0.8045668952205202,0.5938620303700285]],
    "room1_door":[[9.691715766017138,1.5217604867275114,0.13800000000000004],[0.0,0.0,-0.24244331726898682,0.9701655724213313]],
    "room1":[[10.736798930398953,0.7448174998015451,0.13800000000000004],[0.0,0.0,-0.2574460563135963,0.9662926720660654]],
    "room2_door":[[6.537544435383971,4.3316348433177305,0.138],[0.0,0.0,0.9648918243192528,0.262647610615182]],
    "room2":[[5.5297892610434545,5.264583432451764,0.138],[0.0,0.0,0.9994158511783444,0.03417537729807001]],
    "room3_door":[[8.22298229239249,4.349397083586869,0.138],[0.0,0.0,0.09526803280949551,0.9954516572514251]],
    "room3":[[12.327291034458657,3.304469254668874,0.13799999999999998],[0.0,0.0,0.805453873254884,0.5926584666219705]],
    "over":[[13.19744753608687,6.398889936126782,0.138],[0.0,0.0,0.6696931467824362,0.7426379260128306]],
    "switch0":[[6.716918665441541,2.908774259808008,0.138],[0.0,0.0,0.6763061213575394,0.7366206827223365]],
    "switch2":[[6.73580267040642,4.782803263482188,0.13800000000000004],[0.0,0.0,-0.03746954537696707,0.9992977700211501]],
    "trash_can":[[7.75411979092854,0.10896294607847778,0.13799999999999998],[0.0,0.0,-0.06640034611866587,0.9977930617293955]],
    "bed":[],
    "trash_1":[[9.278166391819127,-0.5184489134108972,0.138],[0.0,0.0,-0.7357672027548817,0.6772345408721093]],
    "trash_2":[[6.664121087244222,5.6827707243846675,0.13799999999999998],[0.0,0.0,0.5639161366401948,0.8258320597051177]],
    "trash_3":[[10.559512956171279,3.111156131558544,0.13799999999999998],[0.0,0.0,-0.26106838395348886,0.9653203089648087]]
}

# 主人名字
target_name = ["杰克","安东","李华"]

# ===== 2026-09-20 修改：统一定义行为识别结果，便于后续分发动作 =====
ACTION_SIT = "坐下"
ACTION_LIE = "躺下"
ACTION_FALL = "摔倒"
ACTION_WAVE = "挥手"
ACTION_UNKNOWN = "躺着"

# 房间内分段旋转搜索参数：检测 3 个方向，每次左转 120°，完成一整圈。
ROOM_SCAN_VIEW_COUNT = 3
ROOM_SCAN_STEP_DEGREES = 120.0
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

ROOM_DISPLAY_NAMES = {
    "room0": "客厅",
    "room1": "厨房",
    "room2": "卧室",
    "room3": "餐厅",
}

# 各房间垃圾点是机器人可直接到达的抓取站位，不是垃圾本身的坐标。
ROOM_TRASH_POINTS = {
    "room1": "trash_1",
    "room2": "trash_2",
    "room3": "trash_3",
}

# 人物有效区域（map坐标系，单位：米）。
# TODO：根据当前比赛地图填写矩形区域的最小/最大X、Y坐标。
# 参数未填写或范围无效时，人物候选会被拒绝并在终端提示。
PERSON_AREA_MIN_X = -10.0
PERSON_AREA_MAX_X = 10.0
PERSON_AREA_MIN_Y = -10.0
PERSON_AREA_MAX_Y = 10.0

# 从上述矩形边界向场内收缩的安全余量，单位：米。
# 不需要向内收缩时保持0.0；边界附近容易误收场外人员时可设为0.1～0.2。
PERSON_AREA_MARGIN = 0.0

# 人物识别需要保留较远距离，确保站立、坐下或躺下时脸部能够进入画面。
OWNER_GOAL_MAX_RADIUS = 1.8
OWNER_GOAL_MIN_RADIUS = 1.4

# 完成人脸和姿态识别后，使用同一个人物地图坐标再次规划更近的导航点。
OWNER_CLOSE_MAX_RADIUS = 1.0
OWNER_CLOSE_MIN_RADIUS = 0.6

# Azure Kinect Body Tracking动态库及行为识别采样帧数。
# 可通过环境变量覆盖动态库路径，便于不同机器人部署。
BEHAVIOR_K4ABT_LIB_PATH = os.environ.get("K4ABT_LIB_PATH","/lib/libk4abt.so")
BEHAVIOR_RECOGNIZE_FRAMES = 20

# 固定站位抓取时由RealSense检测的目标类别。
TRASH_TARGET_CLASSES = ["empty_bottle", "paper_ball"]
TRASH_TARGET_COUNT = 3
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
        self.photo_path = '/home/zq/catkin_ws/src/cmoon/src/hangzhou2026/tongyong_25/face'  # 替换为你想要保存照片的路径
        self.face = Detector(self.photo_path)
        print("==============视觉初始化完成==============")

        self.speak = SummerTTSSpeaker()
        self.voice = CompetitionVoiceService(owner_names=target_name, speaker=self.speak)
        print("==============语音初始化完成==============")

        self.person_info = {}
        self.recognized_owner_ids = set()
        self.owner_observations = {}
        self.valid_person_map_coords = None
        self.cleaned_trash_count = 0
        self.trash_observations = []
      
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
        pos_x = 0.6337276630383335  # <-- 在这里填入你的X坐标
        pos_y = 0.27157631732486287  # <-- 在这里填入你的Y坐标
        
        # 3. 设置你的机器人的初始朝向 (四元数)
        quat_x = 0.0  # <-- 在这里填入你的四元数X
        quat_y = 0.0  # <-- 在这里填入你的四元数Y
        quat_z = -0.01756594836124237  # <-- 在这里填入你的四元数Z
        quat_w = 0.9998457068258934  # <-- 在这里填入你的四元数W
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

        self.voice.say(f"主人{owner_index}", wait=True)

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
                retry_prompt="",
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
        return register_result
    


    def find_room(self):
        """
        巡游四个房间：门口先找人，未找到再进入房间搜索；
        厨房、卧室、餐厅各前往一个固定站位抓取垃圾。
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
            if room_name == "room0":
                # 主人注册在start点完成，此时不重复导航。
                print("room0使用当前start点进行门口检测")
            else:
                if not self.navigator.goto(door_name):
                    print(f"无法到达{door_name}，尝试直接进入{room_name}")
                    door_scan_available = False

            observation = None
            if door_scan_available:
                person_candidate = self.find_targets_at_door(room_name, door_name)
                if person_candidate is not None:
                    print(
                        f"{door_name}处理人物候选，"
                        f"距离={person_candidate['distance']:.2f}m，"
                        f"置信度={person_candidate['confidence']:.2f}"
                    )
                    observation = self.people_room(room_name, person_result=person_candidate)
                    if observation is not None:
                        room_results.append(observation)

            if observation is None:
                print(f"{door_name}未完成人物任务，进入{room_name}继续搜索")
                if self.navigator.goto(room_name):
                    self.voice.say(room_arrival_speech[room_name], wait=False)
                    observation = self.people_room(room_name)
                else:
                    print(f"无法到达{room_name}，跳过房间内的人物搜索")
                if observation is not None:
                    room_results.append(observation)

            if room_name in ROOM_TRASH_POINTS:
                self.trash_room(room_name)

            if (len(self.recognized_owner_ids) >= 3 and self.cleaned_trash_count >= TRASH_TARGET_COUNT):
                print("三位主人任务和三个垃圾任务均已完成")
                break
        print("巡游房间结束")
        print(f"共完成主人任务数量：{len(self.recognized_owner_ids)}")
        print(f"共完成垃圾投放数量：{self.cleaned_trash_count}")
        return room_results


    def find_targets_at_door(self, room_name, door_name):
        """在房间门口的预设朝向检测一次人物。"""
        person_result = None
        self.valid_person_map_coords = None

        try:
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
        except Exception as error:
            print(f"{door_name}门口快速检测发生异常：{error}")
        finally:
            try:
                self.camera.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

        return person_result


    def people_room(self, room_name, person_result=None):
        """执行当前房间的人物搜索、身份/行为识别和人机交互。"""
        if person_result is None:
            person_result = self.find_owner_in_room(room_name)
        if person_result is None:
            print(f"{room_name}中没有找到主人")
            return None

        if not self.approach_owner(person_result["map_coords"]):
            print(f"无法接近{room_name}中的人物")
            return None

        owner_result = self.recognize_owner()
        owner_recognized = owner_result is not None
        if not owner_recognized:
            print(f"{room_name}没有匹配到已注册主人，按杰克继续后续流程")
            face_id = None
            owner_index = 1
            person_name = "杰克"
            self.voice.say("发现杰克", wait=True)
        else:
            face_id = owner_result["face_id"]
            owner_index = owner_result["owner_index"]
            person_name = owner_result["person_name"]
            if face_id in self.recognized_owner_ids:
                print(f"{person_name}已经完成识别，跳过重复交互")
                return None
            self.voice.announce_owner_recognized(person_name, owner_index)

        behavior = self.recognize_behavior(face_id, person_name)
        self.voice.say(
            f"{person_name}在{ROOM_DISPLAY_NAMES[room_name]}{behavior}",
            wait=True,
        )

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
        # 摔倒救助和挥手问需求前需要第二次靠近；坐下、躺下无需靠近。
        if behavior in (ACTION_FALL, ACTION_WAVE):
            if not self.approach_owner_closer(person_result["map_coords"]):
                print(f"无法进一步接近{room_name}中的主人，取消{behavior}交互")
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


    def trash_room(self, room_name):
        """前往当前房间的固定抓取站位，用RealSense确认、抓取并投放垃圾。"""
        if self.cleaned_trash_count >= TRASH_TARGET_COUNT:
            print("三个垃圾均已投放，跳过后续垃圾搜索")
            return None

        trash_point = ROOM_TRASH_POINTS.get(room_name)
        if trash_point is None or not self.location.get(trash_point):
            print(f"{room_name}没有可用的固定垃圾抓取站位")
            return None

        if not self.navigator.goto(trash_point):
            print(f"无法到达{room_name}的垃圾抓取站位{trash_point}")
            return None

        try:
            # 固定站位只确定机械臂工作位置；物品类别及精确位置由RealSense确认。
            from catch_ty import KinovaRobotGroud

            grasp_result = KinovaRobotGroud.catch_ground(
                self.kinova,
                weights_path=TRASH_MODEL_PATH,
                target_items=TRASH_TARGET_CLASSES,
            )
            if not grasp_result:
                print(f"{trash_point}未检测到可抓取垃圾或抓取失败")
                return None

            trash_name = grasp_result.name
            if trash_name == "empty_bottle":
                self.voice.say("发现空瓶", wait=True)
            elif trash_name == "paper_ball":
                self.voice.say("发现纸团", wait=True)

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
        trash_result = {
            "room_name": room_name,
            "trash_name": trash_name,
            "grasp_point": trash_point,
            "put_success": True,
        }
        self.trash_observations.append(trash_result)
        print(
            f"{room_name}垃圾已投放，"
            f"完成数量={self.cleaned_trash_count}/{TRASH_TARGET_COUNT}"
        )
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
            rospy.logwarn_throttle(5.0, "人物有效区域参数尚未填写，忽略人物候选")
            return False

        min_x, max_x, min_y, max_y = map(float, bounds)
        margin = max(0.0, float(PERSON_AREA_MARGIN))
        valid_min_x = min_x + margin
        valid_max_x = max_x - margin
        valid_min_y = min_y + margin
        valid_max_y = max_y - margin
        if valid_min_x >= valid_max_x or valid_min_y >= valid_max_y:
            rospy.logerr_throttle(5.0, "人物有效区域参数无效，请检查最小值、最大值和边界余量")
            return False

        person_x = float(map_coords[0])
        person_y = float(map_coords[1])
        if not (valid_min_x <= person_x <= valid_max_x and valid_min_y <= person_y <= valid_max_y):
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
            return self.handle_switch_behavior(observation["room_name"], person_name)
        if behavior == ACTION_FALL:
            return self.handle_fall_behavior(person_name)
        if behavior == ACTION_WAVE:
            return self.handle_wave_behavior(person_name)

        print(f"{person_name}的行为尚未识别，暂不执行交互动作")
        self.voice.say("暂时没有识别出你的行为", wait=False)
        return False

    def handle_switch_behavior(self, room_name, person_name):
        """坐下或躺下：识别开/关需求，再前往所在房间操作开关。"""
        if rospy.is_shutdown():
            return False

        switch_point = {"room0": "switch0", "room2": "switch2"}.get(room_name)
        if switch_point is None:
            print(f"{room_name}不是客厅或卧室，没有对应开关导航点")
            return False
        if not self.location.get(switch_point):
            print(f"尚未填写{switch_point}的导航点坐标")
            return False

        action = None
        for _ in range(3):
            if rospy.is_shutdown():
                return False
            request_text = self.voice.ask_raw_text(
                prompt=f"{person_name}，你需要打开还是关闭开关？",
                duration=5.0,
                retries=1,
                repeat=False,
                retry_prompt="",
                free_grammar=False,
            ).strip()
            print(f"开关需求识别原文：{request_text}")
            # “开关”是目标名称，不能把其中的“开、关”误判为动作。
            action_text = request_text.replace("开关", "")
            has_open = any(word in action_text for word in ("打开", "开启", "开"))
            has_close = any(word in action_text for word in ("关闭", "关掉", "关"))
            if has_open == has_close:
                print("未能唯一确定打开或关闭，重新询问")
                continue
            parsed = self.voice.parser.parse_switch_command(action_text)
            if parsed.action in ("开", "关"):
                action = parsed.action
                break

        if action is None:
            print("连续三次未识别出明确的开关需求，不执行机械臂动作")
            return False

        action_word = "打开" if action == "开" else "关闭"
        self.voice.say(
            f"你的需求是{action_word}{ROOM_DISPLAY_NAMES[room_name]}的开关",
            wait=True,
        )

        try:
            if not self.navigator.goto(switch_point):
                print(f"无法到达开关导航点{switch_point}")
                return False
            if room_name == "room0":
                import switch0 as switch_controller
            else:
                import switch2 as switch_controller
            print(f"已到达{switch_point}，开始执行按钮动作")
            success = switch_controller.operate_switch()
            if success:
                self.voice.say("开关操作已完成", wait=False)
            else:
                self.voice.say("开关操作失败", wait=False)

            # switch0.py / switch2.py 的 operate_switch() 内部已回到 home。
            return success
        except (Exception, SystemExit) as error:
            print(f"执行开关动作发生异常：{error}")
            return False


    def handle_fall_behavior(self, person_name):
        """主人摔倒：机械臂直接伸到预先标定的固定位置。"""
        try:
            from arm2people_final import TARGET_POSE

            self.voice.say(f"{person_name}，我来帮助你", wait=False)
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
        """主人挥手：拼音匹配六种需求，只复述，不执行对应任务。"""

        if rospy.is_shutdown():
            return False

        for attempt in range(1, 4):
            request_text = self.voice.ask_raw_text(
                prompt=f"{person_name}，请问需要什么帮助？",
                duration=5.0,
                retries=1,
                repeat=False,
                free_grammar=True,
            ).strip()
            request, score = self.voice.parser.match_wave_help_request(request_text)
            print(f"挥手需求第 {attempt}/3 次：原文={request_text!r}，匹配={request}，分数={score:.2f}")
            if request:
                self.voice.say(f"好的，{request}", wait=True)
                return True

        print("连续三次未匹配到六种挥手需求")
        return False




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
                f"抓取站位={result['grasp_point']}"
            )

        """---自主离场---"""
        self.voice.say("开始自主离场", wait=False)
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
