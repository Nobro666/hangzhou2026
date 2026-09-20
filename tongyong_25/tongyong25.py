from  summer_tts_speaker import SummerTTSSpeaker
from get_keyword import FuzzyKeywordMatcher
from face_to_person import facetoPerson
from goal_calculator import calculate_facing_goal
import sys
sys.path.append(r"/home/zq/catkin_ws/src/cmoon/src")
import rospy
import os
import cv2
from navigator import Navigator  # 导航模块
from pathlib import Path
# from chinese_tts.chinese_tts import speak
# from vosk_speech_recognition.vosk_speech_recognition import (
#     recognize_speech, 
#     recognize_from_file, 
#     record_and_recognize,
#     show_recognition_log,
#     clear_recognition_log
# )
from vosk_speech_recognition.vosk_speech_recognition import record_and_recognize,get_recognizer_instance 
#因模型较大 get_recognizer_instance ·提前加载模型
from base_controller import Base  # 底盘运动模块
from std_msgs.msg import String  # std_msgs中包含消息类型string，发布的消息类型为String，从String.data中可获得信息，
# 面部识别
from find_seat import Follower
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


cn2en = {
    "饼干":"Biscuit",
    "薯片":"Chip",
    "薯愿":"Lays",
    "曲奇":"Cookie",
    "洗手液":"Handwash",
    "洗洁精":"Dishsoap",
    "水":"Water",
    "雪碧":"Sprite",
    "可乐":"Cola",
    "芬达":"Orange juice",
    "洗发水":"Shampoo"
}

en2ch = {
    "Cola":"可乐",
    "Biscuit":"饼干",
    "Water":"水",
    "Dishsoap":"洗洁精",
    "Sprite":"雪碧",
    "Lays":"乐事薯片",
    "Chip":"薯片",
    "Orange juice":"芬达",
    "Handwash":"洗手液",
    "Cookie":"曲奇",
    "Shampoo":"洗发水"
}

fxz_ch = {
    "cola":"可乐",
    "biscuit":"饼干",
    "water":"水",
    "dishsoap":"洗洁精",
    "sprite":"雪碧",
    "lays":"乐事薯片",
    "chip":"薯片",
    "orange_juice":"芬达",
    "handwash":"洗手液",
    "cookie":"曲奇",
    "shampoo":"洗发水"
}

fxz_en = {
    "饼干":"biscuit",
    "薯片":"chip",
    "薯愿":"lays",
    "曲奇":"cookie",
    "洗手液":"handwash",
    "洗洁精":"dishsoap",
    "水":"water",
    "雪碧":"sprite",
    "可乐":"cola",
    "芬达":"orange_juice",
    "洗发水":"shampoo"
}

# 储存导航路径点
LOCATION = {  
    "chu":[[1.4772394901202788,-0.1074459089207907,0.138],[0.0,0.0,0.009568443632437487,0.999954221395386]],
    "start":[[1.8529739247400663,1.600456677934387,0.138],[0.0,0.0,0.4744579883761413,0.8802781476704198]],
    "findpeople0":[[2.048071561950728,1.0019788854951344,0.13799999999999998],[0.0,0.0,-0.022180430461259282,0.9997539839903381]],
    # "findpeople1":[[7.093495781106096,2.848914082290762,0.138],[-0.0,-0.0,0.4039637087081509,0.9147750117087569]],
    "findpeople1":[[7.739090970766754,2.7652174916338135,0.138],[-0.0,-0.0,0.48537297020932046,0.8743071998961133]],
    "findpeople2":[[9.105194983706733,6.659275465634871,0.13800000000000004],[0.0,0.0,-0.1876265347794262,0.9822404407509725]],
    # "findpeople4":[[10.50392047351025,3.4772769423656738,0.138],[0.0,0.0,0.27430479689633647,0.9616428018758626]],
    # "findpeople5":[[12.270588750407617,5.622214309151977,0.138],[0.0,0.0,0.9327557206953028,0.36050903665537476]],
    "finditem0":[[10.981630968740424,0.8247155549401685,0.138],[0.0,0.0,0.7027613591734619,0.7114256616489656]],
    "finditem1":[[12.880395098896749,0.632126301391674,0.138],[0.0,0.0,-0.021780457911442955,0.9997627776893716]],    
    "finditem2":[[11.758017084419718,0.08028025037711281,0.138],[0.0,0.0,-0.7219450871041688,0.6919503531368085]],
    # "catchitem00":[[10.827578094201728,1.773132115318352,0.138],[0.0,0.0,0.712857399783404,0.7013090100476709]],
    # "catchitem01":[[11.045866042851758,1.7695691590115685,0.138],[0.0,0.0,0.7178337144422177,0.6962145922128382]],    
    # "catchitem02":[[11.19627738050572,1.862519687976637,0.138],[0.0,0.0,0.7251101696882853,0.6886328788364856]],
    # "catchitem10":[[13.584358291762408,0.7808351209382638,0.138],[0.0,0.0,-0.026241568420047663,0.9996556307483372]],
    # "catchitem11":[[13.58096256903741,0.6033541054800855,0.138],[0.0,0.0,-0.009421054685890131,0.9999556208795496]],    
    # "catchitem12":[[13.607477070314856,0.43604472622303664,0.138],[0.0,0.0,-0.011906655971881232,0.9999291132593187]],
    # "catchitem20":[[11.895011078914106,-0.7301673025198324,0.138],[0.0,0.0,-0.6849296878356244,0.7286091700777545]],
    # "catchitem21":[[11.685720821806212,-0.732825852734131,0.138],[0.0,0.0,-0.6974932431033113,0.7165913590221594]],    
    # "catchitem22":[[11.475494609604562,-0.7149297545878077,0.138],[0.0,0.0,-0.7017492623839632,0.7124240119083324]],
    "rubbish0":[[7.6314030101831305,6.779877791942028,0.138],[0.0,0.0,0.8677982157263316,0.4969167503538145]],
    "rubbish1":[[9.952079120494568,6.367085883185631,0.138],[0.0,0.0,-0.7087765430100073,0.7054330670437723]],
    "catchrubbish":[[9.945219413204144,5.717162116900239,0.138],[0.0,0.0,-0.7170868452379824,0.6969838279233155]],
    "can":[[7.615016243162566,0.12344215729025354,0.138],[0.0,0.0,0.0008073016831941379,0.9999996741319431]],
    "over":[[12.730777003887127,8.653895617512477,0.138],[0.0,0.0,0.7073123393797158,0.7069011632195786]]
}

target_keywords = [
        "饼干", "薯片", "乐事薯片", "曲奇", "洗手液", "洗洁精", 
        "水", "雪碧", "可乐", "芬达", "洗发水"
]
#根据现场抽到的物品，修改target_keywords

class Controller:
    def __init__(self, name, room,drink):
        rospy.init_node(name, anonymous=True)  # 初始化ros节点
        self.findseat = rospy.Publisher('/findseat', String, queue_size=1)
        rospy.Subscriber('/start_signal', String, self.control)  # 创建订阅者订阅recognizer发出的地点作为启动信号
        self.base = Base()  # 实例化移动底盘模块
        self.location=LOCATION
        self.navigator=Navigator(self.location)
        self.fdseat=Follower(self.location)
        self.kinova = KinovaRobot("j2n6s300")
        self.detector = RealSenseYolo11Detector(weights=Path("/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/model/allbest.pt"))
        self.camera = KinectCamera()
        self.people_detector = PersonDetector()
        self.items_detector = ItemsDetector()
        self.transpoint = CoordinateConverter()
        self.goalpoint = SmartGoalFinder()
        self.ftp = facetoPerson()
        self.photo_path = '/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/face'  # 替换为你想要保存照片的路径
        self.face = Detector(self.photo_path)
        self.speak = SummerTTSSpeaker()
        get_recognizer_instance('zh') # 这会触发模型加载 后续调用recognize函数时不会重复加载
        self.matcher = FuzzyKeywordMatcher(keywords=target_keywords)        
        self.publish_initial_pose()
        time.sleep(1)
        self.control()
    
    def execute_command(self,command):
        result=subprocess.run(command, shell=True, capture_output=True, text=True)
        output=result.stdout.strip()
        return output
    """
    返回传入相机坐标 返回目标点
    """
    def grip_object_coor(self,list):
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
        pos_x = 0.2640757750552908  # <-- 在这里填入你的X坐标
        pos_y = -0.011357155403294445  # <-- 在这里填入你的Y坐标
        
        # 3. 设置你的机器人的初始朝向 (四元数)
        quat_x = 0.0  # <-- 在这里填入你的四元数X
        quat_y = 0.0  # <-- 在这里填入你的四元数Y
        quat_z = -0.008099910362467287  # <-- 在这里填入你的四元数Z
        quat_w = 0.9999671951879822  # <-- 在这里填入你的四元数W
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
        initial_pose_msg.pose.pose.position.z = 0.138 # 2D导航通常z为0

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


    def control(self):

        # self.navigator.goto('finditem2')
        # text = "可乐"
        # self.camera.open_camera()
        # has_target, coords = self.items_detector.detect(self.camera, target=cn2en[text], max_distance=5.0)
        # print(f"是否检测到目标: {has_target}")
        # self.camera.release()
        # if has_target:
        #     self.speak.speak(f"抓取{text}")
        #     print(f"三维坐标: x={coords[0]:.2f}m, y={coords[1]:.2f}m, z={coords[2]:.2f}m")
        #     item_map = self.grip_object_coor(coords)
        #     LOCATION["item_zx"] = item_map
        #     self.navigator.goto("item_zx")
        #     self.kinova.catch_table_short(target=[fxz_en[text]])
        # self.mmmmmmm()
        



        text = None 
        self.kinova.close_finger()
        self.navigator.goto("chu")
        self.navigator.goto("start")
        self.speak.speak("已到达入场点")
        time.sleep(1)

        """---寻找客人(两人版)---"""
        ke = False
        zhu = False
        for i in range(3):
            self.navigator.goto("findpeople"+str(i))
            if i == 0:
                self.camera.open_camera()
                has_person, coords = self.people_detector.detect_person(self.camera, max_distance=6.0)
                print(f"人体坐标:{coords}")
                self.camera.release()
                if has_person is not False:
                    zhu = True
                    zrposition = self.transpoint.get_map_coords(coords)
                    print(f"zrposition:{zrposition}")
                    ftpgoal = self.ftp.face_to_person(zrposition)
                    LOCATION["person_zr"] = ftpgoal
                    self.navigator.goto("person_zr")             
                    self.speak.speak("主人你好")
                    time.sleep(2)
            else:
                """寻找人,判断是否为客人"""
                print("检测人脸")
                faceresult = self.face.detect_known_faces()
                print(f"1人脸坐标:{faceresult}")
                self.face.close_k4a()
                list = [0,0,0]
                if  faceresult[0] == list and faceresult[1] != list:
                    ke = True
                    person_map = self.transpoint.get_map_coords(faceresult[1])
                    print(f"2人的位置:{person_map}")
                    testgoal = self.goalpoint.find_best_goal(person_map)
                    LOCATION["person_zx"] = testgoal
                    self.navigator.goto("person_zx")
                    self.speak.speak("客人你好")
                    time.sleep(1)
                    self.speak.speak("你要什么")
                    time.sleep(2)
                    # 中文识别
                    while True:
                        text0, audio_file = record_and_recognize('zh', duration=5)
                        print(audio_file)
                        print(f"识别结果: {text0}")
                        #提取关键词 原识别结果为text0 关键词为text
                        try:
                            text, score = self.matcher.find_best_match(text0, min_confidence=0.85)
                            print(f"匹配成功: '{text}' (相似度: {score:.2%})\n")
                        except:
                            pass
                        time.sleep(1)
                        if text in cn2en:
                            self.speak.speak(f"好的{text}")
                            break
                        else:
                            self.speak.speak("请再说一遍")
                            time.sleep(2)
                    # break
                    if ke == True and zhu == True:
                        break
                elif faceresult[0] != list and faceresult[1] == list:
                    zhu = True
                    zrposition = self.transpoint.get_map_coords(faceresult[0])
                    print(f"3人的位置:{zrposition}")
                    ftpgoal = self.ftp.face_to_person(zrposition)
                    LOCATION["person_zr"] = ftpgoal
                    self.navigator.goto("person_zr")
                    self.speak.speak("主人你好")
                    time.sleep(1)
                    if ke == True and zhu == True:
                        break
                elif faceresult[0] != list and faceresult[1] != list:
                    ke = True
                    zhu = True
                    zrposition = self.transpoint.get_map_coords(faceresult[0])
                    print(f"4人的位置:{zrposition}")
                    ftpgoal = self.ftp.face_to_person(zrposition)
                    LOCATION["person_zr"] = ftpgoal
                    self.navigator.goto("person_zr")
                    self.speak.speak("主人你好")
                    time.sleep(1)

                    person_map = self.transpoint.get_map_coords(faceresult[1])
                    print(f"6人的位置:{person_map}")
                    testgoal = self.goalpoint.find_best_goal(person_map)
                    LOCATION["person_zx"] = testgoal
                    self.navigator.goto("person_zx")
                    self.speak.speak("客人你好")
                    time.sleep(1)
                    self.speak.speak("你要什么")
                    time.sleep(2)
                    # 中文识别
                    while True:
                        text0, audio_file = record_and_recognize('zh', duration=5)
                        print(audio_file)
                        print(f"识别结果: {text0}")
                        #提取关键词 原识别结果为text0 关键词为text
                        try:
                            text, score = self.matcher.find_best_match(text0, min_confidence=0.85)
                            print(f"匹配成功: '{text}' (相似度: {score:.2%})\n")
                        except:
                            pass
                        time.sleep(1)
                        if text in cn2en:
                            self.speak.speak(f"好的{text}")
                            break
                        else:
                            self.speak.speak("请再说一遍")
                            time.sleep(2)
                    if ke == True and zhu == True:
                        break

        # ke = False
        # zhu = False
        # """---寻找客人(单人版)---"""
        # for i in range(6):
        #     self.navigator.goto("findpeople"+str(i))
        #     """寻找人,判断是否为客人"""
        #     print("正在检查是否有人")
        #     self.camera.open_camera()
        #     has_person, coords = self.people_detector.detect_person(self.camera, max_distance=3.0)
        #     self.camera.release()
        #     print("正在检测人脸")
        #     if has_person is not False:
        #         faceresult = self.face.detect_known_faces()
        #         self.face.close_k4a()
        #         if has_person is not False and faceresult == 0:
        #             ke = True
        #             person_map = self.transpoint.get_map_coords(coords)
        #             testgoal = self.goalpoint.find_best_goal(person_map)
        #             LOCATION["person_kr"] = testgoal
        #             self.navigator.goto("person_kr")
        #             self.speak.speak("客人你好")
        #             time.sleep(1)
        #             self.speak.speak("你要什么")
        #             time.sleep(2)
        #             # 中文识别
        #             while True:
        #                 text0, audio_file = record_and_recognize('zh', duration=5)
        #                 print(audio_file)
        #                 print(f"识别结果: {text0}")
        #                 #提取关键词 原识别结果为text0 关键词为text
        #                 try:
        #                     text, score = self.matcher.find_best_match(text0, min_confidence=0.85)
        #                     print(f"匹配成功: '{text}' (相似度: {score:.2%})\n")
        #                 except:
        #                     pass
        #                 time.sleep(1)
        #                 if text in cn2en:
        #                     self.speak.speak(f"好的{text}")
        #                     break
        #                 else:
        #                     self.speak.speak("请再说一遍")
        #                     time.sleep(1)
        #             if ke == True and zhu == True:
        #                 break
        #         elif has_person is not False and faceresult == 1:
        #             zhu = True
        #             zrposition = self.transpoint.get_map_coords(coords)
        #             ftpgoal = self.ftp.face_to_person(zrposition)
        #             LOCATION["person_zr"] = ftpgoal
        #             self.navigator.goto("person_zr")
        #             self.speak.speak("主人你好")
        #             time.sleep(2)
        #             if ke == True and zhu == True:
        #                 break

        """---寻找和抓取物品---"""
        self.camera.open_camera()
        for index in range(3):
            self.navigator.goto('finditem'+str(index))
            self.kinova.close_finger()
            """检测物品"""
            object_classes = self.items_detector.get_object_classes_sorted(self.camera, range=0.8)
            # if object_classes:
            #     self.speak.speak(f"找到{text}")
            if cn2en[text] in object_classes:
                self.speak.speak(f"找到{text}")
                has_target, coords = self.items_detector.detect(self.camera, target=cn2en[text], max_distance=5.0)
                print(f"是否检测到目标: {has_target}")
                if has_target:
                    self.speak.speak(f"抓取{text}")
                    print(f"三维坐标: x={coords[0]:.2f}m, y={coords[1]:.2f}m, z={coords[2]:.2f}m")
                    item_map = self.grip_object_coor(coords)
                    LOCATION["item_zx"] = item_map
                    self.navigator.goto("item_zx")
                    if index < 2:
                        self.kinova.catch_table(target=[fxz_en[text]])
                    else:
                        self.kinova.catch_table_short(target=[fxz_en[text]])
                    # self.kinova.catch_table(target=["bottle"])
                output=self.execute_command("rosrun kinova_demo fingers_action_client.py -v -r j2n6s300 percent -- 0 0 0")
                if int(output.split('\n')[4][36:40])>=6750:#6800完全闭合
                    self.speak.speak("帮帮我抓物品")
                    self.kinova.finger_run(finger_target=[5,5,5])
                    time.sleep(5)
                    self.kinova.finger_run(finger_target=[85,85,85])
                    self.kinova.finger_run(finger_target=[95,95,95])
                break
            # print("-------检测到的物品--------")
            # print(object_classes)
            # print("--------------------------")
            # if cn2en[text] in object_classes:
            #     i = 0
            #     for item in object_classes:
            #         # self.speak.speak(str(key for key, val in cn2en.items() if val == item))
            #         # time.sleep(1)
            #         if cn2en[text] == item and i < 3:
            #             self.speak.speak(f"找到{text}")
            #             self.navigator.goto('catchitem'+str(index)+str(i))
            #             if index < 2:
            #                 self.kinova.catch_table(target=[cn2en[text]])
            #             else:
            #                 self.kinova.catch_table_short(target=[cn2en[text]])
            #             # self.kinova.catch_table(target=["bottle"])
            #             output=self.execute_command("rosrun kinova_demo fingers_action_client.py -v -r j2n6s300 percent -- 0 0 0")
            #             if int(output.split('\n')[4][36:40])>=6750:#6800完全闭合
            #                 self.speak.speak("帮帮我")
            #                 self.kinova.help()
            #             break
            #         else :
            #             i += 1
            #     time.sleep(1)
            #     break
        self.camera.release()
        self.kinova.close_finger()

        """---拿回给客人(双人版)---"""
        for i in range(1,3):
            self.navigator.goto("findpeople"+str(i))
            """寻找人,判断是否为客人"""
            print("检测人脸")
            faceresult = self.face.detect_known_faces()
            self.face.close_k4a()
            list = [0,0,0]
            if  (faceresult[0] == list and faceresult[1] != list) or (faceresult[0] != list and faceresult[1] != list):
                person_map = self.transpoint.get_map_coords(faceresult[1])
                testgoal = self.goalpoint.find_best_goal(person_map)
                LOCATION["person_zx"] = testgoal
                self.navigator.goto("person_zx")
                self.speak.speak(f"给你{text}")
                time.sleep(1)
                # 松开爪子
                self.kinova.open_finger()
                break
        self.kinova.close_finger()
            
        
        # """---拿回给客人(单人版)---"""
        # for i in range(2,5):
        #     self.navigator.goto("findpeople"+str(i))
        #     """寻找人,判断是否为客人"""
        #     print("是否有人")
        #     self.camera.open_camera()
        #     has_person, coords = self.people_detector.detect_person(self.camera, max_distance=3.0)
        #     self.camera.release()
        #     print("检测人脸")
        #     if has_person is not False:
        #         faceresult = self.face.detect_known_faces()
        #         self.face.close_k4a()
        #         if has_person is not False and faceresult == 0:
        #             person_map = self.transpoint.get_map_coords(coords)
        #             testgoal = self.goalpoint.find_best_goal(person_map)
        #             LOCATION["person_zx"] = testgoal
        #             self.navigator.goto("person_zx")
        #             self.speak.speak(f"给你{text}")
        #             time.sleep(1)
        #             # 松开爪子
        #             self.kinova.open_finger()
        #             break
        
        """---捡垃圾---"""
        self.camera.open_camera()
        for index in range(2):
            self.navigator.goto('rubbish'+str(index))
            self.kinova.close_finger()
            if index < 1:
                """realsense检测"""
                self.kinova.goto_detect()
                result = self.detector.detect_targets(target_items = [cn2en[text]])
                # result = self.detector.detect_targets(target_items = ["bottle"])
                if result:
                    if text is not None:
                        self.speak.speak(f"找到{text}")
                    else:
                        self.speak.speak(f"找到{en2ch[result]}")
                    self.kinova.catch_ground(result = result)

                    output=self.execute_command("rosrun kinova_demo fingers_action_client.py -v -r j2n6s300 percent -- 0 0 0")
                    if  int(output.split('\n')[4][36:40])>=6750:#6800完全闭合
                        self.speak.speak("帮帮我抓物品")
                        self.kinova.finger_run(finger_target=[5,5,5])
                        time.sleep(5)
                        self.kinova.finger_run(finger_target=[85,85,85])
                        self.kinova.finger_run(finger_target=[95,95,95])

                    self.navigator.goto('can')
                    self.kinova.open_finger()
                    break
                else:
                    self.kinova.goto_home()
            else:
                """kinect检测"""
                # object_classes = self.items_detector.get_object_classes_sorted(self.camera, range=0.8)
                # if object_classes:
                #     if text is not None:
                #         self.speak.speak(f"找到{text}")
                #     else:
                #         self.speak.speak(f"找到{en2ch[object_classes[0]]}")
                #     self.navigator.goto("catchrubbish")
                #     self.kinova.catch_table(target=[object_classes[0]])
                self.speak.speak(f"找到{text}")
                has_target, coords = self.items_detector.detect(self.camera, target=cn2en[text], max_distance=5.0)
                print(f"是否检测到目标: {has_target}")
                if has_target:
                    print(f"三维坐标: x={coords[0]:.2f}m, y={coords[1]:.2f}m, z={coords[2]:.2f}m")
                    item_map = self.grip_object_coor(coords)
                    LOCATION["item_zx"] = item_map
                    self.navigator.goto("item_zx")
                    self.kinova.catch_table(target=[fxz_en[text]])

                output=self.execute_command("rosrun kinova_demo fingers_action_client.py -v -r j2n6s300 percent -- 0 0 0")
                if int(output.split('\n')[4][36:40])>=6750:#6800完全闭合
                    self.speak.speak("帮帮我抓物品")
                    self.kinova.finger_run(finger_target=[5,5,5])
                    time.sleep(5)
                    self.kinova.finger_run(finger_target=[85,85,85])
                    self.kinova.finger_run(finger_target=[95,95,95])

                self.navigator.goto('can')
                self.kinova.open_finger()
        self.camera.release()
        self.kinova.close_finger()
    
        self.navigator.goto("over")
            
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, required=False, default = 1)
    parser.add_argument('--d', type=int, required=False, default = 1)
    opt = parser.parse_args()
    try:
        Controller('reception', opt.r, opt.d)  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题，直到节点已经关闭
    except rospy.ROSInterruptException:
        pass
