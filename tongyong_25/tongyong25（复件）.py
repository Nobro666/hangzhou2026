from  summer_tts_speaker import SummerTTSSpeaker
from get_keyword import FuzzyKeywordMatcher
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
from catch_ground.src.detector_items import ItemsDetector
from camera_to_map import CoordinateConverter
from position_last_second import SmartGoalFinder
# from face_detect import Detector #单人版
from face_detect_tongyong import Detector #双人版

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

# 储存导航路径点
LOCATION = {  
    "start":[[2.4840319119075254,-2.238934462417788,0.138],[0.0,0.0,0.7074893517204788,0.7067240035559404]],
    "findpeople0":[[2.8195689000433033,-2.2413569233334294,0.138],[0.0,0.0,0.8053245892457833,0.5928341302220295]],
    "findpeople1":[[2.73942594175554,-2.310976992660554,0.138],[0.0,0.0,0.9883815452998383,0.15199316073660543]],
    "findpeople2":[[3.5480701906667624,-2.559736959690771,0.138],[0.0,0.0,-0.22544511927386016,0.9742558689561972]],
    "findpeople3":[[3.545855374399566,-2.6734852021311695,0.138],[0.0,0.0,-0.6114889329377067,0.7912529841300473]],
    "finditem0":[[4.800237725382473,-0.6456325620279505,0.138],[0.0,0.0,0.6926466542607478,0.7212770704392261]],
    "finditem1":[[4.800237725382473,-0.6456325620279505,0.138],[0.0,0.0,0.6926466542607478,0.7212770704392261]],    
    "finditem2":[[4.800237725382473,-0.6456325620279505,0.138],[0.0,0.0,0.6926466542607478,0.7212770704392261]],
    "catchitem00":[[4.61050469868861,0.013664033131819991,0.138],[0.0,0.0,0.6982498145991385,0.7158541725884321]],
    "catchitem01":[[4.850678104584338,0.03057323015669111,0.138],[0.0,0.0,0.7052158283933219,0.7089926906418154]],    
    "catchitem02":[[5.09118055986546,0.060244865623190036,0.138],[0.0,0.0,0.7036682231213296,0.7105286987654128]],
    "catchitem10":[[4.562054625906155,-0.3918555385429725,0.138],[0.0,0.0,0.6900394658059088,0.7237717427685997]],
    "catchitem11":[[4.8290403209194706,-0.4360718042057745,0.138],[0.0,0.0,0.6852889872531239,0.7282712433905295]],    
    "catchitem12":[[5.177604791264276,-0.4015559482173361,0.138],[0.0,0.0,0.71476714571835,0.6993625150103794]],
    "catchitem20":[[4.562054625906155,-0.3918555385429725,0.138],[0.0,0.0,0.6900394658059088,0.7237717427685997]],
    "catchitem21":[[4.8290403209194706,-0.4360718042057745,0.138],[0.0,0.0,0.6852889872531239,0.7282712433905295]],    
    "catchitem22":[[5.177604791264276,-0.4015559482173361,0.138],[0.0,0.0,0.71476714571835,0.6993625150103794]],
    "rubbish0":[[5.039155629852422,-4.375484376452967,0.13799999999999998],[0.0,0.0,-0.6856638768866832,0.7279182975669202]],
    "rubbish1":[[4.807819601853032,-4.7985607108998405,0.13799999999999998],[0.0,0.0,-0.6941653063003329,0.7198156205091446]],
    "rubbish2":[[4.807819601853032,-4.7985607108998405,0.13799999999999998],[0.0,0.0,-0.6941653063003329,0.7198156205091446]],
    "rubbish3":[[4.807819601853032,-4.7985607108998405,0.13799999999999998],[0.0,0.0,-0.6941653063003329,0.7198156205091446]],
    "can":[[5.198877394552888,-2.6328117965142677,0.13799999999999998],[0.0,0.0,0.7292843682594019,0.6842107206208368]],
    "over":[[5.502173253708023,-4.702069637806176,0.13800000000000004],[0.0,0.0,0.056718688971219046,0.9983901994317583]]
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
        self.detector = RealSenseYolo11Detector(weights=Path("/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/model/best.pt"))
        self.camera = KinectCamera()
        self.people_detector = PersonDetector()
        self.items_detector = ItemsDetector()
        self.transpoint = CoordinateConverter()
        self.goalpoint = SmartGoalFinder()
        self.photo_path = '/home/zq/catkin_ws/src/cmoon/src/shijiazhuang_2025/tongyong_25/face'  # 替换为你想要保存照片的路径
        self.face = Detector(self.photo_path)
        self.speak = SummerTTSSpeaker()
        get_recognizer_instance('zh') # 这会触发模型加载 后续调用recognize函数时不会重复加载
        self.matcher = FuzzyKeywordMatcher(keywords=target_keywords)
        time.sleep(1)
        self.control()
        
    def control(self):
        text = None

        self.navigator.goto("start")
        time.sleep(1)

        """---寻找客人(两人版)---"""
        for i in range(4):
            self.navigator.goto("findpeople"+str(i))
            self.fdseat.xuanzhuan("findpeople"+str(i))
            """寻找人,判断是否为客人"""
            print("检测人脸")
            faceresult = self.face.detect_known_faces()
            self.face.close_k4a()
            list = [0,0,0]
            if  faceresult[0] == list and faceresult[1] != list:
                person_map = self.transpoint.get_map_coords(faceresult[1])
                testgoal = self.goalpoint.find_best_goal(person_map)
                LOCATION["person_zx"] = testgoal
                self.navigator.goto("person_zx")
                self.fdseat.xuanzhuan("person_zx")
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
                # break
            elif faceresult[0] != list and faceresult[1] == list:
                person_map = self.transpoint.get_map_coords(faceresult[0])
                testgoal = self.goalpoint.find_best_goal(person_map)
                LOCATION["person_zx"] = testgoal
                self.navigator.goto("person_zx")
                self.fdseat.xuanzhuan("person_zx")
                self.speak.speak("主人你好")
                time.sleep(1)
            elif faceresult[0] != list and faceresult[1] != list:
                person_map = self.transpoint.get_map_coords(faceresult[0])
                testgoal = self.goalpoint.find_best_goal(person_map)
                LOCATION["person_zx"] = testgoal
                self.navigator.goto("person_zx")
                self.fdseat.xuanzhuan("person_zx")
                self.speak.speak("主人你好")
                time.sleep(1)

                person_map = self.transpoint.get_map_coords(faceresult[1])
                testgoal = self.goalpoint.find_best_goal(person_map)
                LOCATION["person_zx"] = testgoal
                self.navigator.goto("person_zx")
                self.fdseat.xuanzhuan("person_zx")
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

        
        # """---寻找客人(单人版)---"""
        # for i in range(4):
        #     self.navigator.goto("findpeople"+str(i))
        #     self.fdseat.xuanzhuan("findpeople"+str(i))
        #     """寻找人,判断是否为客人"""
        #     print("是否有人")
        #     self.camera.open_camera()
        #     has_person, coords = self.people_detector.detect_person(self.camera, max_distance=3.0)
        #     self.camera.release()
        #     print("检测人脸")
        #     faceresult = self.face.detect_known_faces()
        #     self.face.close_k4a()
        #     if has_person is not False and faceresult == 0:
        #         person_map = self.transpoint.get_map_coords(coords)
        #         testgoal = self.goalpoint.find_best_goal(person_map)
        #         LOCATION["person_zx"] = testgoal
        #         self.navigator.goto("person_zx")
        #         self.fdseat.xuanzhuan("person_zx")
        #         self.speak.speak("客人你好")
        #         time.sleep(1)
        #         self.speak.speak("你要什么")
        #         time.sleep(2)
        #         # 中文识别
        #         while True:
        #             text0, audio_file = record_and_recognize('zh', duration=5)
        #             print(audio_file)
        #             print(f"识别结果: {text0}")
        #             #提取关键词 原识别结果为text0 关键词为text
        #             try:
        #                 text, score = self.matcher.find_best_match(text0, min_confidence=0.85)
        #                 print(f"匹配成功: '{text}' (相似度: {score:.2%})\n")
        #             except:
        #                 pass
        #             time.sleep(1)
        #             if text in cn2en:
        #                 self.speak.speak(f"好的{text}")
        #                 break
        #             else:
        #                 self.speak.speak("请再说一遍")
        #         # break
        #     elif has_person is not False and faceresult == 1:
        #         self.speak.speak("主人你好")


        """---寻找和抓取物品---"""
        self.camera.open_camera()
        for index in range(1):
            self.navigator.goto('finditem'+str(index))
            self.fdseat.xuanzhuan('finditem'+str(index))
            """检测物品"""
            object_classes = self.items_detector.get_object_classes_sorted(self.camera, range=0.8)
            print("-------检测到的物品--------")
            print(object_classes)
            print("--------------------------")
            if cn2en[text] in object_classes:
                i = 0
                for item in object_classes:
                    # self.speak.speak(str(key for key, val in cn2en.items() if val == item))
                    # time.sleep(1)
                    if cn2en[text] == item and i < 3:
                        self.speak.speak(f"找到{text}")
                        self.navigator.goto('catchitem'+str(index)+str(i))
                        self.fdseat.xuanzhuan('catchitem'+str(index)+str(i))
                        self.kinova.catch_table(target=[cn2en[text]])
                        # self.kinova.catch_table(target=["bottle"])
                        break
                    else :
                        i += 1
                time.sleep(1)
                break
        self.camera.release()

        """---拿回给客人(双人版)---"""
        for i in range(4):
            self.navigator.goto("findpeople"+str(i))
            self.fdseat.xuanzhuan("findpeople"+str(i))
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
                self.fdseat.xuanzhuan("person_zx")
                self.speak.speak(f"给你{text}")
                time.sleep(1)
                # 松开爪子
                self.kinova.open_finger()
                break
            
        
        # """---拿回给客人(单人版)---"""
        # for i in range(4):
        #     self.navigator.goto("findpeople"+str(i))
        #     self.fdseat.xuanzhuan("findpeople"+str(i))
        #     """寻找人,判断是否为客人"""
        #     print("是否有人")
        #     self.camera.open_camera()
        #     has_person, coords = self.people_detector.detect_person(self.camera, max_distance=3.0)
        #     self.camera.release()
        #     print("检测人脸")
        #     faceresult = self.face.detect_known_faces()
        #     self.face.close_k4a()
        #     if has_person is not False and faceresult == 0:
        #         person_map = self.transpoint.get_map_coords(coords)
        #         testgoal = self.goalpoint.find_best_goal(person_map)
        #         LOCATION["person_zx"] = testgoal
        #         self.navigator.goto("person_zx")
        #         self.fdseat.xuanzhuan("person_zx")
        #         self.speak.speak(f"给你{text}")
        #         time.sleep(1)
        #         # 松开爪子
        #         self.kinova.open_finger()
        #         break
        
        """---捡垃圾---"""
        self.camera.open_camera()
        for index in range(1):
            self.navigator.goto('rubbish'+str(index))
            self.fdseat.xuanzhuan('rubbish'+str(index))
            if index < 2:
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
                    self.navigator.goto('can')
                    self.fdseat.xuanzhuan('can')
                    self.kinova.open_finger()
                    break
            else:
                """kinect检测"""
                object_classes = self.items_detector.get_object_classes_sorted(self.camera, range=0.8)
                if cn2en[text] in object_classes:
                    self.kinova.catch_table(target=[cn2en[text]])
        self.camera.release()
    
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
