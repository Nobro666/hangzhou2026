#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""2026 home robot: navigation and collection program."""


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


LOCATION = {
    "entry": [
        [1.4442159164730417,0.12393709459396401,0.138],[0.0,0.0,0.15602370067869278,0.9877533117264278]
    ],
    # The requested point is the original start location.
    "point": [
        [4.6003966923517154,1.0459468574130684,0.13800000000000004],[0.0,0.0,0.16525682855011448,0.986250566852845]
    ],
    "ground": [
        [2.733324741708853,0.7250536048895461,0.138],[0.0,0.0,-0.005270137648095871,0.9999861127281569]
    ],
    "table": [
        [5.599555222316705,-1.3830625515214168,0.138],[0.0,0.0,0.12488564807055481,0.9921711419437664]
    ],
    "shelf": [
        [0.13158331728377007,-0.006716505400027814,0.138],[0.0,0.0,0.0010637781280346942,0.9999994341878871]
    ],
    "over": [
        [4.068038268894237,-4.397159735867198,0.138],[0.0,0.0,-0.7343875552222743,0.6787303726330884]
    ]
}


class Controller:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.navigator = Navigator(LOCATION)
        self.kinova = KinovaRobot("j2n6s300")
        self.publish_initial_pose()
        self.control()

    def publish_initial_pose(self):
        publisher = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=10
        )
        rospy.sleep(1.0)

        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "map"
        initial_pose.pose.pose.position.x = 0.1686358744614841
        initial_pose.pose.pose.position.y = -0.0372035539253579
        initial_pose.pose.pose.position.z = 0.0
        initial_pose.pose.pose.orientation.x = 0.0
        initial_pose.pose.pose.orientation.y = 0.0
        initial_pose.pose.pose.orientation.z = 0.06810539735143235
        initial_pose.pose.pose.orientation.w = 0.9976781318900417
        initial_pose.pose.covariance = [
            0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0685,
        ]

        publisher.publish(initial_pose)
        rospy.sleep(1.0)
        publisher.publish(initial_pose)

    def control(self):
        if not self.navigator.goto("entry"):
            rospy.logerr("Failed to reach entry; stopping the route.")
            self.navigator.stop()
            return

        if not self.navigator.goto("ground"):
            rospy.logerr("Failed to reach ground; stopping the route.")
            self.navigator.stop()
            return
        self.kinova.catch_ground()
        if not self.navigator.goto("point"):
            rospy.logerr("Failed to return to point; stopping the route.")
            self.navigator.stop()
            return
        put_result = KinovaRobotGroud.put_rubbish(self.kinova)

        if not self.navigator.goto("table"):
            rospy.logerr("Failed to reach table; stopping the route.")
            self.navigator.stop()
            return
        self.kinova.catch_table()
        if not self.navigator.goto("point"):
            rospy.logerr("Failed to return to point; stopping the route.")
            self.navigator.stop()
            return

        if not self.navigator.goto("shelf"):
            rospy.logerr("Failed to reach shelf; stopping the route.")
            self.navigator.stop()
            return
        
        self.kinova.catch_ground()
        if not self.navigator.goto("point"):
            rospy.logerr("Failed to return to point; stopping the route.")
            self.navigator.stop()
            return

        if not self.navigator.goto("over"):
            rospy.logerr("Failed to reach over; stopping the route.")

        self.navigator.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--r", type=int, required=False, default=1)
    parser.add_argument("--d", type=int, required=False, default=1)
    parser.parse_args()

    try:
        Controller("jujia26")
    except rospy.ROSInterruptException:
        pass
