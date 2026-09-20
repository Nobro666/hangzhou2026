#!/usr/bin/env python3
# coding: UTF-8
# created by TG

import cv2
import rospy
import math
import time
import pykinect_azure as pykinect
from soundplayer import Soundplayer 
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from soundplayer import Soundplayer
from sensor_msgs.msg import LaserScan

follow_id = 1
goal_z = 0.88
start = time.time()
class Wit_recogenizer:
    def __init__(self):
        # rospy.init_node('wit_recogenizer', anonymous=True)  # 初始化ros节点
        self.pub1 = rospy.Publisher('/cmd_vel', Twist, queue_size=2)
        self.pub2 = rospy.Publisher('/xfwords', String, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scanmsg)
        pass

    
    def scanmsg(self,scans):
        self.scanss = scans




    def gesture(self):
        pre_bias_z=0
        pre_bias_x=0
        bias_x=0
        bias_z=0
        global follow_id,start
        global goal_z
        pykinect.initialize_libraries(track_body=True)
        device_config = pykinect.default_configuration
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        device = pykinect.start_device(config=device_config)
        bodyTracker = pykinect.start_body_tracker()
        start_follow_new = None
        while True:
			
            self.capture = device.update()
            body_frame = bodyTracker.update()
            body1 = body_frame.get_bodies()
            ret, colorfulframe = self.capture.get_color_image()
            flag_once = False
            bddd = None
            mnz = 1000000
            cmd = Twist()
            body=None
            for key in body1:
                xa = key.joints[2].position.x
                za = key.joints[2].position.z
                if(math.sqrt(za*za+xa*xa)<mnz):
                    mnz=math.sqrt(za*za+xa*xa)
                    body = key

            if(body is None): 
                cmd.angular.z=0.2
                cmd.linear.x=0.0
                self.pub1.publish(cmd)
                continue
            x = body.joints[2].position.x / 1000
            z = body.joints[2].position.z / 1000
            body_id = body_frame.get_body_id()

            if(body_id is not None):
                    start = time.time()
                    
                    # cmd.linear.x = 0.23 if scale_z*(z- goal_z)>0.20 else scale_z*(z- goal_z)
                    # cmd.linear.x*=1.5
                    # cmd.angular.z = -0.5*scale_x*(x+0.05) if(math.fabs(x+0.05)>0.2) else 0
                    if z>0.93 or z<0.88:
                       bias_z=z-0.9
                       cmd.linear.x=1.2*bias_z+0.7*(bias_z-pre_bias_z)
                       pre_bias_z=bias_z
                    else:
                       cmd.linear.x=0   
                    if x>0.12 or x<-0.1:   
                       bias_x=x-0.05
                       cmd.angular.z= -1.0*(x-0.05)-0.1*(bias_x-pre_bias_x)
                       pre_bias_x=bias_x
                    else:
                       cmd.angular.z=0
                       
                    if(cmd.linear.x>0):
                        for i in range(0,1080):
                            if(i>=420 and i<=735 and self.scanss.ranges[i]>0.02 and self.scanss.ranges[i]<=0.25):
                                cmd.linear.x = 0
                                break
                            if(i>=940 and i<=1000 and self.scanss.ranges[i]>0.02 and self.scanss.ranges[i]<=0.53*i/970):
                                cmd.linear.x = 0
                                break
                   
                    if (start_follow_new is None or abs(cmd.linear.x)>0.12 or abs(cmd.angular.z)>0.1):
                        start_follow_new = time.time()
                    start_follow_newxxx = time.time()
                    if((int(start_follow_newxxx)-int(start_follow_new))>6):
                        flag_once = True
                        break
                    if(z>1.5):
                        self.pub2.publish('please wait for me')
                    self.pub1.publish(cmd)

            if(flag_once):
                cv2.destroyAllWindows()
                device.stop_cameras()
                device.close()
                break
     

        