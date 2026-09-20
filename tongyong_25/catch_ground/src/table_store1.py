"""
  @File: kinovarobot.py
  @Brief: Control program of j6n2s300, Kinova Robot, percent command to test gripper goals, cartesian position to test cartesian goals for arm
 
  @Author: Benxiaogu
  @Github: https://github.com/Benxiaogu
  @CSDN: https://blog.csdn.net/weixin_51995147?type=blog
 
  @Date: 2024-12-19
"""


import roslib; roslib.load_manifest('kinova_demo')
import rospy
import numpy as np
import math
import time
from geometry_msgs.msg import Twist
from detector2025 import KinectCamera, Detector
from find_seat import Follower
from navigator import Navigator
from soundplayer import Soundplayer  # 语音合成模块
from datetime import datetime
# from ultralytics import YOLO
from kinova_msgs.msg import FingerPosition

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import subprocess
things={
    "food":["Biscuit","Chip","Lays","Cookie","Cereal","Bread"],
    "cleaning stuff":["Handwash","Dishsoap","Shampoo","Cereal bowl","Spoon"],
    "drink":["Sprite","Cola","Orange juice","Milk"],
    "heavy":["Water"]
}

#有几个物体就写几个物体，例如有两个瓶子就写两个bottle
names=["Biscuit","Chip","Lays","Cookie","Cereal","Bread","Water","Handwash",
       "Dishsoap","Shampoo","Cereal bowl","Spoon","Sprite","Cola",
       "Orange juice","Milk"]

location = {
        '6': [
            [0.15331095328124028, 0.294858255536295, 0.138], [0.0, 0.0, -0.007873026585157296
, 0.9999690072459193 ]],#杂物架靠左
        '7': [[0.13876079115715023, 0.011479344748880238, 0.138], [0.0, 0.0,  -0.03461508000093895
, 0.9994007185491356 ]],#杂物架中间
        '8': [[0.12453099993728695, -0.21308350943641796, 0.138], [0.0, 0.0,  -0.0023648683560521244
, 0.9999972036949196 ]],#杂物架靠右
        "1": [[7.211056482448738, 8.059370689912413, 0.138],
              [0.0, 0.0,0.004199894818354422,0.9999911804028647]],#绿1最左边
        "2": [[7.211056482448738,8.401515146524269,0.138],[0,0,0.9999645572003524,0.008419284001816714]],
        "3": [[7.211056482448738,8.706250371252767,0.138],[0,0,0.010496951770350095,0.9999449054840627]],
        '4': [[-2.2551562584750595, 2.992455057965569, 0.138], [0.0, 0.0,  0.6519100839350997
, 0.7582962761768853 ]],#客厅最右边座位
        '5': [[-2.094060433351585, 4.194709920185249, 0.138], [0.0, 0.0,  0.9920446886687214
, 0.12588620132556122 ]],#客厅右后边座位
        '9': [[-2.3539173677092, -0.072568157736902, 0.138], [0.0, 0.0,  0.9990553325146179
, -0.04345621444749128 ]],#餐厅
    }
class thing:
    def __init__(self,name,floor,category):
        self.name = name
        self.floor = floor
        self.category = category
        pass

class KinovaRobot:
    """
        kinova 机械臂控制， 此程序中采用笛卡尔坐标进行控制机械臂
    """
    def __init__(self,kinova_robotType) -> None:
        rospy.init_node(kinova_robotType)
        rospy.loginfo("init_node successfully")
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        self.kinectcamera = KinectCamera()   # kinect相机对象初始化
        self.kinectcamera.open_camera()  # 打开kinect相机
        self.detector = Detector()   # 检测器对象初始化
        self.kinova_robotType = kinova_robotType
        # self.kinova_robotType = 'j2n6s300'
        self.prefix = self.kinova_robotType + "_"
        robot_category = kinova_robotType[0]
        robot_category_version = int(kinova_robotType[1])
        wrist_type = kinova_robotType[2]
        self.arm_joint_number = int(kinova_robotType[3])
        self.finger_number = int(kinova_robotType[5])
        self.finger_maxDist = 18.9/2/1000
        self.finger_maxTurn = 6800
        self.currentFingerPosition = [0.0, 0.0, 0.0]
        # self.currentCartesianCommand = [0.21258243918418884, -0.25638914108276367, 0.50766521692276, 1.648742437362671, 1.1138312816619873, 0.50766521692276] # default home in unit mq
        self.currentCartesianCommand = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 1.6373136043548584, 1.1021580696105957, 0.5095799565315247]
        # self.homePositionMdeg = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
        self.init_pose = [0.36157482862472534, 0.10528099536895752, 0.5518933534622192, 81.040, 83.972, 11.606]
        self.getcurrentCartesianCommand()
        self.action_address_arm = '/' + self.prefix + 'driver/pose_action/tool_pose'
        self.client_arm = actionlib.SimpleActionClient(self.action_address_arm, kinova_msgs.msg.ArmPoseAction)
        self.client_arm.wait_for_server()
        self.goal_arm = kinova_msgs.msg.ArmPoseGoal()
        self.goal_arm.pose.header = std_msgs.msg.Header(frame_id=(self.prefix + 'link_base'))
        rospy.loginfo("arm service connect successfully")
        # self.arm_run(pose_target=self.homePositionMdeg)

        rospy.Subscriber('/j2n6s300_driver/out/finger_position', FingerPosition, self.finger_pose)

        self.getCurrentFingerPosition()
        self.action_address_finger = '/' + self.prefix + 'driver/fingers_action/finger_positions'
        self.client_finger = actionlib.SimpleActionClient(self.action_address_finger, kinova_msgs.msg.SetFingersPositionAction)
        self.client_finger.wait_for_server()
        self.goal_finger = kinova_msgs.msg.SetFingersPositionGoal()
        rospy.loginfo("finger service connect successfully")
        self.finger_run(finger_target=[95,95,95])
        self.floor = None
        self.thisnum = 0
        self.zhonglei = None
        # self.observe = [0.3975765824317932, -0.09243141114711761, 0.223219232559204, 57.052, 89.704, 36.111]
        # self.observe_up = [0.3975765824317932, -0.09243141114711761, 0.443219232559204, 57.052, 89.704, 36.111]

        # self.arm_run(pose_target=self.init_pose)
        # self.arm_run(pose_target=self.observe)


# translation: 

        self.kinectA2kinova_matrix = np.array([[ 0.0175, -0.4997, 0.8660, -0.11490607],
                                              [ -0.9994, -0.0349, 0, -0.00222445],
                                              [ 0.0302, -0.8654, -0.5 , 1.07570852],
                                              [ 0. , 0. , 0. , 1. ]])
        

        # self.kinectB2kinova_matrix = np.array([[0]])

        self.realsense2kinova_matrix = np.array([
                                                [-0.81552114, -0.02233797, -0.57829602,  1.37482036],
                                                [ 0.06292209, -0.99675352, -0.05023179, -0.04230057],
                                                [-0.57529651, -0.07735268,  0.81427912,  0.84682352],
                                                [ 0.        ,  0.        ,  0.        ,  1.        ]
                                                ])


    def quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """Convert quaternion to a 3x3 rotation matrix."""
        r00 = 1 - 2 * (qy**2 + qz**2)
        r01 = 2 * (qx * qy - qz * qw)
        r02 = 2 * (qx * qz + qy * qw)
        r10 = 2 * (qx * qy + qz * qw)
        r11 = 1 - 2 * (qx**2 + qz**2)
        r12 = 2 * (qy * qz - qx * qw)
        r20 = 2 * (qx * qz - qy * qw)
        r21 = 2 * (qy * qz + qx * qw)
        r22 = 1 - 2 * (qx**2 + qy**2)
        return np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])



    def arm_run(self,unit='mdeg',pose_target=None,relative=False):
        pose_mq, pose_mdeg, pose_mrad = self.unitParser_arm(unit,pose_target,relative)
        try:
            self.poses = [float(n) for n in pose_mq]
            self.cartesian_pose_client(self.poses[:3], self.poses[3:])
            print('Cartesian pose sent!')
        except rospy.ROSInterruptException:
            print("program interrupted before completion")

    def finger_run(self,unit='percent',finger_target=None,relative=False):
        finger_turn, finger_meter, finger_percent = self.unitParser_finger(unit, finger_target, relative)
        try:
            if self.finger_number == 0:
                print('Finger number is 0, check with "-h" to see how to use this node.')
                self.positions = []  # Get rid of static analysis warning that doesn't see the exit()
                exit()
            else:
                positions_temp1 = [max(0.0, n) for n in finger_turn]
                positions_temp2 = [min(n, self.finger_maxTurn) for n in positions_temp1]
                self.positions = [float(n) for n in positions_temp2]

            print('Sending finger position ...')
            
            result = self.gripper_client(self.positions)

            # print(f"finger: {self.finger_position}")


            # 判断夹爪闭合角度，这里设为20，具体你们再调
            i = 0
            while i<3:
                if self.finger_position[0]>400 and self.finger_position[1]>400:
                    print('----------------- Finger position sent -----------------')
                    break
                else:
                    result = self.gripper_client(self.positions)
                i+=1

        except rospy.ROSInterruptException:
            print('program interrupted before completion')

    def cartesian_pose_client(self, position, orientation):
        """Send a cartesian goal to the action server."""
        self.goal_arm.pose.pose.position = geometry_msgs.msg.Point(
            x=position[0], y=position[1], z=position[2])
        self.goal_arm.pose.pose.orientation = geometry_msgs.msg.Quaternion(
            x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])

        self.client_arm.send_goal(self.goal_arm)

        if self.client_arm.wait_for_result(rospy.Duration(10.0)):
            return self.client_arm.get_result()
        else:
            self.client_arm.cancel_all_goals()
            print('the cartesian action timed-out')
            return None
        
    def gripper_client(self, finger_positions):
        """Send a gripper goal to the action server."""
        self.goal_finger.fingers.finger1 = float(finger_positions[0])
        self.goal_finger.fingers.finger2 = float(finger_positions[1])
        # The MICO arm has only two fingers, but the same action definition is used
        if len(finger_positions) < 3:
            self.goal_finger.fingers.finger3 = 0.0
        else:
            self.goal_finger.fingers.finger3 = float(finger_positions[2])
        self.client_finger.send_goal(self.goal_finger)

        # 如果等待结果超时，会取消所有未完成的目标
        if self.client_finger.wait_for_result(rospy.Duration(8.0)):
            return self.client_finger.get_result()
        else:
            self.client_finger.cancel_all_goals()
            rospy.logwarn('the gripper action timed-out')
            return None

    # 手指角度订阅者的回调函数，可以通过访问self.finger_position来实时查看三个手指的闭合角度（严谨来说不是角度，可能是转数，你们理解就好），目前来看最小为6，最大为7200+，你们可以运行此程序，然后通过手柄控制夹爪看输出结果
    def finger_pose(self, finger_percent):
        self.finger_position = [finger_percent.finger1, finger_percent.finger2, finger_percent.finger3]
        # 实时输出手指角度，仅用于调试，调试好之后，不需要输出了，因为这个是实时的，会一直输出
        print(f"finger1: {self.finger_position[0]}, finger2: {self.finger_position[1]}, finger3:{self.finger_position[2]}")
    #######################################################
        
    def unitParser_arm(self, unit_, pose_value_, relative_):
        """ Argument unit """
        position_ = pose_value_[:3]
        orientation_ = pose_value_[3:]
        print(f"position_:{position_}, orientation_:{orientation_}")

        for i in range(0,3):
            if relative_:
                position_[i] = pose_value_[i] + self.currentCartesianCommand[i]
            else:
                position_[i] = pose_value_[i]

        # print('pose_value_ in unitParser 1: {}'.format(pose_value_))  # debug

        if unit_ == 'mq':
            # 四元数
            if relative_:
                orientation_XYZ = self.Quaternion2EulerXYZ(orientation_)
                orientation_xyz_list = [orientation_XYZ[i] + self.currentCartesianCommand[3+i] for i in range(0,3)]
                orientation_q = self.EulerXYZ2Quaternion(orientation_xyz_list)
            else:
                orientation_q = orientation_

            orientation_rad = self.Quaternion2EulerXYZ(orientation_q)
            orientation_deg = list(map(math.degrees, orientation_rad))

        elif unit_ == 'mdeg':
            # 角度欧拉角
            if relative_:
                orientation_deg_list = list(map(math.degrees, self.currentCartesianCommand[3:]))
                orientation_deg = [orientation_[i] + orientation_deg_list[i] for i in range(0,3)]
            else:
                orientation_deg = orientation_

            orientation_rad = list(map(math.radians, orientation_deg))
            orientation_q = self.EulerXYZ2Quaternion(orientation_rad)

        elif unit_ == 'mrad':
            # 弧度欧拉角
            if relative_:
                orientation_rad_list =  self.currentCartesianCommand[3:]
                orientation_rad = [orientation_[i] + orientation_rad_list[i] for i in range(0,3)]
            else:
                orientation_rad = orientation_

            orientation_deg = list(map(math.degrees, orientation_rad))
            orientation_q = self.EulerXYZ2Quaternion(orientation_rad)

        else:
            raise Exception("Cartesian value have to be in unit: mq, mdeg or mrad")

        pose_mq_ = position_ + orientation_q
        pose_mdeg_ = position_ + orientation_deg
        pose_mrad_ = position_ + orientation_rad

        # print('pose_mq in unitParser 1: {}'.format(pose_mq_))  # debug

        return pose_mq_, pose_mdeg_, pose_mrad_
    
    def Quaternion2Eulerransform_zyx(self,w,x,y,z):
        """
            四元素欧->欧拉角
        """
        angle_is_not_rad = True

        r = math.atan2(2 * (w * x-y*z), 1-2*y*y-2*x*x)
        p = math.asin(2 * (w * y + z * x))
        y = math.atan2(2 * (w * z - y * x), 1-2*(y*y+z*z))
        
        
        #将结果转换为角度
        if angle_is_not_rad : # pi -> 180 
            #将数值转换为角度
            r = math.degrees(r)
            p = math.degrees(p)
            y = math.degrees(y)
        return [r,p,y]
        
    def QuaternionNorm(self, Q_raw):
        qx_temp,qy_temp,qz_temp,qw_temp = Q_raw[0:4]
        qnorm = math.sqrt(qx_temp*qx_temp + qy_temp*qy_temp + qz_temp*qz_temp + qw_temp*qw_temp)
        qx_ = qx_temp/qnorm
        qy_ = qy_temp/qnorm
        qz_ = qz_temp/qnorm
        qw_ = qw_temp/qnorm
        Q_normed_ = [qx_, qy_, qz_, qw_]
        return Q_normed_


    def Quaternion2EulerXYZ(self, Q_raw):
        Q_normed = self.QuaternionNorm(Q_raw)
        qx_ = Q_normed[0]
        qy_ = Q_normed[1]
        qz_ = Q_normed[2]
        qw_ = Q_normed[3]

        tx_ = math.atan2((2 * qw_ * qx_ - 2 * qy_ * qz_), (qw_ * qw_ - qx_ * qx_ - qy_ * qy_ + qz_ * qz_))
        ty_ = math.asin(2 * qw_ * qy_ + 2 * qx_ * qz_)
        tz_ = math.atan2((2 * qw_ * qz_ - 2 * qx_ * qy_), (qw_ * qw_ + qx_ * qx_ - qy_ * qy_ - qz_ * qz_))
        EulerXYZ_ = [tx_,ty_,tz_]
        return EulerXYZ_


    def EulerXYZ2Quaternion(self, EulerXYZ_):
        print("EulerXYZ_:",EulerXYZ_)
        tx_, ty_, tz_ = EulerXYZ_[0:3]
        sx = math.sin(0.5 * tx_)
        cx = math.cos(0.5 * tx_)
        sy = math.sin(0.5 * ty_)
        cy = math.cos(0.5 * ty_)
        sz = math.sin(0.5 * tz_)
        cz = math.cos(0.5 * tz_)

        qx_ = sx * cy * cz + cx * sy * sz
        qy_ = -sx * cy * sz + cx * sy * cz
        qz_ = sx * sy * cz + cx * cy * sz
        qw_ = -sx * sy * sz + cx * cy * cz

        Q_ = [qx_, qy_, qz_, qw_]
        return Q_
    


    def unitParser_finger(self, unit_, finger_value_, relative_):
        """ Argument unit """
        # 根据用户指定的单位将目标值转换为内部表示
        # transform between units
        if unit_ == 'turn':
            # get absolute value
            if relative_:
                finger_turn_absolute_ = [finger_value_[i] + self.currentFingerPosition[i] for i in range(0, len(finger_value_))]
            else:
                finger_turn_absolute_ = finger_value_

            finger_turn_ = finger_turn_absolute_
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]

        elif unit_ == 'mm':
            # get absolute value
            finger_turn_command = [x/1000 * self.finger_maxTurn / self.finger_maxDist for x in finger_value_]
            if relative_:
                finger_turn_absolute_ = [finger_turn_command[i] + self.currentFingerPosition[i] for i in range(0, len(finger_value_))]
            else:
                finger_turn_absolute_ = finger_turn_command

            finger_turn_ = finger_turn_absolute_
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]
        elif unit_ == 'percent':
            # get absolute value
            finger_turn_command = [x/100.0 * self.finger_maxTurn for x in finger_value_]
            if relative_:
                finger_turn_absolute_ = [finger_turn_command[i] + self.currentFingerPosition[i] for i in
                                        range(0, len(finger_value_))]
            else:
                finger_turn_absolute_ = finger_turn_command

            finger_turn_ = finger_turn_absolute_
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]
        else:
            raise Exception("Finger value have to be in turn, mm or percent")

        return finger_turn_, finger_meter_, finger_percent_

    def getcurrentCartesianCommand(self):
        # wait to get current position
        topic_address = '/' + self.prefix + 'driver/out/cartesian_command'
        rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, self.setcurrentCartesianCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
        print('position listener obtained message for Cartesian pose. ')

    def setcurrentCartesianCommand(self, feedback):
        currentCartesianCommand_str_list = str(feedback).split("\n")

        for index in range(0,len(currentCartesianCommand_str_list)):
            temp_str=currentCartesianCommand_str_list[index].split(": ")
            self.currentCartesianCommand[index] = float(temp_str[1])
        # the following directly reading only read once and didn't update the value.
        # self.currentCartesianCommand = [feedback.X, feedback.Y, feedback.Z, feedback.ThetaX, feedback.ThetaY, feedback.Z] 
        # print('currentCartesianCommand in setcurrentCartesianCommand is: ', self.currentCartesianCommand)


    def getCurrentFingerPosition(self):
        # wait to get current position
        # 获取当前夹爪手指的位置
        topic_address = '/' + self.prefix + 'driver/out/finger_position'
        rospy.Subscriber(topic_address, kinova_msgs.msg.FingerPosition, self.setCurrentFingerPosition) # 将接收到的手指位置通过setCurrentFingerPosition存入全局变量
        rospy.wait_for_message(topic_address, kinova_msgs.msg.FingerPosition)
        print('obtained current finger position ')

    def setCurrentFingerPosition(self,feedback):
        self.currentFingerPosition[0] = feedback.finger1
        self.currentFingerPosition[1] = feedback.finger2
        self.currentFingerPosition[2] = feedback.finger3

    def transform(self, position):
        """
            将目标位置由相机坐标系转换到kinova机器人base坐标系
        """
        # translation = self.kinectA2kinova_matrix[:3, 3]
        # print("translation: ",translation)
        # rotation = self.kinectA2kinova_matrix[:3, :3]
        # print("rotation: ",rotation)
        # pos_camera = np.array([position[0],position[1],position[2]])
        # position_end_effector = np.dot(rotation,pos_camera) + translation
        # print("position_end_effector: ",position_end_effector)
        pos_camera = np.array([position[0],position[1],position[2],1.0])
        position_end_effector = self.kinectA2kinova_matrix.dot(pos_camera)

        position_end_effector = position_end_effector[:3]
        print("position_end_effector: ",position_end_effector)
        

        return position_end_effector
    
    def image_to_arm(self,camera_x,camera_y,camera_z):
        """
            眼在手上，相机坐标系转末端执行器坐标系
        """
        # qingdao 
        x= 0.020816677729270594
        y= 0.06788506740074149
        z= -0.10464690917725247
        translation = np.array([[x],[y],[z]])
        q0= 0.5112811368530545
        q1= -0.4841087597350034
        q2= -0.49895024240575236
        q3= -0.5052513863903056
# 
        R = np.array([[-0.00846062,0.99974337,-0.02101445],
                        [-0.03355864,0.02071949,0.99922196],
                        [ 0.99940094,0.00915925,0.03337473]])



        R_1 = np.array([[1-2*q2*q2-2*q3*q3, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],  
                            [2*q1*q2+2*q0*q3, 1-2*q1*q1-2*q3*q3, 2*q2*q3-2*q0*q1],  
                            [2*q1*q3-2*q0*q2, 2*q2*q2+2*q0*q1, 1-2*q1*q1-2*q2*q2]])
        # R = np.array([[4.56550802e-04 , 9.98906302e-01 , 4.67545876e-02 ], 
        #              [-1.74852234e-02 ,-4.67554189e-02 , 9.98753322e-01 ],
        #              [9.99847018e-01 ,-3.61532781e-04 , 1.74874461e-02 ]]) # 手眼标定转换矩阵
        P_camera = np.array([[camera_x], [camera_y], [camera_z]])
        P_robot = np.dot(R, P_camera) + translation 
        # print("R",R,"R_1",R_1)
        # print(P_robot)
        return P_robot
    

    def verboseParser(self, verbose=False):
        """ Argument verbose """
        position_ = self.poses[:3]
        orientation_q = self.poses[3:]
        if verbose:
            orientation_rad = self.Quaternion2EulerXYZ(orientation_q)
            orientation_deg = list(map(math.degrees, orientation_rad))
            print('Cartesian position is: {}'.format(position_))
            print('Cartesian orientation in Quaternion is: ')
            print('qx {:0.3f}, qy {:0.3f}, qz {:0.3f}, qw {:0.3f}'.format(orientation_q[0], orientation_q[1], orientation_q[2], orientation_q[3]))
            print('Cartesian orientation in Euler-XYZ(radian) is: ')
            print('tx {:0.3f}, ty {:0.3f}, tz {:0.3f}'.format(orientation_rad[0], orientation_rad[1], orientation_rad[2]))
            print('Cartesian orientation in Euler-XYZ(degree) is: ')
            print('tx {:3.1f}, ty {:3.1f}, tz {:3.1f}'.format(orientation_deg[0], orientation_deg[1], orientation_deg[2]))

            finger_turn_ = self.positions
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]
            print('Finger values in turn are: ')
            print(', '.join('finger{:1.0f} {:4.0f}'.format(k[0] + 1, k[1]) for k in enumerate(finger_turn_)))
            print('Finger values in mm are: ')
            print(', '.join('finger{:1.0f} {:2.1f}'.format(k[0]+1, k[1]*1000) for k in enumerate(finger_meter_)))
            print('Finger values in percentage are: ')
            print(', '.join('finger{:1.1f} {:3.1f}%'.format(k[0]+1, k[1]) for k in enumerate(finger_percent_)))
        return position_, orientation_deg

    def kinova_coordinate(self,xyz):
        #由相机坐标转换到机械臂坐标
        xyz_arm = kinova.transform(xyz)
        print("xyz_arm:",xyz_arm)
        if xyz_arm[2]<0.518:
            xyz_arm[2]=0.518
        x_move=0
        x_move_error=0
        catch_heigh=0
        paper_bag_height=0
        paper_x_move=0
        grip_orientation=[81.040, 83.972, 11.606]
        approach_height=0
        z_move = 0
        if self.name=="Water":
            approach_height=0.04
            catch_heigh=0
        elif self.name in ["Bread","Cereal"]:
            grip_orientation = self.Quaternion2Eulerransform_zyx(-0.0015861168503761292,0.9776652455329895,-0.026017673313617706,0.20854552090168)
            approach_height = 0.12
            catch_heigh=-0.05
            x_move = 0
            x_beforeMove = -0.15
            z_move=0
        else:
            catch_heigh=0
            grip_orientation=[81.040, 83.972, 11.606]
            approach_height=0
            z_move=0
        x_move=-0.15 if abs(xyz_arm[0]-0.2104809731245041)>0.20 else 0
        

        pose_target_1 = [xyz_arm[0]+x_move,
                         xyz_arm[1]-0.03,
                         xyz_arm[2]+approach_height]
        #真正抓到物体
        pose_target_2 = [xyz_arm[0],
                         xyz_arm[1]-0.03,
                         xyz_arm[2]+approach_height+catch_heigh-0.02]
        #抓到后上提
        pose_target_3 = [xyz_arm[0]+paper_x_move,
                         xyz_arm[1]-0.03,
                         xyz_arm[2]+0.10+approach_height+paper_bag_height]
        # tx 81.040, ty 83.972, tz 11.606
        #将旋转角加到坐标中
        print(pose_target_2)
        print(pose_target_3)
        pose_target_1.extend(grip_orientation)
        pose_target_2.extend(grip_orientation)
        pose_target_3.extend(grip_orientation)
        return pose_target_1,pose_target_2,pose_target_3
    
    def execute_command(self,command):
        result=subprocess.run(command, shell=True, capture_output=True, text=True)
        output=result.stdout.strip()
        return output     

    def move_back_x(self,move_dis):
        k = int(math.fabs(move_dis)/0.01+0.01)
        cmd = Twist()
        cmd.linear.x = 0.15 if(move_dis>0) else -0.15
        cmd.angular.z = 0
        for i in range(0,k):
            self.vel.publish(cmd)
            time.sleep(0.1)
        cmd.linear.x = 0
        self.vel.publish(cmd)

    def table_store(self):
        """
            names和things到现场需要根据实物更新
            thing更新实例
            假如现场有两个Water,一个Cola,三个Sprite
            则：   
            names=["Water","Water","Cola","Sprite","Sprite","Sprite"]
                    Water,Cola,Sprite的顺序无所谓,但数量要严格相同
        """
        # names
        # store_first_position
        # store_second_position

        goods_length=len(names)
        # flag_first=0
        # flag_second=0


        follower=Follower(location)
        navigator=Navigator(location)
        home_position= [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
        #抓取后首先回去的位子，防止碰撞
        home_position_1= [0.39078488945961+0.1, -0.14549310505390167, 0.5201553106307983+0.15, 81.040, 83.972, 11.606]
        first_table=0
        self.arm_run(pose_target=home_position)
        
        # 去柜子

        # navigator.goto("9")
        # follower.xuanzhuan("9")

        for i in range(4):
            self.double = 0
            position = str(i)
            #运行到指定的桌子前面
            navigator.goto(position)
            follower.xuanzhuan(position)
            
            #执行检测
            results_detect = self.detector.detect(self.kinectcamera, pattern='find',target='bottle',depth=True,range=1.0,rotate=0.08)

            #采集货架第一、二层的位姿
            store_first_position=[0.44204944372177154,0.0050716763362288475,0.6999208331108093+0.0155,81.040, 83.972, 11.606]
            store_second_position=[0.4631043076515198,-0.06340128928422928,0.30569732189178467+0.06,81.040, 83.972, 11.606]


            #记录识别到的物体名名称
            goods=[]
            #采集准备抓取的位姿
            table_position=[]
            #记录真实的抓取位子
            coordinate=[]
            #记录抓取后的上台位子
            up_position=[]
            self.name=[]
            print(results_detect)
            # xyz1 = None

            for res in results_detect:
            #提取出物体名称，抓取前坐标，抓取坐标
                for name,xyz in res.items():
                    if name in things["heavy"]:
                        self.zhonglei="heavy"
                        pose_target_ready,pose_target_real,pose_target_up=self.kinova_coordinate(xyz)
                        #存下预备姿态
                        table_position.append(pose_target_ready)
                        #存下所有物体的真实抓取位点
                        coordinate.append(pose_target_real)
                        up_position.append(pose_target_up)
                        #存所有的物体名称
                        goods.append(name)

                    
                    # for items in floor:
                    #     if  self.zhonglei == items[0] or self.zhonglei == items[1]:
                    #         print("name:",name)
                    #         print("xyz:",xyz)
                    #         pose_target_ready,pose_target_real,pose_target_up=self.kinova_coordinate(xyz)
                    #         #存下预备姿态
                    #         table_position.append(pose_target_ready)
                    #         #存下所有物体的真实抓取位点
                    #         coordinate.append(pose_target_real)
                    #         up_position.append(pose_target_up)
                    #         #存所有的物体名称
                    #         goods.append(name)
                    #         self.double = self.double + 1
                
            
            print("所有识别到的物体")
            print(goods)
            length=len(coordinate)


            self.soundplayer.say("it is water,belong heavy")
            now=datetime.now().strftime('%H:%M:%s')
            with open('/home/zq/catkin_ws/src/cmoon/src/beijing_2025/table_to_store/src/output.txt','a') as f:
                f.write('{}:the name is water\n'.format(now))
            
            now=datetime.now().strftime('%H:%M:%s')
            # with open('/home/zq/catkin_ws/src/cmoon/src/beijing_2025/table_to_store/src/output.txt','a') as f:
            #     f.write('{}:the category is {}\n'.format(now,category_goods))
            goods_length-=1
            # print(goods[j])
        
            print("--------开始执行----------")
            kinova.finger_run(finger_target=[5,5,5])
            #抓取桌子上的物体
            self.arm_run(pose_target=table_position[0])
            # time.sleep(3)
            print("--------预备抓取---------")
            # time.sleep(5)
            self.arm_run(pose_target=coordinate[0])
            # time.sleep(3)
            
            print("--------实际抓取---------")
            # time.sleep(5)
            #关闭手指抓取
            kinova.finger_run(finger_target=[55,55,55])
            # kinova.finger_run(finger_target=[85,85,85])
            kinova.finger_run(finger_target=[90,90,90])
            output=self.execute_command("rosrun kinova_demo fingers_action_client.py -v -r j2n6s300 percent -- 0 0 0")
            if int(output.split('\n')[4][36:40])>=6800:#6800完全闭合
                continue
            kinova.arm_run(pose_target=up_position[0])
            print("----------抬高----------")
            # kinova.arm_run(pose_target=table_position[j])
            self.arm_run(pose_target=home_position_1)
            self.move_back_x(move_dis=-0.3)
            kinova.finger_run(finger_target=[5,5,5])
            # break


            # if first_table:
            #     #记录识别到的物体名名称
            #     goods=[]
            #     #采集准备抓取的位姿
            #     table_position=[]
            #     #记录真实的抓取位子
            #     coordinate=[]
            #     #记录抓取后的上台位子
            #     up_position=[]
            #     # navigator.goto("9")
            #     # follower.xuanzhuan("9")
            #     results_detect = self.detector.detect(self.kinectcamera, pattern='find',target='bottle',depth=True,range=0.8,rotate=0.08)
            #     for res in results_detect:
            #     #提取出物体名称，抓取前坐标，抓取坐标
            #         for name,xyz in res.items():
            #             print("name:",name)
            #             print("xyz:",xyz)
            #             pose_target_ready,pose_target_real,pose_target_up=self.kinova_coordinate(xyz)
            #             #存下预备姿态
            #             table_position.append(pose_target_ready)
            #             #存下所有物体的真实抓取位点
            #             coordinate.append(pose_target_real)
            #             up_position.append(pose_target_up)
            #             #存所有的物体名称
            #             goods.append(name)
            #     now_length=len(coordinate)
            #     first_table=1
            #     category_now=[]
            #     flag=True
                #打开手指
                
                                



                        

                # self.arm_run(pose_target=home_position)
                
                # #判断桌子上的物体属于哪一类
                # # for category,list in things.items():
                # #     # print("判断物体属于哪一类")
                # #     print(goods[temp])



                # if goods[temp] in list:
                #     category_now=category
                #     for items in floor:
                #         if items[0] == category_now:
                #             pose_target_realtime = store_first_position
                #             self.soundplayer.say(items[0],0)
                #             now=datetime.now().strftime('%H:%M:%s')
                #             with open('/home/zq/catkin_ws/src/cmoon/src/beijing_2025/table_to_store/src/output.txt','a') as f:
                #                 f.write('{}:fisrt floor\n'.format(now))
                #         elif items[1] == category_now:
                #             pose_target_realtime = store_second_position
                #             self.soundplayer.say(items[1],0)
                #             now=datetime.now().strftime('%H:%M:%s')
                #             with open('/home/zq/catkin_ws/src/cmoon/src/beijing_2025/table_to_store/src/output.txt','a') as f:
                #                 f.write('{}:second floor\n'.format(now))

                #         #先移动到准备位置
                #         self.arm_run(pose_target=pose_target_realtime)
                #         #打开手指
                #         kinova.finger_run(finger_target=[5,5,5])
                #         # self.arm_run(pose_target=store_first_position)
                #         self.move_back_x(move_dis=-0.3)
                #         self.arm_run(pose_target=home_position)
                #         self.double = self.double-1



                # if self.double > 0:
                #     i = i-1
                #     break
                # else:
                #     break


if __name__ == "__main__":
    kinova = KinovaRobot("j2n6s300")
    # rospy.timer.sleep(3)
    kinova.table_store()
    # kinova.catch_bag()
