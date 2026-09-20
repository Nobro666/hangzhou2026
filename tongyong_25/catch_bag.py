#!/usr/bin/env python3

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import numpy as np
import math
import time

from realsense_yolo11 import RealSenseYolo11Detector

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
import std_srvs.srv


class KinovaRobot:
    def __init__(self,kinova_robotType) -> None:
        rospy.init_node(kinova_robotType)
        rospy.loginfo("init_node successfully")
        self.kinova_robotType = kinova_robotType
        self.prefix = self.kinova_robotType + "_"

        robot_category = kinova_robotType[0]
        robot_category_version = int(kinova_robotType[1])
        wrist_type = kinova_robotType[2]
        self.arm_joint_number = int(kinova_robotType[3])
        self.finger_number = int(kinova_robotType[5])
        self.finger_maxDist = 18.9/2/1000
        self.finger_maxTurn = 6800
        self.currentFingerPosition = [0.0, 0.0, 0.0]
        self.currentCartesianCommand = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 1.6373136043548584, 1.1021580696105957, 0.5095799565315247]
        self.homePositionMdeg = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
        self.getcurrentCartesianCommand()
        self.action_address_arm = '/' + self.prefix + 'driver/pose_action/tool_pose'
        self.client_arm = actionlib.SimpleActionClient(self.action_address_arm, kinova_msgs.msg.ArmPoseAction)
        self.client_arm.wait_for_server()
        self.goal_arm = kinova_msgs.msg.ArmPoseGoal()
        self.goal_arm.pose.header = std_msgs.msg.Header(frame_id=(self.prefix + 'link_base'))
        rospy.loginfo("arm service connect successfully")

        self.getCurrentFingerPosition()
        self.action_address_finger = '/' + self.prefix + 'driver/fingers_action/finger_positions'
        self.client_finger = actionlib.SimpleActionClient(self.action_address_finger, kinova_msgs.msg.SetFingersPositionAction)
        self.client_finger.wait_for_server()
        self.goal_finger = kinova_msgs.msg.SetFingersPositionGoal()
        rospy.loginfo("finger service connect successfully")
        self.finger_run(finger_target=[5,5,5])

        self.arm_run(pose_target=self.homePositionMdeg)
        home_position= [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
        qw=0.3053669885856701
        qx=-0.5314301858623786
        qy=0.6901383166981908
        qz= -0.38476234699017126
        tx= -0.11921030589536183
        ty= -0.024341172016762613
        tz= 1.083284416786497

        R = self.quaternion_to_rotation_matrix(qw,qx,qy,qz)
        self.kinectA2kinova_matrix = np.eye(4)
        self.kinectA2kinova_matrix[:3, :3] = R
        self.kinectA2kinova_matrix[:3, 3] = [tx, ty, tz]
        
        self.kinectA2kinova_matrix = np.array([[-0.09811982, -0.50830614,  0.85556845, -0.11490607],
                                                [-0.98824356, -0.05152143, -0.14394517, -0.00222445],
                                                [ 0.11724832, -0.85963388, -0.49727499,  0.4],
                                                [ 0. ,         0.  ,        0.   ,       1.        ]])

        self.realsense2kinova_matrix = np.array([
                                                [-0.81552114, -0.02233797, -0.57829602,  0.2],
                                                [ 0.06292209, -0.99675352, -0.05023179, 0.0],
                                                [-0.57529651, -0.07735268,  0.81427912,  0.4],
                                                [ 0.        ,  0.        ,  0.        ,  1.        ]
                                                ])

    def quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
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
                self.positions = []
                exit()
            else:
                positions_temp1 = [max(0.0, n) for n in finger_turn]
                positions_temp2 = [min(n, self.finger_maxTurn) for n in positions_temp1]
                self.positions = [float(n) for n in positions_temp2]

            print('Sending finger position ...')
            result = self.gripper_client(self.positions)
            print('Finger position sent!')

        except rospy.ROSInterruptException:
            print('program interrupted before completion')

    def cartesian_pose_client(self, position, orientation):
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
        self.goal_finger.fingers.finger1 = float(finger_positions[0])
        self.goal_finger.fingers.finger2 = float(finger_positions[1])
        if len(finger_positions) < 3:
            self.goal_finger.fingers.finger3 = 0.0
        else:
            self.goal_finger.fingers.finger3 = float(finger_positions[2])
        self.client_finger.send_goal(self.goal_finger)

        if self.client_finger.wait_for_result(rospy.Duration(5.0)):
            return self.client_finger.get_result()
        else:
            self.client_finger.cancel_all_goals()
            rospy.logwarn('the gripper action timed-out')
            return None
        
    def unitParser_arm(self, unit_, pose_value_, relative_):
        position_ = pose_value_[:3]
        orientation_ = pose_value_[3:]
        print(f"position_:{position_}, orientation_:{orientation_}")

        for i in range(0,3):
            if relative_:
                position_[i] = pose_value_[i] + self.currentCartesianCommand[i]
            else:
                position_[i] = pose_value_[i]

        if unit_ == 'mq':
            if relative_:
                orientation_XYZ = self.Quaternion2EulerXYZ(orientation_)
                orientation_xyz_list = [orientation_XYZ[i] + self.currentCartesianCommand[3+i] for i in range(0,3)]
                orientation_q = self.EulerXYZ2Quaternion(orientation_xyz_list)
            else:
                orientation_q = orientation_

            orientation_rad = self.Quaternion2EulerXYZ(orientation_q)
            orientation_deg = list(map(math.degrees, orientation_rad))

        elif unit_ == 'mdeg':
            if relative_:
                orientation_deg_list = list(map(math.degrees, self.currentCartesianCommand[3:]))
                orientation_deg = [orientation_[i] + orientation_deg_list[i] for i in range(0,3)]
            else:
                orientation_deg = orientation_

            orientation_rad = list(map(math.radians, orientation_deg))
            orientation_q = self.EulerXYZ2Quaternion(orientation_rad)

        elif unit_ == 'mrad':
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

        return pose_mq_, pose_mdeg_, pose_mrad_
        
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
        if unit_ == 'turn':
            if relative_:
                finger_turn_absolute_ = [finger_value_[i] + self.currentFingerPosition[i] for i in range(0, len(finger_value_))]
            else:
                finger_turn_absolute_ = finger_value_

            finger_turn_ = finger_turn_absolute_
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]

        elif unit_ == 'mm':
            finger_turn_command = [x/1000 * self.finger_maxTurn / self.finger_maxDist for x in finger_value_]
            if relative_:
                finger_turn_absolute_ = [finger_turn_command[i] + self.currentFingerPosition[i] for i in range(0, len(finger_value_))]
            else:
                finger_turn_absolute_ = finger_turn_command

            finger_turn_ = finger_turn_absolute_
            finger_meter_ = [x * self.finger_maxDist / self.finger_maxTurn for x in finger_turn_]
            finger_percent_ = [x / self.finger_maxTurn * 100.0 for x in finger_turn_]
        elif unit_ == 'percent':
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
        topic_address = '/' + self.prefix + 'driver/out/cartesian_command'
        rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, self.setcurrentCartesianCommand)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
        print('position listener obtained message for Cartesian pose. ')

    def setcurrentCartesianCommand(self, feedback):
        currentCartesianCommand_str_list = str(feedback).split("\n")

        for index in range(0,len(currentCartesianCommand_str_list)):
            temp_str=currentCartesianCommand_str_list[index].split(": ")
            self.currentCartesianCommand[index] = float(temp_str[1])

    def getCurrentFingerPosition(self):
        topic_address = '/' + self.prefix + 'driver/out/finger_position'
        rospy.Subscriber(topic_address, kinova_msgs.msg.FingerPosition, self.setCurrentFingerPosition)
        rospy.wait_for_message(topic_address, kinova_msgs.msg.FingerPosition)
        print('obtained current finger position ')

    def setCurrentFingerPosition(self,feedback):
        self.currentFingerPosition[0] = feedback.finger1
        self.currentFingerPosition[1] = feedback.finger2
        self.currentFingerPosition[2] = feedback.finger3

    def transform(self, position, camera_type='realsense'):
        pos_camera = np.array([position[0], position[1], position[2], 1.0])
        if camera_type == 'realsense':
            position_end_effector = self.realsense2kinova_matrix.dot(pos_camera)
        else:
            position_end_effector = self.kinectA2kinova_matrix.dot(pos_camera)
        position_end_effector = position_end_effector[:3]
        print("position_end_effector: ",position_end_effector)
        return position_end_effector
    
    # def image_to_arm(self, camera_x, camera_y, camera_z):
    #     import numpy as np
        
    #     tx = 0.10
    #     ty = 0.00
    #     tz = -0.05
        
    #     R = np.array([
    #         [1.0, 0.0, 0.0],
    #         [0.0, 1.0, 0.0],
    #         [0.0, 0.0, 1.0]
    #     ])
        
    #     P_camera = np.array([[camera_x], [camera_y], [camera_z]])
    #     P_end = np.dot(R, P_camera) + np.array([[tx], [ty], [tz]])
        
    #     return P_end.flatten()
    
    def end_to_base(self, point_in_end):
        import numpy as np
        import math
        
        current_pose = self.currentCartesianCommand
        
        x, y, z = current_pose[0], current_pose[1], current_pose[2]
        
        rx = math.radians(current_pose[3])
        ry = math.radians(current_pose[4])
        rz = math.radians(current_pose[5])
        
        R = self.euler_to_rotation_matrix(rx, ry, rz)
        
        t = np.array([[x], [y], [z]])
        
        point_in_end_array = np.array(point_in_end).reshape(3, 1)
        point_in_base = np.dot(R, point_in_end_array) + t
        
        return point_in_base.flatten()
    
    def euler_to_rotation_matrix(self, rx, ry, rz):
        import numpy as np
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        return np.dot(Rz, np.dot(Ry, Rx))
    
    def image_to_arm(self,camera_x,camera_y,camera_z):
        x = 0.03081667772927059
        y = 0.05788506740074149
        z = -0.11464690917725247

        translation = np.array([[x], [y], [z]])

        q0 = 0.5012811368530545
        q1 = -0.4941087597350034
        q2 = -0.5089502424057524
        q3 = -0.4952513863903056

        R = np.array([[-0.02846062, 0.97974337, -0.04101445],
              [-0.05355864, 0.04071949, 0.97922196],
              [0.97940094, 0.02915925, 0.05337473]])

        R_1 = np.array([[1-2*q2*q2-2*q3*q3, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2],  
                            [2*q1*q2+2*q0*q3, 1-2*q1*q1-2*q3*q3, 2*q2*q3-2*q0*q1],  
                            [2*q1*q3-2*q0*q2, 2*q2*q2+2*q0*q1, 1-2*q1*q1-2*q2*q2]])
        P_camera = np.array([[camera_x], [camera_y], [camera_z]])
        P_robot = np.dot(R, P_camera) + translation 
        return P_robot

    def verboseParser(self, verbose=False):
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


def put_bag():
    kinova = KinovaRobot("j2n6s300")
    home_position = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
    wait_position = [0.35, -0.20, 0.55, 81.040, 83.972, 11.606]
    ground_position=[0.5402693748474121,-0.05474631115794182,0.0748402550816536,81.040,83.972,11.606]
    
    rospy.sleep(2)
    
    rospy.loginfo("移动到等待位置...")
    kinova.arm_run(pose_target=wait_position)
    rospy.sleep(2)

    kinova.arm_run(pose_target=ground_position)
    rospy.sleep(2)

    kinova.finger_run(finger_target=[5,5,5])
    rospy.sleep(1)

    kinova.arm_run(pose_target=home_position)
    rospy.sleep(2)
    

    # rospy.loginfo("初始化相机...")
    # from realsense_yolo11 import Path
    # detector = RealSenseYolo11Detector(weights=Path('/home/cqr/catkin_ws/src/test_1/scripts/model/yolo11n.pt'))
    
    # rospy.loginfo(f"等待检测目标: {target}...")
    
    # result = None
    # max_attempts = 50
    # for attempt in range(max_attempts):
    #     rospy.loginfo(f"检测尝试 {attempt+1}/{max_attempts}...")
    #     result = detector.detect_targets(target_items=[target], max_retry=1, show_window=False)
    #     if result is not None:
    #         rospy.loginfo(f"检测到目标！")
    #         break
    #     rospy.sleep(1)
    
    # if result is None:
    #     rospy.logwarn(f"未检测到目标: {target}")
    #     return
    
    # rospy.loginfo(f"检测到 {result.name}，相机坐标: x={result.x:.3f}, y={result.y:.3f}, z={result.z:.3f}")
    
    # xyz = [result.x, result.y, result.z]
    
    # if xyz[2] == 0 or xyz[2] < 0.1 or xyz[2] > 2.0:
    #     rospy.logwarn(f"深度数据异常 (z={xyz[2]:.3f})，使用估计深度 0.5m")
    #     xyz = [xyz[0], xyz[1], 0.5]
    
    # point_in_end = kinova.image_to_arm(xyz[0], xyz[1], xyz[2])
    # rospy.loginfo(f"末端坐标: x={point_in_end[0]:.3f}, y={point_in_end[1]:.3f}, z={point_in_end[2]:.3f}")
    
    # point_in_base = kinova.end_to_base(point_in_end)
    # rospy.loginfo(f"基座坐标: x={point_in_base[0]:.3f}, y={point_in_base[1]:.3f}, z={point_in_base[2]:.3f}")
    
    # if not (0.2 < point_in_base[2] < 0.8):
    #     rospy.logwarn(f"目标Z坐标 {point_in_base[2]:.3f} 超出安全范围，使用默认值")
    #     point_in_base[2] = 0.5
    
    # forward_distance = 0.15
    # grasp_x = point_in_base[0] + forward_distance
    # grasp_y = point_in_base[1]
    # grasp_z = point_in_base[2]
    
    # forward_pose = [grasp_x, grasp_y, grasp_z, 81.040, 83.972, 11.606]
    
    # rospy.loginfo(f"向前抓取位置: {forward_pose[:3]}")
    # rospy.loginfo("向前移动抓取...")
    # kinova.arm_run(pose_target=forward_pose)
    # rospy.sleep(1.5)
    
    # rospy.loginfo("闭合手指...")
    # kinova.finger_run(finger_target=[85, 85, 85])
    # rospy.sleep(0.5)
    # kinova.finger_run(finger_target=[95, 95, 95])
    # rospy.sleep(1)
    
    # rospy.loginfo("后退回等待位置...")
    # kinova.arm_run(pose_target=wait_position)
    # rospy.sleep(2)
    
    # rospy.loginfo("返回 home 位置...")
    # kinova.arm_run(pose_target=home_position)
    # rospy.sleep(2)
    
    # rospy.loginfo("抓取完成！")
    
def catch_door():
    kinova = KinovaRobot("j2n6s300")
    home_position = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
    wait_position = [0.35, -0.20, 0.55, 81.040, 83.972, 11.606]
    forward_position=[0.37738725543022156,-0.24977520108222961,0.6328005194664001,112.25,125.38,-15.69]
    back_position=[0.17738725543022156,-0.24977520108222961,0.6328005194664001,112.25,125.38,-15.69]
    
    rospy.sleep(2)
    
    rospy.loginfo("移动到等待位置...")
    kinova.arm_run(pose_target=wait_position)
    rospy.sleep(2)

    kinova.arm_run(pose_target=forward_position)
    rospy.sleep(2)

    rospy.loginfo("闭合手指...")
    kinova.finger_run(finger_target=[85, 85, 85])
    rospy.sleep(0.5)
    kinova.finger_run(finger_target=[95, 95, 95])
    rospy.sleep(1)
    
    kinova.arm_run(pose_target=back_position)
    rospy.sleep(2)

    kinova.finger_run(finger_target=[5,5,5])
    rospy.sleep(1)

    kinova.arm_run(pose_target=home_position)
    rospy.sleep(2)

    # rospy.loginfo("初始化相机...")
    # from realsense_yolo11 import Path
    # detector = RealSenseYolo11Detector(weights=Path('/home/cqr/catkin_ws/src/test_1/scripts/model/yolo11n.pt'))
    
    # rospy.loginfo(f"等待检测目标: {target}...")
    
    # result = None
    # max_attempts = 50
    # for attempt in range(max_attempts):
    #     rospy.loginfo(f"检测尝试 {attempt+1}/{max_attempts}...")
    #     result = detector.detect_targets(target_items=[target], max_retry=1, show_window=False)
    #     if result is not None:
    #         rospy.loginfo(f"检测到目标！")
    #         break
    #     rospy.sleep(1)
    
    # if result is None:
    #     rospy.logwarn(f"未检测到目标: {target}")
    #     return
    
    # rospy.loginfo(f"检测到 {result.name}，相机坐标: x={result.x:.3f}, y={result.y:.3f}, z={result.z:.3f}")
    
    # xyz = [result.x, result.y, result.z]
    
    # if xyz[2] == 0 or xyz[2] < 0.1 or xyz[2] > 2.0:
    #     rospy.logwarn(f"深度数据异常 (z={xyz[2]:.3f})，使用估计深度 0.5m")
    #     xyz = [xyz[0], xyz[1], 0.5]
    
    # point_in_end = kinova.image_to_arm(xyz[0], xyz[1], xyz[2])
    # rospy.loginfo(f"末端坐标: x={point_in_end[0]:.3f}, y={point_in_end[1]:.3f}, z={point_in_end[2]:.3f}")
    
    # point_in_base = kinova.end_to_base(point_in_end)
    # rospy.loginfo(f"基座坐标: x={point_in_base[0]:.3f}, y={point_in_base[1]:.3f}, z={point_in_base[2]:.3f}")
    
    # if not (0.2 < point_in_base[2] < 0.8):
    #     rospy.logwarn(f"目标Z坐标 {point_in_base[2]:.3f} 超出安全范围，使用默认值")
    #     point_in_base[2] = 0.5
    
    # forward_distance = 0.15
    # grasp_x = point_in_base[0] + forward_distance
    # grasp_y = point_in_base[1]
    # grasp_z = point_in_base[2]
    
    # forward_pose = [grasp_x, grasp_y, grasp_z, 81.040, 83.972, 11.606]
    
    # rospy.loginfo(f"向前抓取位置: {forward_pose[:3]}")
    # rospy.loginfo("向前移动抓取...")
    # kinova.arm_run(pose_target=forward_pose)
    # rospy.sleep(1.5)
    
    # rospy.loginfo("闭合手指...")
    # kinova.finger_run(finger_target=[85, 85, 85])
    # rospy.sleep(0.5)
    # kinova.finger_run(finger_target=[95, 95, 95])
    # rospy.sleep(1)
    
    # rospy.loginfo("向前推开门...")
    # kinova.arm_run(pose_target=forward_pose)
    # rospy.sleep(2)
    # rospy.loginfo("开门完成！")


def catch_bag(target='bottle'):
    kinova = KinovaRobot("j2n6s300")
    home_position = [0.2104809731245041, -0.25873029232025146, 0.5095799565315247, 81.040, 83.972, 11.606]
    wait_position = [0.35, -0.20, 0.55, 81.040, 83.972, 11.606]
    
    rospy.sleep(2)
    
    rospy.loginfo("移动到等待位置...")
    kinova.arm_run(pose_target=wait_position)
    rospy.sleep(2)
    
    rospy.loginfo("初始化相机...")
    from realsense_yolo11 import Path
    detector = RealSenseYolo11Detector(weights=Path('/home/cqr/catkin_ws/src/test_1/scripts/model/yolo11m.pt'))
    
    rospy.loginfo(f"等待检测目标: {target}...")
    
    result = None
    max_attempts = 50
    for attempt in range(max_attempts):
        rospy.loginfo(f"检测尝试 {attempt+1}/{max_attempts}...")
        result = detector.detect_targets(target_items=[target], max_retry=1, show_window=False)
        if result is not None:
            rospy.loginfo(f"检测到目标！")
            break
        rospy.sleep(1)
    
    if result is None:
        rospy.logwarn(f"未检测到目标: {target}")
        return
    
    rospy.loginfo(f"检测到 {result.name}，相机坐标: x={result.x:.3f}, y={result.y:.3f}, z={result.z:.3f}")
    
    xyz = [result.x, result.y, result.z]
    
    if xyz[2] == 0 or xyz[2] < 0.1 or xyz[2] > 2.0:
        rospy.logwarn(f"深度数据异常 (z={xyz[2]:.3f})，使用估计深度 0.5m")
        xyz = [xyz[0], xyz[1], 0.5]
    
    point_in_end = kinova.image_to_arm(xyz[0], xyz[1], xyz[2])
    rospy.loginfo(f"末端坐标: x={point_in_end[0]:.3f}, y={point_in_end[1]:.3f}, z={point_in_end[2]:.3f}")
    
    point_in_base = kinova.end_to_base(point_in_end)
    rospy.loginfo(f"基座坐标: x={point_in_base[0]:.3f}, y={point_in_base[1]:.3f}, z={point_in_base[2]:.3f}")
    
    if not (0.2 < point_in_base[2] < 0.8):
        rospy.logwarn(f"目标Z坐标 {point_in_base[2]:.3f} 超出安全范围，使用默认值")
        point_in_base[2] = 0.5
    
    forward_distance = 0.15
    grasp_x = point_in_base[0] + forward_distance
    grasp_y = point_in_base[1]
    grasp_z = point_in_base[2]
    
    
    pre_grasp_pose = [grasp_x, grasp_y, grasp_z + 0.05, 81.040, 83.972, 11.606]#预抓取位置
    grasp_pose = [grasp_x, grasp_y, grasp_z, 81.040, 83.972, 11.606]#抓取位置
    lift_pose = [grasp_x, grasp_y, grasp_z + 0.15, 81.040, 83.972, 11.606]#上移量

    rospy.loginfo("上移")
    kinova.arm_run(pose_target=lift_pose)
    rospy.sleep(1)
    
    rospy.loginfo(f"抓取位置: {grasp_pose[:3]}")
    rospy.loginfo("向前移动抓取...")
    kinova.arm_run(pose_target=grasp_pose)
    rospy.sleep(1.5)
    
    rospy.loginfo("闭合手指...")
    kinova.finger_run(finger_target=[85, 85, 85])
    rospy.sleep(0.5)
    kinova.finger_run(finger_target=[95, 95, 95])
    rospy.sleep(1)
    
    rospy.loginfo("后退回等待位置...")
    kinova.arm_run(pose_target=wait_position)
    rospy.sleep(2)
    
    rospy.loginfo("返回 home 位置...")
    kinova.arm_run(pose_target=home_position)
    rospy.sleep(2)
    
    rospy.loginfo("抓取完成！")

if __name__ == "__main__":
    put_bag()