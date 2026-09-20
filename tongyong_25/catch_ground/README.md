# Arm Catch based on YOLO detect

Ubuntu20.04+ROS-Noetic

Python>=3.8

Install YOLO11

https://github.com/ultralytics/ultralytics

```bash
pip install ultralytics==8.3.54
```

Install API for Kinect Azure DK and Intel RealSense:

```bash
pip install pykinect_azure==0.0.4 pyrealsense2==2.55.1.6486
```

ROS package for kinova arm:

```bash
cd ~/catkin_ws/src
git clone -b noetic-devel git@github.com:Kinovarobotics/kinova-ros.git kinova-ros
cd ~/catkin_ws
catkin_make
```

How to use:

**detector:**

```python
from detector import Detector

if __name__ == "__main__":
    detector = Detector()
    results = detector.detect(camera='k4a',pattern='find',target='bottle',depth=True,range=0.8)
    print("len results:", len(results))
    if len(results)>0:
        (name, point), = results[0].items()
        print(f"Class:{name}, Point:{point}")
```

**arm-catch:**

```python
from kinovarobot import KinovaRobot
from detector import Detector

if __name__ == "__main__":
    kinova = KinovaRobot("j2n6s300")
    rospy.timer.sleep(3)
    detector = Detector()
    result = detector.detect(camera='k4a',pattern='find',target='bottle',depth=True,range=0.9)
    for name, xyz in result[0].items():
        print(f"xyz: {xyz}")
        # xyz[2]
        xyz_arm = kinova.transform(xyz)
        pose_target_1 = [xyz_arm[0],xyz_arm[1],xyz_arm[2]+0.2]
        pose_target_2 = [xyz_arm[0],xyz_arm[1],xyz_arm[2]]
        # 81.040, ty 83.972, tz 11.606
        pose_target_1.extend([81.040, 83.972, 11.606])
        print("pose_target_1: ",pose_target_1)
        kinova.arm_run(pose_target=pose_target_1)
        # time.sleep(2)
        pose_target_2.extend([81.040, 83.972, 11.606])
        print("pose_target_2: ",pose_target_2)
        kinova.arm_run(pose_target=pose_target_2)
        time.sleep(2)
        kinova.finger_run(finger_target=[95,95,95])
```

