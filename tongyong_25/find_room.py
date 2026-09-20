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

    self.speak.speak("开始巡游房间")

    room_results = []

    for room_index in range(4):
        room_name = "room" + str(room_index)

        print("--------------------------------")
        print(f">>> 正在巡游房间：{room_name}")
        print("--------------------------------")

        # 使用 navigator.py
        self.navigator.goto(room_name)
        time.sleep(1)

        # 使用 detect_people.py
        person_result = self.find_owner_in_room(room_name)
        if person_result is None:
            print(f"{room_name}中没有找到主人")
            continue

        # 使用 camera_to_map.py 和 position_last_second.py
        approach_success = self.approach_owner(person_result["map_coords"])
        if not approach_success:
            print(f"无法接近{room_name}中的人物")
            continue

        # 使用 face_detect_tongyong.py
        owner_result = self.recognize_owner()
        if owner_result is None:
            print(f"{room_name}中的人物不是已注册主人")
            self.speak.speak("没有识别出主人")
            continue

        face_id = owner_result["face_id"]
        owner_index = owner_result["owner_index"]
        person_name = owner_result["person_name"]

        # 避免重复识别同一个主人
        if face_id in self.recognized_owner_ids:
            print(f"{person_name}已经完成识别，跳过")
            continue

        self.speak.speak(f"识别到主人{owner_index}，{person_name}")

        # 行为识别后续再实现
        behavior = self.recognize_behavior(face_id , person_name)                                          # 未实现

        print("--------------------------------")
        print("主人识别结果")
        print(f"房间：{room_name}")
        print(f"主人编号：{owner_index}")
        print(f"人脸ID：{face_id}")
        print(f"姓名：{person_name}")
        print(f"行为：{behavior}")
        print("--------------------------------")

        self.speak.speak(f"{person_name}的行为是{behavior}")

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

        self.owner_observations[face_id] = observation

        self.recognized_owner_ids.add(face_id)

        room_results.append(observation)
        # 三位主人都找到后结束巡游
        if len(self.recognized_owner_ids) >= 3:
            print("三位主人均已找到")
            break

    print("巡游房间结束")
    print(f"共找到主人数量："f"{len(self.recognized_owner_ids)}")

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

    print(f">>> 准备识别{person_name}的行为")
    behavior = "未知行为"                       # 未实现
    return behavior