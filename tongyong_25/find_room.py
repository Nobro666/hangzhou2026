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
        approach_success = self.approach_owner(person_result["camera_coords"])
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