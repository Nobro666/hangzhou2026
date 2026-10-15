#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""只测试 jujia26.py 中的固定导航点，不启动相机、机械臂或语音。

用法：
    python3 test_navigation.py --list
    python3 test_navigation.py
    python3 test_navigation.py start room0 room1_door room1 over
"""

import argparse
import ast
import math
from pathlib import Path


def load_locations():
    """读取 LOCATION 字面量，避免导入主程序时初始化其它模块。"""
    source = Path(__file__).with_name("jujia26.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "LOCATION"
               for target in node.targets):
            locations = ast.literal_eval(node.value)
            if not isinstance(locations, dict):
                raise ValueError("jujia26.py 中的 LOCATION 必须是字典")
            return locations
    raise ValueError("jujia26.py 中没有找到 LOCATION")


def validate_point(name, point):
    """空点位返回 False；格式错误则明确报错，不发送导航目标。"""
    if not point:
        return False
    if (not isinstance(point, (list, tuple)) or len(point) != 2 or
            not isinstance(point[0], (list, tuple)) or len(point[0]) != 3 or
            not isinstance(point[1], (list, tuple)) or len(point[1]) != 4):
        raise ValueError("{} 的坐标格式应为 [[x,y,z],[qx,qy,qz,qw]]".format(name))
    values = tuple(point[0]) + tuple(point[1])
    if not all(isinstance(value, (int, float)) and math.isfinite(value)
               for value in values):
        raise ValueError("{} 的坐标必须是有限数字".format(name))
    if math.isclose(sum(value * value for value in point[1]), 0.0):
        raise ValueError("{} 的方向四元数不能全为 0".format(name))
    return True


def make_goal(point, rospy, MoveBaseGoal):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    position = goal.target_pose.pose.position
    position.x, position.y, position.z = point[0]
    orientation = goal.target_pose.pose.orientation
    orientation.x, orientation.y, orientation.z, orientation.w = point[1]
    return goal


def main():
    parser = argparse.ArgumentParser(description="逐点测试 jujia26.py 的固定导航点")
    parser.add_argument("points", nargs="*", help="指定要测试的点名；省略时测试所有已填写点位")
    parser.add_argument("--list", action="store_true", help="只列出点位，不连接 ROS 或移动底盘")
    parser.add_argument("--continue-on-failure", action="store_true", help="某点失败后继续测试下一点")
    parser.add_argument("--server-timeout", type=float, default=7.0, help="等待 move_base 的秒数")
    parser.add_argument("--goal-timeout", type=float, default=40.0, help="每个点的导航超时秒数")
    args = parser.parse_args()

    if args.server_timeout <= 0 or args.goal_timeout <= 0:
        parser.error("超时时间必须大于 0")

    locations = load_locations()
    if args.list:
        for name, point in locations.items():
            status = "已填写" if validate_point(name, point) else "未填写，运行时跳过"
            print("{}: {} {}".format(name, status, point))
        return 0

    unknown = [name for name in args.points if name not in locations]
    if unknown:
        parser.error("未知点位：{}".format(", ".join(unknown)))

    # 默认按 LOCATION 顺序测试；离场点统一放在最后。
    names = args.points or ([name for name in locations if name != "over"] +
                            (["over"] if "over" in locations else []))
    selected = []
    for name in names:
        if validate_point(name, locations[name]):
            selected.append(name)
        else:
            print("跳过未填写的点位：{}".format(name))
    if not selected:
        print("没有可测试的导航点")
        return 1

    print("导航顺序：{}".format(" -> ".join(selected)))

    import actionlib
    import rospy
    from actionlib_msgs.msg import GoalStatus
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

    rospy.init_node("test_navigation", anonymous=True)
    client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    rospy.on_shutdown(client.cancel_all_goals)
    if not client.wait_for_server(rospy.Duration(args.server_timeout)):
        rospy.logerr("等待 move_base 超时（%.1f 秒）", args.server_timeout)
        return 1

    failed = []
    for index, name in enumerate(selected, 1):
        if rospy.is_shutdown():
            return 1
        print("[{}/{}] 前往 {}".format(index, len(selected), name))
        client.send_goal(make_goal(locations[name], rospy, MoveBaseGoal))
        if not client.wait_for_result(rospy.Duration(args.goal_timeout)):
            client.cancel_goal()
            client.wait_for_result(rospy.Duration(2.0))
            print("{}：导航超时".format(name))
            failed.append(name)
        elif client.get_state() == GoalStatus.SUCCEEDED:
            print("{}：到达成功".format(name))
        else:
            print("{}：导航失败，状态码={}".format(name, client.get_state()))
            failed.append(name)

        if failed and not args.continue_on_failure:
            break

    if failed:
        print("未到达的点位：{}".format(", ".join(failed)))
        return 1
    print("所有选定点位导航完成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("导航测试已中断")
        raise SystemExit(130)
