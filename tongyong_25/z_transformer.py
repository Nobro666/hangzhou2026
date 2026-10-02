import math

"""
为应对tf2坐标变换z轴数值异常 进行手动计算 判断人是躺还是睡觉

created by zx 2025-10-12
"""
def transform_camera_to_map_z(point_camera):
    """
    将相机坐标系下的一个点转换到地图坐标系 但只计算其Z轴坐标 高度

    Args:
        point_camera (list or tuple): 在相机坐标系下的点 [x, y, z]，单位为米。
                                       - x: 相机右方
                                       - y: 相机下方
                                       - z: 相机前方

    Returns:
        float: 该点在地图坐标系下的Z轴坐标 高度 单位为米。
    """
    camera_pitch_degrees = -10.0

    camera_height_on_chassis = 1.35
    chassis_height_on_map = 0.138

    camera_pitch_radians = math.radians(camera_pitch_degrees)

    _, y_cam, z_cam = point_camera

    """
    根据理论推导的公式: z_rotated = z_cam * sin(θ) - y_cam * cos(θ)
    θ = -10度
    z_rotated = z_cam * sin(-10°) - y_cam * cos(-10°)
    """
    z_rotated_component = z_cam * math.sin(camera_pitch_radians) - y_cam * math.cos(camera_pitch_radians)

    z_map = z_rotated_component + camera_height_on_chassis + chassis_height_on_map

    return z_map
